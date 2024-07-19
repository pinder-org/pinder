from __future__ import annotations
from functools import lru_cache

from pathlib import Path
from typing import Iterable, Optional, Union, TYPE_CHECKING

import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.atoms import AtomArray
from biotite.structure.io.pdb import PDBFile
from numpy.typing import NDArray
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pinder.core.utils import setup_logger, constants as pc
from pinder.core.structure.superimpose import superimpose_chain
from pinder.core.index.utils import set_mapping_column_types
from pinder.core.utils.dataclass import stringify_dataclass
from pinder.core.structure.atoms import (
    atom_array_from_pdb_file,
    get_seq_aligned_structures,
    get_per_chain_seq_alignments,
    invert_chain_seq_map,
    resn2seq,
    write_pdb,
)
from pinder.core.structure.contacts import get_atom_neighbors, pairwise_contacts
from pinder.core.structure import surgery


if TYPE_CHECKING:
    import torch


log = setup_logger(__name__)


def reverse_dict(mapping: dict[int, int]) -> dict[int, int]:
    return {v: k for k, v in mapping.items()}


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class Structure:
    filepath: Path
    uniprot_map: Union[Optional[Path], Optional[pd.DataFrame]] = None
    pinder_id: Optional[str] = None
    atom_array: AtomArray = None

    @staticmethod
    def read_pdb(path: Path, pdb_engine: str = "fastpdb") -> AtomArray:
        try:
            arr = atom_array_from_pdb_file(path, pdb_engine, extra_fields=["b_factor"])
        except:
            # Sometimes b-factor field parsing fails, try without it.
            arr = atom_array_from_pdb_file(path, pdb_engine)
            annotation_arr: NDArray[np.double | np.str_] = np.repeat(0.0, arr.shape[0])
            arr.set_annotation("b_factor", annotation_arr)
        return arr

    def to_pdb(self, filepath: Path | None = None) -> None:
        """Write Structure Atomarray to a PDB file.

        Parameters
        ----------
        filepath : Path | None
            Filepath to output PDB.
            If not provided, will write to self.filepath,
            potentially overwriting if the file already exists!

        Returns
        -------
        None

        """
        if not filepath:
            filepath = self.filepath
        write_pdb(self.atom_array, filepath)

    def __repr__(self) -> str:
        class_str: str = stringify_dataclass(self, 4)
        return class_str

    def __add__(self, other: Structure) -> Structure:
        combined_arr = self.atom_array + other.atom_array
        pdb_name = f"{self.filepath.stem}--{other.filepath.name}"
        combined_pdb = self.filepath.parent / pdb_name
        map_dfs = [self.uniprot_mapping, other.uniprot_mapping]
        map_dfs = [
            df for df in map_dfs if isinstance(df, pd.DataFrame) and not df.empty
        ]
        if map_dfs:
            # All-NA or category dtypes now raise FutureWarning in pd.concat
            # FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.
            # In a future version, this will no longer exclude empty or all-NA columns when determining the result dtypes.
            if len(map_dfs) == 2:
                assert isinstance(map_dfs[0], pd.DataFrame)
                assert isinstance(map_dfs[1], pd.DataFrame)
                df1: pd.DataFrame = map_dfs[0].copy()
                df2: pd.DataFrame = map_dfs[1].copy()
                uniprot_map: pd.DataFrame = pd.concat(
                    [df1.astype(df2.dtypes), df2.astype(df1.dtypes)], ignore_index=True
                )
            else:
                uniprot_map = pd.concat(map_dfs, ignore_index=True)
            uniprot_map = set_mapping_column_types(uniprot_map)
        else:
            uniprot_map = None

        return Structure(
            filepath=combined_pdb, uniprot_map=uniprot_map, atom_array=combined_arr
        )

    def filter(
        self,
        property: str,
        mask: Iterable[bool | int | str],
        copy: bool = True,
        negate: bool = False,
    ) -> Structure | None:
        atom_mask = np.isin(getattr(self.atom_array, property), list(mask))
        if negate:
            atom_mask = ~atom_mask
        if copy:
            arr = self.atom_array[atom_mask].copy()
            return Structure(
                filepath=self.filepath,
                uniprot_map=self.uniprot_mapping,
                pinder_id=self.pinder_id,
                atom_array=arr,
            )
        self.atom_array = self.atom_array[atom_mask]
        return None

    def set_chain(self, chain_id: str) -> None:
        self.atom_array.chain_id = np.repeat(chain_id, self.atom_array.shape[0])

    def get_per_chain_seq_alignments(
        self,
        other: Structure,
    ) -> dict[str, dict[int, int]]:
        self2other_seq: dict[str, dict[int, int]] = get_per_chain_seq_alignments(
            other.atom_array, self.atom_array
        )
        return self2other_seq

    def align_common_sequence(
        self,
        other: Structure,
        copy: bool = True,
        remove_differing_atoms: bool = True,
        renumber_residues: bool = False,
        remove_differing_annotations: bool = False,
    ) -> tuple[Structure, Structure]:
        ref_at = other.atom_array.copy()
        target_at = self.atom_array.copy()
        target2ref_seq = get_per_chain_seq_alignments(ref_at, target_at)
        ref2target_seq = invert_chain_seq_map(target2ref_seq)
        ref_at, target_at = get_seq_aligned_structures(ref_at, target_at)
        if remove_differing_atoms:
            # Even if atom counts are identical, annotation categories must be the same
            # First modify annotation arrays to use struc.filter_intersection,
            # then filter original structure with annotations to match res_id, res_name, atom_name
            # of intersecting structure
            ref_at_mod = ref_at.copy()
            target_at_mod = target_at.copy()
            ref_at_mod, target_at_mod = surgery.fix_annotation_mismatch(
                ref_at_mod, target_at_mod, ["element", "ins_code", "b_factor"]
            )
            ref_target_mask = struc.filter_intersection(ref_at_mod, target_at_mod)
            target_ref_mask = struc.filter_intersection(target_at_mod, ref_at_mod)
            if remove_differing_annotations:
                ref_at = ref_at_mod[ref_target_mask].copy()
                target_at = target_at_mod[target_ref_mask].copy()
            else:
                ref_at = ref_at[ref_target_mask].copy()
                target_at = target_at[target_ref_mask].copy()
        if not renumber_residues:
            target_at.res_id = np.array(
                [ref2target_seq[at.chain_id][at.res_id] for at in target_at]
            )
        if copy:
            self_struct = Structure(
                filepath=self.filepath,
                uniprot_map=self.uniprot_map,
                pinder_id=self.pinder_id,
                atom_array=target_at,
            )
            other_struct = Structure(
                filepath=other.filepath,
                uniprot_map=other.uniprot_map,
                pinder_id=other.pinder_id,
                atom_array=ref_at,
            )
            return self_struct, other_struct
        other.atom_array = ref_at
        self.atom_array = target_at
        return self, other

    def get_contacts(
        self,
        radius: float = 5.0,
        heavy_only: bool = False,
        backbone_only: bool = False,
    ) -> set[tuple[str, str, int, int]]:
        # Contacts defined as any atom within 5A of each other
        # https://pubmed.ncbi.nlm.nih.gov/12784368/
        assert len(self.chains) == 2
        R_chain = "R"
        L_chain = "L"
        arr = self.atom_array.copy()
        contacts: set[tuple[str, str, int, int]] = pairwise_contacts(
            arr,
            [R_chain],
            [L_chain],
            radius=radius,
            heavy_only=heavy_only,
            backbone_only=backbone_only,
        )
        return contacts

    def get_interface_mask(
        self,
        interface_residues: dict[str, list[int]],
        calpha_only: bool = True,
        remove_hetero: bool = False,
    ) -> NDArray[np.bool_]:
        mask = None
        arr = self.atom_array.copy()
        for ch, rlist in interface_residues.items():
            if mask is None:
                mask = np.isin(arr.chain_id, [ch]) & np.isin(arr.res_id, rlist)
            else:
                mask = mask | (np.isin(arr.chain_id, [ch]) & np.isin(arr.res_id, rlist))
        if remove_hetero:
            # Remove non-standard / hetero res
            mask = mask & (~arr.hetero)
        if calpha_only:
            mask = mask & (arr.atom_name == "CA")
        if mask is None:
            raise TypeError("Unable to create mask, no matching atoms!")
        return mask

    def get_interface_residues(
        self,
        contacts: set[tuple[str, str, int, int]] | None = None,
        radius: float = 5.0,
        heavy_only: bool = False,
        backbone_only: bool = False,
        calpha_mask: bool = True,
    ) -> dict[str, list[int]]:
        # Contacts defined as any atom within 5A of each other
        # https://pubmed.ncbi.nlm.nih.gov/12784368/
        assert len(self.chains) == 2
        R_chain = "R"
        L_chain = "L"
        arr = self.atom_array.copy()

        if not contacts:
            contacts = self.get_contacts(radius, heavy_only, backbone_only)
        # List of contacts per chain
        chain_map: dict[str, list[int]] = {ch: [] for ch in self.chains}
        for cp in contacts:
            c1, c2, r1, r2 = cp
            chain_map[c1].append(r1)
            chain_map[c2].append(r2)

        # Create set of residues in interface split into receptor and ligand
        # for bound structure
        rec_res = list(set(chain_map[R_chain]))
        lig_res = list(set(chain_map[L_chain]))
        # Make sure CA atoms present for all residues
        R_mask = np.isin(arr.chain_id, [R_chain]) & np.isin(arr.res_id, rec_res)
        L_mask = np.isin(arr.chain_id, [L_chain]) & np.isin(arr.res_id, lig_res)
        if calpha_mask:
            R_mask = R_mask & (arr.atom_name == "CA")
            L_mask = L_mask & (arr.atom_name == "CA")

        r_ca = set(arr[R_mask].res_id)
        rec_res = list(set(rec_res).intersection(r_ca))
        l_ca = set(arr[L_mask].res_id)
        lig_res = list(set(lig_res).intersection(l_ca))
        return {R_chain: rec_res, L_chain: lig_res}

    def superimpose(
        self,
        other: Structure,
    ) -> tuple[Structure, float, float]:
        # max_iterations=1 -> no outlier removal
        superimposed, _, other_anchors, self_anchors = _superimpose_common_atoms(
            other.atom_array, self.atom_array, max_iterations=1
        )
        raw_rmsd = struc.rmsd(
            other.atom_array.coord[other_anchors],
            superimposed.coord[self_anchors],
        )

        superimposed, _, other_anchors, self_anchors = _superimpose_common_atoms(
            other.atom_array, self.atom_array
        )
        refined_rmsd = struc.rmsd(
            other.atom_array.coord[other_anchors],
            superimposed.coord[self_anchors],
        )
        return (
            Structure(
                filepath=self.filepath,
                uniprot_map=self.uniprot_map,
                pinder_id=self.pinder_id,
                atom_array=superimposed,
            ),
            raw_rmsd,
            refined_rmsd,
        )

    @property
    def uniprot_mapping(self) -> pd.DataFrame | None:
        if isinstance(self.uniprot_map, Path):
            df = pd.read_parquet(self.uniprot_map)
            df = set_mapping_column_types(df)
            return df
        elif isinstance(self.uniprot_map, pd.DataFrame):
            return self.uniprot_map
        return None

    @property
    def resolved_mapping(self) -> pd.DataFrame | None:
        # Ensure mapping only contains those residues that are actually
        # present in the PDB file
        mapping = self.uniprot_mapping
        if isinstance(mapping, pd.DataFrame):
            resolved = self.residues
            mapping = mapping[~mapping.resi_uniprot.isna()]
            mapping = mapping[mapping["resi"].isin(resolved)].reset_index(drop=True)
        return mapping

    @property
    def resolved_pdb2uniprot(self) -> dict[int, int]:
        # Ensure mapping only contains those residues that are actually
        # present in the PDB file
        mapping = self.resolved_mapping
        if isinstance(mapping, pd.DataFrame):
            pdb2uni = {
                int(row["resi"]): int(row["resi_uniprot"])
                for row in mapping.to_dict(orient="records")
            }
        else:
            pdb2uni = {res: res for res in self.residues}
        return pdb2uni

    @property
    def resolved_uniprot2pdb(self) -> dict[int, int]:
        # Ensure mapping only contains those residues that are actually
        # present in the PDB file
        return reverse_dict(self.resolved_pdb2uniprot)

    @property
    def coords(self) -> NDArray[np.double]:
        coord: NDArray[np.double] = self.atom_array.coord
        return coord

    @property
    def dataframe(self) -> pd.DataFrame:
        three_to_one = pc.three_to_one_noncanonical_mapping
        return pd.DataFrame(
            [
                {
                    "chain_id": at.chain_id,
                    "res_name": at.res_name,
                    "res_code": three_to_one.get(at.res_name, "X"),
                    "res_id": at.res_id,
                    "atom_name": at.atom_name,
                    "b_factor": at.b_factor,
                    "ins_code": at.ins_code,
                    "hetero": at.hetero,
                    "element": at.element,
                    "x": at.coord[0],
                    "y": at.coord[1],
                    "z": at.coord[2],
                }
                for at in self.atom_array
            ]
        )

    @property
    def backbone_mask(self) -> NDArray[np.bool_]:
        mask: NDArray[np.bool_] = struc.filter_peptide_backbone(self.atom_array)
        return mask

    @property
    def calpha_mask(self) -> NDArray[np.bool_]:
        mask: NDArray[np.bool_] = self.atom_array.atom_name == "CA"
        return mask

    @property
    def n_atoms(self) -> int:
        n: int = self.atom_array.shape[0]
        return n

    @property
    def chains(self) -> list[str]:
        ch_list = self._attr_from_atom_array(
            self.atom_array, "chain_id", distinct=True, sort=True
        )
        return [str(ch) for ch in ch_list]

    @property
    def chain_sequence(self) -> dict[str, list[str]]:
        ch_seq: dict[str, list[str]] = (
            self.dataframe[["chain_id", "res_code", "res_name", "res_id"]]
            .drop_duplicates()
            .groupby("chain_id")["res_code"]
            .apply(list)
            .to_dict()
        )
        return ch_seq

    @property
    def sequence(self) -> str:
        numbering, resn = struc.get_residues(self.atom_array)
        seq: str = resn2seq(resn)
        return seq

    @property
    def fasta(self) -> str:
        fasta_str: str = "\n".join([f">{self.filepath.stem}", self.sequence])
        return fasta_str

    @property
    def tokenized_sequence(self) -> "torch.Tensor":
        import torch

        seq_encoding = torch.tensor([pc.AA_TO_INDEX[x] for x in self.sequence])
        tokenized: torch.Tensor = seq_encoding.long()
        return tokenized

    @property
    def residue_names(self) -> list[str]:
        res_list = self._attr_from_atom_array(
            self.atom_array, "res_name", distinct=True, sort=True
        )
        return [str(r) for r in res_list]

    @property
    def residues(self) -> list[int]:
        res_list = self._attr_from_atom_array(
            self.atom_array, "res_id", distinct=True, sort=True
        )
        return [int(r) for r in res_list]

    @property
    def atom_names(self) -> list[str]:
        at_list = self._attr_from_atom_array(
            self.atom_array, "atom_name", distinct=True, sort=True
        )
        return [str(a) for a in at_list]

    @property
    def b_factor(self) -> list[float]:
        b_factor = self._attr_from_atom_array(
            self.atom_array, "b_factor", distinct=False, sort=False
        )
        return [float(b) for b in b_factor]

    @staticmethod
    def _attr_from_atom_array(
        array: AtomArray, attr: str, distinct: bool = False, sort: bool = False
    ) -> list[str] | list[int] | list[float]:
        prop = getattr(array, attr)
        if distinct:
            prop = set(prop)
        if sort:
            prop = sorted(prop)
        return list(prop)

    def __post_init_post_parse__(self) -> None:
        if self.atom_array is None:
            self.atom_array = self.read_pdb(self.filepath)
        if not self.pinder_id:
            self.pinder_id = self.filepath.stem

    def __post_init__(self) -> None:
        # pydantic v2 renames this to dataclass post_init
        return self.__post_init_post_parse__()


def find_potential_interchain_bonded_atoms(
    structure: Structure,
    interface_res: dict[str, list[int]] | None = None,
    radius: float = 2.3,
) -> AtomArray:
    if interface_res is None:
        interface_res = structure.get_interface_residues(calpha_mask=False)
    interface_mask = structure.get_interface_mask(interface_res, calpha_only=False)
    interface = structure.atom_array[interface_mask].copy()
    assert set(interface_res.keys()) == {"R", "L"}
    interface_R = interface[(interface.chain_id == "R") & (interface.element != "H")]
    interface_L = interface[(interface.chain_id == "L") & (interface.element != "H")]
    L_neigh = get_atom_neighbors(interface_L, interface_R, radius=radius)
    R_neigh = get_atom_neighbors(interface_R, interface_L, radius=radius)
    interchain_at = R_neigh + L_neigh
    return interchain_at


def mask_common_uniprot(
    mono_A: Structure, mono_B: Structure
) -> tuple[Structure, Structure]:
    # Ensure mapping only contains those residues that are actually
    # present in the PDB file
    map_A = mono_A.resolved_mapping
    map_B = mono_B.resolved_mapping
    mask_A: list[int] | None = None
    mask_B: list[int] | None = None
    if all(isinstance(df, pd.DataFrame) for df in [map_A, map_B]):
        # Case when apo and holo uniprot mapping exists
        assert isinstance(map_A, pd.DataFrame)
        assert isinstance(map_B, pd.DataFrame)
        map_A.loc[:, "uniprot_uuid"] = (
            map_A.resi_uniprot.astype(str) + "-" + map_A.uniprot_acc.astype(str)
        )
        map_B.loc[:, "uniprot_uuid"] = (
            map_B.resi_uniprot.astype(str) + "-" + map_B.uniprot_acc.astype(str)
        )
        uniprot_mask = set(map_A.uniprot_uuid).intersection(set(map_B.uniprot_uuid))
        map_A = map_A[map_A.uniprot_uuid.isin(uniprot_mask)]
        map_B = map_B[map_B.uniprot_uuid.isin(uniprot_mask)]
        mask_A = sorted(list(set(map_A.resi.astype(int))))
        mask_B = sorted(list(set(map_B.resi.astype(int))))
    elif isinstance(map_A, pd.DataFrame):
        # Case where mono_B is predicted and already in uniprot numbering
        # Ensure uniprot used in holo mapping corresponds to predicted (could be chimeric)
        if isinstance(mono_B.pinder_id, str):
            mono_B_uniprot = mono_B.pinder_id.split("__")[1].split("-")[0]
            map_A = map_A.query(f"uniprot_acc == '{mono_B_uniprot}'").reset_index(
                drop=True
            )
        mask_B_resolved = set(map_A.resi_uniprot.astype(int)).intersection(
            set(mono_B.atom_array.res_id)
        )
        mask_B = sorted(list(mask_B_resolved))

        mask_A_resolved = set(map_A.resi.astype(int)).intersection(
            set(mono_A.atom_array.res_id)
        )
        mask_A = sorted(list(mask_A_resolved))

    elif isinstance(map_B, pd.DataFrame):
        # Case where mono_A is predicted and already in uniprot numbering
        # Ensure uniprot used in holo mapping corresponds to predicted (could be chimeric)
        if isinstance(mono_A.pinder_id, str):
            mono_A_uniprot = mono_A.pinder_id.split("__")[1].split("-")[0]
            map_B = map_B.query(f"uniprot_acc == '{mono_A_uniprot}'").reset_index(
                drop=True
            )
        mask_A_resolved = set(map_B.resi_uniprot.astype(int)).intersection(
            set(mono_A.atom_array.res_id)
        )
        mask_A = sorted(list(mask_A_resolved))
        map_B = map_B[map_B["resi_uniprot"].isin(mask_A)].reset_index(drop=True)
        mask_B_resolved = set(mono_B.atom_array.res_id).intersection(
            set(map_B.resi.astype(int))
        )
        mask_B = sorted(list(mask_B_resolved))

    if not (mask_A and mask_B):
        # Could happen if different domains are crystallized
        # Or if our mapping is incorrect
        log.error(
            "no common residues found! " f"{mono_A.pinder_id}--{mono_B.pinder_id}"
        )
        return mono_A, mono_B

    assert len(mask_A) == len(mask_B)

    mono_A_common = mono_A.filter("res_id", mask_A)
    mono_B_common = mono_B.filter("res_id", mask_B)
    assert isinstance(mono_A_common, Structure)
    assert isinstance(mono_B_common, Structure)
    return mono_A_common, mono_B_common


@lru_cache(maxsize=1)
def canonical_atom_type_mask(atom_tys: tuple[str]) -> "torch.Tensor":
    """Canonical atom masks for 21 (20 standard + missing) residue types"""
    import torch

    msk = torch.zeros(len(pc.INDEX_TO_AA_THREE), len(atom_tys))
    for aa_idx in range(len(pc.INDEX_TO_AA_THREE)):
        aa = pc.INDEX_TO_AA_THREE[aa_idx]
        for atom_idx, atom in enumerate(atom_tys):
            if atom in pc.BB_ATOMS:
                msk[aa_idx, atom_idx] = 1
            elif atom in pc.AA_TO_SC_ATOMS[aa]:
                msk[aa_idx, atom_idx] = 1
    atom_mask: torch.Tensor = msk.bool()
    return atom_mask


@lru_cache(maxsize=1)
def backbone_atom_tensor(atom_tys: tuple[str]) -> "torch.Tensor":
    import torch

    atoms: torch.Tensor = torch.tensor(
        [i for i, at in enumerate(atom_tys) if at in pc.BB_ATOMS]
    ).long()
    return atoms


def _superimpose_common_atoms(
    fixed: AtomArray, mobile: AtomArray, max_iterations: int = 10
) -> tuple[
    AtomArray,
    struc.AffineTransformation,
    NDArray[np.int_],
    NDArray[np.int_],
]:
    """Try to superimpose two structures based on homology.
    If this fails due to a lack of common anchors (e.g. in case of very short peptides),
    fall back to superimposing corresponding atoms with common atom annotations.
    """
    try:
        return_value: tuple[
            AtomArray,
            struc.AffineTransformation,
            NDArray[np.int_],
            NDArray[np.int_],
        ] = superimpose_chain(fixed, mobile, max_iterations=max_iterations)
        return return_value
    except ValueError as error:
        # Only run fallback if number of anchors is the issue
        if "anchor" not in str(error):
            raise
        fixed_common_mask = struc.filter_intersection(fixed, mobile)
        fixed_coord = fixed.coord[fixed_common_mask]
        mobile_common_mask = struc.filter_intersection(mobile, fixed)
        mobile_coord = mobile.coord[mobile_common_mask]
        _, transformation = struc.superimpose(fixed_coord, mobile_coord)
        mobile_superimposed = transformation.apply(mobile)
        return (
            mobile_superimposed,
            transformation,
            np.where(fixed_common_mask)[0],
            np.where(mobile_common_mask)[0],
        )
