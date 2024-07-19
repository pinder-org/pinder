from __future__ import annotations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from pinder.core.loader.structure import mask_common_uniprot, Structure
from pinder.core.utils import setup_logger
from pinder.core.utils.cloud import Gsutil
from pinder.core.utils.dataclass import stringify_dataclass
from pinder.core.index.utils import (
    IndexEntry,
    MetadataEntry,
    get_index,
    get_metadata,
    get_pinder_location,
    get_pinder_bucket_root,
)
from pinder.core.structure.models import MonomerName
from pinder.core.utils import unbound

log = setup_logger(__name__)
gs = Gsutil()
dataset_root = get_pinder_location()
log.debug(f"Dataset root: {dataset_root}")


class FolderNames:
    apo: str = "apo"
    holo: str = "holo"
    predicted: str = "predicted"
    alphafold: str = "predicted"
    af2: str = "predicted"


def _align_monomers_with_mask(
    monomer1: Structure,
    monomer2: Structure,
    remove_differing_atoms: bool = True,
    renumber_residues: bool = False,
    remove_differing_annotations: bool = False,
) -> tuple[Structure, Structure]:
    monomer2, monomer1 = monomer2.align_common_sequence(
        monomer1,
        remove_differing_atoms=remove_differing_atoms,
        renumber_residues=renumber_residues,
        remove_differing_annotations=remove_differing_annotations,
    )
    monomer2, _, _ = monomer2.superimpose(monomer1)
    return monomer1, monomer2


class PinderSystem:
    """Represents a system within the Pinder framework designed to handle and process
    structural data. It provides functionality to load, align, and analyze
    protein structures within the context of a Pinder index entry.

    Upon initialization, the system prepares directories for different structure types
    (holo, apo, predicted) and loads the respective protein structures.

    Methods include creating complexes, calculating RMSD, difficulty assessment,
    and updating substructure presence based on filtering criteria.

    Attributes:
        entry (IndexEntry): An index entry object containing primary metadata for the system.
        pdbs_path (Path): Path to the directory containing PDB files.
        mappings_path (Path): Path to the directory containing Parquet mapping files.
        native (Structure): The native (ground-truth) structure of the system.
        holo_dir (Path): Directory for the holo structures.
        apo_dir (Path): Directory for the apo structures.
        pred_dir (Path): Directory for the predicted structures.
        holo_receptor (Structure): The holo form of the receptor.
        holo_ligand (Structure): The holo form of the ligand.
        apo_receptor (Structure): The apo form of the receptor.
        apo_ligand (Structure): The apo form of the ligand.
        pred_receptor (Structure): The predicted structure of the receptor.
        pred_ligand (Structure): The predicted structure of the ligand.

    """

    def __init__(
        self,
        entry: str | IndexEntry,
        apo_receptor_pdb_code: str = "",
        apo_ligand_pdb_code: str = "",
        metadata: MetadataEntry | None = None,
        dataset_path: Path | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        """Initializes a PinderSystem instance.
        Loads the respective protein structures and prepares the environment
        based on the Pinder index entry provided.

        Parameters:
            entry (Union[str, IndexEntry]): The Pinder index entry ID or an IndexEntry object containing the protein structure metadata.
            apo_receptor_pdb_code (str, optional): PDB code for the apo form of the receptor if available.
            apo_ligand_pdb_code (str, optional): PDB code for the apo form of the ligand if available.
        """
        if isinstance(entry, str):
            pindex = get_index()
            row = pindex.query(f'id == "{entry}"').squeeze()
            entry = IndexEntry(**row.to_dict())
        self.entry = entry
        if dataset_path and isinstance(dataset_path, Path):
            self.pinder_root = dataset_path
        else:
            self.pinder_root = dataset_root
        self.pdbs_path = self.pinder_root / "pdbs"
        self.test_pdbs_path = self.pinder_root / "test_set_pdbs"
        self.mappings_path = self.pinder_root / "mappings"
        self.download_entry()

        self.native = self.load_structure(self.pdbs_path / self.entry.pinder_pdb)
        holo_pdb_path = (
            self.test_pdbs_path if self.entry.test_system else self.pdbs_path
        )
        self.holo_receptor = self.load_structure(holo_pdb_path / self.entry.holo_R_pdb)
        self.holo_ligand = self.load_structure(holo_pdb_path / self.entry.holo_L_pdb)
        # Can be multiple apo, we grab canonical unless pdb codes provided
        self.apo_receptor = self.load_structure(
            self.pdbs_path / self.entry.apo_R_pdb, chain_id="R"
        )
        self.apo_ligand = self.load_structure(
            self.pdbs_path / self.entry.apo_L_pdb, chain_id="L"
        )
        canon_L = self.entry.apo_L_pdb.split("__")[0]
        canon_R = self.entry.apo_R_pdb.split("__")[0]
        alt_r = self.load_alt_apo_structure(
            self.entry.apo_R_alt, apo_receptor_pdb_code, canon_R, chain_id="R"
        )
        self.apo_receptor = alt_r or self.apo_receptor
        alt_l = self.load_alt_apo_structure(
            self.entry.apo_L_alt, apo_ligand_pdb_code, canon_L, chain_id="L"
        )
        self.apo_ligand = alt_l or self.apo_ligand
        self.pred_receptor = self.load_structure(
            self.pdbs_path / self.entry.predicted_R_pdb, chain_id="R"
        )
        self.pred_ligand = self.load_structure(
            self.pdbs_path / self.entry.predicted_L_pdb, chain_id="L"
        )
        # Create native-aligned holo receptor and ligand prior to filtering on
        # common uniprot indices. To be used when creating unbound complexes.
        assert isinstance(self.native, Structure)
        self.native_R = self.native.filter("chain_id", ["R"])
        self.native_L = self.native.filter("chain_id", ["L"])
        holo_R = self.holo_receptor
        holo_L = self.holo_ligand
        assert isinstance(holo_R, Structure)
        assert isinstance(holo_L, Structure)
        self.aligned_holo_R, _, _ = holo_R.superimpose(self.native_R)
        self.aligned_holo_L, _, _ = holo_L.superimpose(self.native_L)
        # save metadata if provided
        if isinstance(metadata, MetadataEntry):
            self._metadata = metadata

        for k, v in kwargs.items():
            setattr(self, k, v)

    def filter_common_uniprot_res(self) -> None:
        """Filters the loaded protein structures for common UniProt residues, ensuring that
        comparisons between structures are made on a common set of residues.
        """
        structure_pairs = [
            ("holo_ligand", "apo_ligand"),
            ("holo_ligand", "pred_ligand"),
            ("holo_receptor", "apo_receptor"),
            ("holo_receptor", "pred_receptor"),
        ]
        for attr1, attr2 in structure_pairs:
            struc1 = getattr(self, attr1)
            struc2 = getattr(self, attr2)
            if struc1 and struc2:
                struc1, struc2 = mask_common_uniprot(struc1, struc2)
                setattr(self, attr1, struc1)
                setattr(self, attr2, struc2)

    def create_masked_bound_unbound_complexes(
        self,
        monomer_types: Sequence[str] = ["apo", "predicted"],
        remove_differing_atoms: bool = True,
        renumber_residues: bool = False,
        remove_differing_annotations: bool = False,
    ) -> tuple[Structure, Structure, Structure]:
        """Create dimer complexes for apo and predicted cropped to common holo substructures.

        The method applies a pairwise masking procedure which crops both unbound and bound structures
        such that they have equal numbers of residues and atom counts.

        Note: this method may result in very distorted holo (ground-truth) structures if
        the unbound monomer structures have little to no sequence and atoms in common.
        Unless you need all monomer types to be equal shapes, the `PinderSystem.create_complex` method
        or pure-superposition without masking (Structure.superimpose) is more appropriate.

        Parameters:
            monomer_types (Sequence[str]): The unbound monomer types to consider (apo, predicted, or both).
            remove_differing_atoms (bool):
                Whether to remove non-overlappings atoms that may still be present even after sequence-based alignment.
            renumber_residues (bool):
                Whether to renumber the residues in the receptor and ligand `Structure`'s to match numbering of the holo counterparts.
            remove_differing_annotations (bool):
                Whether to remove annotation categories (set to empty str or default value for the category type).
                This is useful if you need to perform biotite.structure.filter_intersection on the resulting structures.
                Note: this can have unintended side-effects like stripping the `element` attribute on structures.
                By default, the annotation categories are removed if they don't match in order to define the intersecting atom mask,
                after which the original structure annotations are preserved by applying the intersecting mask to the original AtomArray.
                Default is False.

        Returns:
            tuple[Structure, Structure, Structure]: A tuple of the cropped holo, apo, and predicted Structure objects, respectively.

        """
        holo_R = self.aligned_holo_R
        holo_L = self.aligned_holo_L
        apo_R = self.apo_receptor or self.aligned_holo_R
        apo_L = self.apo_ligand or self.aligned_holo_L
        pred_R = self.pred_receptor or self.aligned_holo_R
        pred_L = self.pred_ligand or self.aligned_holo_L
        if "apo" in monomer_types:
            # Ensure apo_R == holo_R and apo_L == holo_L
            holo_R, apo_R = _align_monomers_with_mask(
                holo_R,
                apo_R,
                remove_differing_atoms=remove_differing_atoms,
                renumber_residues=renumber_residues,
                remove_differing_annotations=remove_differing_annotations,
            )
            holo_L, apo_L = _align_monomers_with_mask(
                holo_L,
                apo_L,
                remove_differing_atoms=remove_differing_atoms,
                renumber_residues=renumber_residues,
                remove_differing_annotations=remove_differing_annotations,
            )
        if "predicted" in monomer_types:
            # Ensure pred_R == holo_R and pred_L == holo_L
            holo_R, pred_R = _align_monomers_with_mask(
                holo_R,
                pred_R,
                remove_differing_atoms=remove_differing_atoms,
                renumber_residues=renumber_residues,
                remove_differing_annotations=remove_differing_annotations,
            )
            holo_L, pred_L = _align_monomers_with_mask(
                holo_L,
                pred_L,
                remove_differing_atoms=remove_differing_atoms,
                renumber_residues=renumber_residues,
                remove_differing_annotations=remove_differing_annotations,
            )
        if set(monomer_types) == {"apo", "predicted"}:
            # Ensure apo_R == holo_R == pred_R
            apo_R, pred_R = _align_monomers_with_mask(
                apo_R,
                pred_R,
                remove_differing_atoms=remove_differing_atoms,
                renumber_residues=renumber_residues,
                remove_differing_annotations=remove_differing_annotations,
            )
            # Ensure apo_L == holo_L == pred_L
            apo_L, pred_L = _align_monomers_with_mask(
                apo_L,
                pred_L,
                remove_differing_atoms=remove_differing_atoms,
                renumber_residues=renumber_residues,
                remove_differing_annotations=remove_differing_annotations,
            )
        pred_complex = pred_R + pred_L
        apo_complex = apo_R + apo_L
        holo_complex = holo_R + holo_L
        return holo_complex, apo_complex, pred_complex

    def create_complex(
        self,
        receptor: Structure,
        ligand: Structure,
        remove_differing_atoms: bool = True,
        renumber_residues: bool = False,
        remove_differing_annotations: bool = False,
    ) -> Structure:
        """Creates a complex from the receptor and ligand structures.

        The complex is created by aligning the monomers to their
        respective holo forms and combining them into a single structure.

        Parameters:
            receptor (Structure): The receptor structure.
            ligand (Structure): The ligand structure.
            remove_differing_atoms (bool):
                Whether to remove non-overlappings atoms that may still be present even after sequence-based alignment.
            renumber_residues (bool):
                Whether to renumber the residues in the receptor and ligand `Structure`'s to match numbering of the holo counterparts.
            remove_differing_annotations (bool):
                Whether to remove annotation categories (set to empty str or default value for the category type).
                This is useful if you need to perform biotite.structure.filter_intersection on the resulting structures.
                Note: this can have unintended side-effects like stripping the `element` attribute on structures.
                By default, the annotation categories are removed if they don't match in order to define the intersecting atom mask,
                after which the original structure annotations are preserved by applying the intersecting mask to the original AtomArray.
                Default is False.

        Returns:
            Structure: A new Structure instance representing the complex.
        """
        receptor, holo_R = receptor.align_common_sequence(
            self.aligned_holo_R,
            remove_differing_atoms=remove_differing_atoms,
            renumber_residues=renumber_residues,
            remove_differing_annotations=remove_differing_annotations,
        )
        ligand, holo_L = ligand.align_common_sequence(
            self.aligned_holo_L,
            remove_differing_atoms=remove_differing_atoms,
            renumber_residues=renumber_residues,
            remove_differing_annotations=remove_differing_annotations,
        )
        R_super, _, _ = receptor.superimpose(holo_R)
        L_super, _, _ = ligand.superimpose(holo_L)
        binary = R_super + L_super
        return binary

    def create_apo_complex(
        self,
        remove_differing_atoms: bool = True,
        renumber_residues: bool = False,
        remove_differing_annotations: bool = False,
    ) -> Structure:
        """Creates an apo complex using the receptor and ligand structures.
        Falls back to the holo structures if apo structures are not available.

        Parameters:
            remove_differing_atoms (bool):
                Whether to remove non-overlappings atoms that may still be present even after sequence-based alignment.
            renumber_residues (bool):
                Whether to renumber the residues in the apo receptor and ligand `Structure`'s to match numbering of the holo counterparts.
            remove_differing_annotations (bool):
                Whether to remove annotation categories (set to empty str or default value for the category type).
                This is useful if you need to perform biotite.structure.filter_intersection on the resulting structures.
                Note: this can have unintended side-effects like stripping the `element` attribute on structures.
                By default, the annotation categories are removed if they don't match in order to define the intersecting atom mask,
                after which the original structure annotations are preserved by applying the intersecting mask to the original AtomArray.
                Default is False.

        Returns:
            Structure: A new Structure instance representing the apo complex.
        """
        return self.create_complex(
            self.apo_receptor or self.holo_receptor,
            self.apo_ligand or self.holo_ligand,
            remove_differing_atoms=remove_differing_atoms,
            renumber_residues=renumber_residues,
            remove_differing_annotations=remove_differing_annotations,
        )

    def create_pred_complex(
        self,
        remove_differing_atoms: bool = True,
        renumber_residues: bool = False,
        remove_differing_annotations: bool = False,
    ) -> Structure:
        """Creates a predicted complex using the receptor and ligand structures.
        Falls back to the holo structures if predicted structures are not available.

        Parameters:
            remove_differing_atoms (bool):
                Whether to remove non-overlappings atoms that may still be present even after sequence-based alignment.
            renumber_residues (bool):
                Whether to renumber the residues in the predicted receptor and ligand `Structure`'s to match numbering of the holo counterparts.
            remove_differing_annotations (bool):
                Whether to remove annotation categories (set to empty str or default value for the category type).
                This is useful if you need to perform biotite.structure.filter_intersection on the resulting structures.
                Note: this can have unintended side-effects like stripping the `element` attribute on structures.
                By default, the annotation categories are removed if they don't match in order to define the intersecting atom mask,
                after which the original structure annotations are preserved by applying the intersecting mask to the original AtomArray.
                Default is False.

        Returns:
            Structure: A new Structure instance representing the predicted complex.
        """
        return self.create_complex(
            self.pred_receptor or self.holo_receptor,
            self.pred_ligand or self.holo_ligand,
            remove_differing_atoms=remove_differing_atoms,
            renumber_residues=renumber_residues,
            remove_differing_annotations=remove_differing_annotations,
        )

    def unbound_rmsd(self, monomer_name: MonomerName) -> dict[str, float]:
        """Calculates the RMSD of the unbound receptor and ligand with respect to
        their holo forms for a given monomer state (apo or predicted).

        Parameters:
            monomer_name (MonomerName): Enum representing the monomer state.

        Returns:
            dict[str, float]: A dictionary with RMSD values for the receptor and ligand.
        """
        if not isinstance(monomer_name, MonomerName):
            monomer_name = MonomerName(monomer_name)

        if monomer_name.value == "apo":
            apo_R = self.apo_receptor or self.holo_receptor
            apo_L = self.apo_ligand or self.holo_ligand
        elif monomer_name.value == "predicted":
            apo_R = self.pred_receptor or self.holo_receptor
            apo_L = self.pred_ligand or self.holo_ligand

        holo_R = self.aligned_holo_R
        holo_L = self.aligned_holo_L

        # Even if atom counts are identical, annotation categories must be the same
        assert isinstance(apo_R, Structure)
        assert isinstance(apo_L, Structure)
        _, rms_R, _ = apo_R.superimpose(holo_R)
        _, rms_L, _ = apo_L.superimpose(holo_L)
        return {
            "monomer_name": monomer_name.value,
            "receptor_rmsd": rms_R,
            "ligand_rmsd": rms_L,
        }

    def unbound_difficulty(
        self, monomer_name: MonomerName, contact_rad: float = 5.0
    ) -> dict[str, float | str]:
        """Assesses the difficulty of docking the unbound structures based on the given monomer
        state (apo or predicted).

        Parameters:
            monomer_name (MonomerName): Enum representing the monomer state.

        Returns:
            dict[str, Union[float, str]]: A dictionary with difficulty assessment metrics.
        """
        if not isinstance(monomer_name, MonomerName):
            monomer_name = MonomerName(monomer_name)

        if monomer_name.value == "apo":
            apo_R = self.apo_receptor or self.holo_receptor
            apo_L = self.apo_ligand or self.holo_ligand
        elif monomer_name.value == "predicted":
            apo_R = self.pred_receptor or self.holo_receptor
            apo_L = self.pred_ligand or self.holo_ligand

        holo_R = self.aligned_holo_R
        holo_L = self.aligned_holo_L

        fnat_metrics: dict[str, float | str] = unbound.get_unbound_difficulty(
            holo_R,
            holo_L,
            apo_R,
            apo_L,
            contact_rad=contact_rad,
        )
        fnat_metrics["monomer_name"] = monomer_name.value
        return fnat_metrics

    def apo_monomer_difficulty(
        self,
        monomer_name: MonomerName,
        body: str,
        contact_rad: float = 5.0,
    ) -> dict[str, float | str]:
        """Evaluates the difficulty of docking for an individual apo monomer.

        Takes the specified body of the monomer (receptor or ligand) into account.

        Parameters:
            monomer_name (MonomerName): Enum representing the monomer state.
            body (str): String indicating which body ('receptor' or 'ligand') to use.

        Returns:
            dict[str, Union[float, str]]: A dictionary with docking difficulty metrics for the specified monomer body.
        """
        if not isinstance(monomer_name, MonomerName):
            monomer_name = MonomerName(monomer_name)

        if monomer_name.value == "apo":
            apo_R = self.apo_receptor or self.holo_receptor
            apo_L = self.apo_ligand or self.holo_ligand
        elif monomer_name.value == "predicted":
            apo_R = self.pred_receptor or self.holo_receptor
            apo_L = self.pred_ligand or self.holo_ligand

        apo_mono = apo_R if body == "receptor" else apo_L
        holo_L = self.aligned_holo_L
        holo_R = self.aligned_holo_R
        fnat_metrics: dict[str, float | str] = unbound.get_apo_monomer_difficulty(
            holo_R, holo_L, apo_mono, body, contact_rad=contact_rad
        )
        fnat_metrics["monomer_name"] = monomer_name.value
        return fnat_metrics

    def __repr__(self) -> str:
        """Returns a string representation of the PinderSystem instance.

        Returns:
            str: A string representation of the system.
        """
        ps_repr = "\n".join(
            [
                f"entry = {stringify_dataclass(self.entry)}",
                f"native={self.native}",
                f"holo_receptor={self.holo_receptor}",
                f"holo_ligand={self.holo_ligand}",
                f"apo_receptor={self.apo_receptor}",
                f"apo_ligand={self.apo_ligand}",
                f"pred_receptor={self.pred_receptor}",
                f"pred_ligand={self.pred_ligand}",
            ]
        )
        return "\n".join(["PinderSystem(", ps_repr, ")"])

    def download_entry(self) -> None:
        """Downloads data associated with an entry for the PinderSystem instance.

        It checks for the existence of PDB and Parquet files and downloads missing
        files from the Pinder bucket to the local dataset root directory.
        """
        sources = []
        for pdb_kind, pdb_paths in self.entry.pdb_paths.items():
            if not isinstance(pdb_paths, list):
                pdb_paths = [pdb_paths] if pdb_paths else []
            for p in pdb_paths:
                if len(p) == 0:
                    continue
                # Only download what doesn't exist on disk
                if pdb_kind in ["holo_R", "holo_L"] and self.entry.test_system:
                    dest = self.test_pdbs_path / Path(p).name
                else:
                    dest = self.pdbs_path / Path(p).name
                if not dest.is_file():
                    sources.append(f"{get_pinder_bucket_root()}/{p}")

        for map_kind, map_paths in self.entry.mapping_paths.items():
            if not isinstance(map_paths, list):
                map_paths = [map_paths] if map_paths else []
            for p in map_paths:
                # Only download what doesn't exist on disk
                if not (self.mappings_path / Path(p).name).is_file():
                    sources.append(f"{get_pinder_bucket_root()}/{p}")

        if sources:
            anchor = get_pinder_bucket_root()
            gs.cp_paths(sources, self.pinder_root, anchor)

    @property
    def filepaths(self) -> dict[str, str | None]:
        """Retrieves the file paths for the structural data associated with the system.

        Returns:
            dict[str, Optional[str]]:
                A dictionary with keys for each structure type
                (e.g., 'holo_receptor', 'holo_ligand') and values as the
                corresponding file paths or None if not available.
        """
        structure_monomers = [
            "holo_receptor",
            "holo_ligand",
            "apo_receptor",
            "apo_ligand",
            "pred_receptor",
            "pred_ligand",
        ]
        local_paths = {}
        for attr in structure_monomers:
            filepath = None
            structure = getattr(self, attr)
            if structure is not None:
                filepath = structure.filepath
            local_paths[attr] = filepath
        return local_paths

    @property
    def metadata(self) -> MetadataEntry | None:
        """Retrieves the additional metadata associated with an IndexEntry.

        Returns:
            MetadataEntry | None:
                A MetadataEntry object if the metadata exists, otherwise None.
        """
        if hasattr(self, "_metadata"):
            return self._metadata

        else:
            pinder_metadata = get_metadata()
            row = pinder_metadata.query(f'id == "{self.entry.id}"').squeeze()
            if row.shape[0]:
                meta = MetadataEntry(**row.to_dict())
            else:
                meta = None
            self._metadata = meta
        return self._metadata

    @property
    def pymol_script(self) -> str:
        """Constructs a PyMOL script to visualize the loaded structures in the PinderSystem.

        Returns:
            str: A string representing the PyMOL script commands for visualizing the structures.
        """
        native = self.pdbs_path / self.entry.pinder_pdb
        monomers = self.filepaths
        pdb_id = self.entry.pdb_id
        mono_colors = {
            "holo_receptor": "lanthanum",
            "holo_ligand": "samarium",
            "apo_receptor": "protactinium",
            "apo_ligand": "lutetium",
            "pred_receptor": "rhenium",
            "pred_ligand": "radium",
        }
        scene = [
            f"load {native}, {pdb_id}",
            f"color helium, {pdb_id} and chain R",
            f"color praseodymium, {pdb_id} and chain L",
            f"orient {pdb_id}",
            f"set grid_mode, 1",
            f"set grid_slot, -2, {pdb_id}",
        ]
        for k, v in monomers.items():
            if not v:
                continue
            mono_ch = k.split("_")[1][0].upper()
            scene.extend(
                [
                    f"load {v}, {pdb_id}_{k}",
                    f"alter {pdb_id}_{k}, chain='{mono_ch}'",
                    f"color {mono_colors[k]}, {pdb_id}_{k}",
                    f"align {pdb_id}_{k}, {pdb_id} and chain {mono_ch}",
                ]
            )
        return ";\n".join(scene)

    def load_alt_apo_structure(
        self,
        alt_pdbs: list[str],
        code: str,
        canon_code: str,
        chain_id: str | None = None,
    ) -> Structure | None:
        """Loads an alternate apo structure based on the provided PDB codes, if available.

        Parameters:
            alt_pdbs (List[str]): A list of alternative PDB file paths.
            code (str): The specific code to identify the alternate apo structure.
            canon_code (str): The canonical code for the apo structure.

        Returns:
            Optional[Structure]: The loaded Structure object if found, otherwise None.
        """
        if (code != "") & (code != canon_code):
            apo = [f for f in alt_pdbs if f.split("__")[0] == code]
            if apo:
                return self.load_structure(self.pdbs_path / apo[0], chain_id=chain_id)
            log.warning(f"Alternate apo PDB not found with code {code}")
        return None

    @staticmethod
    def load_structure(
        pdb_file: Path | None, chain_id: str | None = None
    ) -> Structure | None:
        """Loads a structure from a PDB file if it exists and is valid.

        Parameters:
            pdb_file (Optional[Path]): The file path to the PDB file.

        Returns:
            Optional[Structure]:
                The loaded Structure object if the file is valid, otherwise None.
        """
        if not pdb_file or Path(pdb_file).suffix != ".pdb":
            return None
        map_pqt = PinderSystem._get_mapping_pqt(pdb_file)
        loaded: Structure = Structure(pdb_file, map_pqt)
        if chain_id:
            chain_arr = np.array([chain_id] * loaded.atom_array.shape[0])
            loaded.atom_array.set_annotation("chain_id", chain_arr)
        return loaded

    @staticmethod
    def _safe_glob_item(folder: Path, expression: str) -> Path | None:
        matched = list(folder.glob(expression))
        if matched:
            return matched[0]
        return None

    @staticmethod
    def _get_mapping_pqt(pdb_file: Path | None) -> Path | None:
        if not pdb_file:
            return None

        map_pqt = pdb_file.parent.parent / f"mappings/{pdb_file.stem}.parquet"
        if map_pqt.is_file():
            return map_pqt
        return None
