from __future__ import annotations

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from numpy import nan
from typing import Union

from pinder.core.index.system import PinderSystem
from pinder.core.index.utils import IndexEntry, MetadataEntry
from pinder.core.loader.structure import mask_common_uniprot, Structure
from pinder.core.loader.utils import create_nx_radius_graph
from pinder.core.structure.atoms import filter_atoms, get_seq_identity
from pinder.core.structure.contacts import pairwise_contacts
from pinder.core.utils.log import setup_logger


log = setup_logger(__name__)

# Type hint for a simple metadata field
Field = Union[float, str, bool, int, None]

structure_monomers = [
    "holo_receptor",
    "holo_ligand",
    "apo_receptor",
    "apo_ligand",
    "pred_receptor",
    "pred_ligand",
]
unbound_monomers = [m for m in structure_monomers if "holo" not in m]


class PinderFilterBase:
    def __call__(self, ps: PinderSystem) -> bool:
        return self.filter(ps)

    def filter(self, ps: PinderSystem) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__


class FilterMetadataFields(PinderFilterBase):
    def __init__(
        self,
        **kwargs: tuple[str, Field],
    ) -> None:
        index_props = IndexEntry.model_json_schema()["properties"]
        metadata_props = MetadataEntry.model_json_schema()["properties"]
        valid_index_fields = {}
        valid_meta_fields = {}
        extra_meta_fields = {}

        for field, expr in kwargs.items():
            if not isinstance(expr, tuple) and len(expr) == 2:
                continue

            if field in index_props.keys():
                valid_index_fields[field] = expr
            elif field in metadata_props.keys():
                valid_meta_fields[field] = expr
            else:
                extra_meta_fields[field] = expr

        if not any((valid_index_fields, valid_meta_fields, extra_meta_fields)):
            log.warning(
                f"No valid index, metadata, or extra fields found in {kwargs.keys()}"
            )
        if any(extra_meta_fields):
            log.warning(f"Found extra fields in {kwargs.keys()}")

        self.valid_index_fields = valid_index_fields
        self.valid_meta_fields = valid_meta_fields
        self.extra_meta_fields = extra_meta_fields

    @staticmethod
    def _eval_expr(model_val: Field, query: tuple[str, Field]) -> bool:
        op, val = query
        # Handling None
        if model_val is None:
            model_val = nan
        if val is None:
            val = nan
        # Default operator
        if op == "":
            op = "=="
        # Handling strings
        if isinstance(model_val, str) and isinstance(val, str):
            expression = f"'{model_val}' {op} '{val}'"
        elif isinstance(model_val, str):
            expression = f"'{model_val}' {op} {val}"
        elif isinstance(val, str):
            expression = f"{model_val} {op} '{val}'"
        else:
            expression = f"{model_val} {op} {val}"

        code = compile(expression, "<string>", "eval")
        check: bool = eval(code)
        return check

    def filter(self, ps: PinderSystem) -> bool:
        meta = ps.metadata
        entry = ps.entry
        if isinstance(meta, MetadataEntry):
            for field, expr in {
                **self.valid_meta_fields,
                **self.extra_meta_fields,
            }.items():
                meta_val = getattr(meta, field)
                check = self._eval_expr(meta_val, expr)
                if not check:
                    return check
        for field, expr in self.valid_index_fields.items():
            index_val = getattr(entry, field)
            check = self._eval_expr(index_val, expr)
            if not check:
                return check
        return True


class ChainQuery:
    def __call__(self, chain: Structure) -> bool:
        return self.query(chain)

    def query(self, chain: Structure) -> bool:
        return True


class DualChainQuery:
    def __call__(self, chain1: Structure, chain2: Structure) -> bool:
        return self.query(chain1, chain2)

    def query(self, chain1: Structure, chain2: Structure) -> bool:
        return self.query(chain1, chain2)


class ResidueCount(ChainQuery):
    def __init__(
        self,
        min_residue_count: int | None = None,
        max_residue_count: int | None = None,
        count_hetero: bool = False,
    ) -> None:
        self.min_residue_count = min_residue_count
        self.max_residue_count = max_residue_count
        self.count_hetero = count_hetero

    def check_length(self, chain: Structure) -> int:
        if self.count_hetero:
            return len(chain.residues)
        else:
            non_het = chain.filter("hetero", [False], copy=True)
            return len(non_het.residues)

    def query(self, chain: Structure) -> bool:
        n_res = self.check_length(chain)
        if self.min_residue_count is not None:
            if n_res < self.min_residue_count:
                return False
        if self.max_residue_count is not None:
            if n_res > self.max_residue_count:
                return False
        return True


class AtomTypeCount(ChainQuery):
    def __init__(
        self,
        min_atom_type: int | None = None,
        max_atom_type: int | None = None,
        count_hetero: bool = False,
    ) -> None:
        self.min_atom_type = min_atom_type
        self.max_atom_type = max_atom_type
        self.count_hetero = count_hetero

    def check_length(self, chain: Structure) -> int:
        if self.count_hetero:
            return len(chain.atom_names)
        else:
            non_het = chain.filter("hetero", [False], copy=True)
            return len(non_het.atom_names)

    def query(self, chain: Structure) -> bool:
        n_at = self.check_length(chain)
        if self.min_atom_type is not None:
            if n_at < self.min_atom_type:
                return False
        if self.max_atom_type is not None:
            if n_at > self.max_atom_type:
                return False
        return True


class CompleteBackBone(ChainQuery):
    def __init__(self, fraction: float = 0.9) -> None:
        self.fraction = fraction

    def check_backbone(self, protein: Structure) -> bool:
        arr = protein.atom_array
        backbone = arr[protein.backbone_mask]
        calpha = arr[protein.calpha_mask]
        frac_complete = backbone.shape[0] / (calpha.shape[0] * 3)
        check: bool = frac_complete >= self.fraction
        return check

    def query(self, chain: Structure) -> bool:
        return self.check_backbone(chain)


class CheckChainElongation(ChainQuery):
    def __init__(self, max_var_contribution: float = 0.92) -> None:
        self.max_var = max_var_contribution

    def __call__(self, chain: Structure) -> bool:
        calpha_coords = chain.coords[chain.calpha_mask]
        return self.get_max_var(calpha_coords) <= self.max_var

    @staticmethod
    def get_max_var(coords: NDArray[np.double]) -> float:
        _, _, V = np.linalg.svd(coords - coords.mean(axis=0))
        projection = np.matmul(coords, V)
        variance = np.var(projection, axis=0)
        max_var: float = (variance / variance.sum()).max()
        return max_var


class DetachedChainQuery(ChainQuery):
    def __init__(self, radius: int = 12, max_components: int = 2) -> None:
        self.radius = radius
        self.max_components = max_components

    def query(self, chain: Structure) -> bool:
        calpha_coords = chain.coords[chain.calpha_mask]
        graph = create_nx_radius_graph(calpha_coords, radius=self.radius)
        check: bool = nx.number_connected_components(graph) <= self.max_components
        return check


class CheckContacts(DualChainQuery):
    def __init__(
        self,
        min_contacts: int = 5,
        radius: float = 10.0,
        calpha_only: bool = True,
        backbone_only: bool = True,
        heavy_only: bool = True,
    ) -> None:
        self.min_contacts = min_contacts
        self.radius = radius
        self.calpha_only = calpha_only
        self.backbone_only = backbone_only
        self.heavy_only = heavy_only

    def query(self, chain1: Structure, chain2: Structure) -> bool:
        R_chain = chain1.chains
        L_chain = chain2.chains
        binary = chain1 + chain2
        arr = binary.atom_array.copy()
        bound_contacts = pairwise_contacts(
            arr,
            R_chain,
            L_chain,
            radius=self.radius,
            heavy_only=self.heavy_only,
            backbone_only=self.backbone_only,
            calpha_only=self.calpha_only,
        )
        chain_map: dict[str, list[int]] = {ch: [] for ch in R_chain + L_chain}
        for cp in bound_contacts:
            c1, c2, r1, r2 = cp
            chain_map[c1].append(r1)
            chain_map[c2].append(r2)

        # Create set of residues in interface split into receptor and ligand
        rec_res = chain_map[R_chain[0]]
        lig_res = chain_map[L_chain[0]]
        arr = filter_atoms(arr, self.calpha_only, self.backbone_only, self.heavy_only)
        interface_mask = (
            np.isin(arr.chain_id, R_chain) & np.isin(arr.res_id, rec_res)
        ) | (np.isin(arr.chain_id, L_chain) & np.isin(arr.res_id, lig_res))
        check: bool = interface_mask.sum() >= self.min_contacts
        return check


class FilterByResidueCount(PinderFilterBase):
    def __init__(self, **kwargs: int | None | bool) -> None:
        self.query = ResidueCount(**kwargs)  # type: ignore

    def filter(self, ps: PinderSystem) -> bool:
        """Filter by residue count in holo monomers.

        Examples
        --------
        >>> from pinder.core import PinderSystem
        >>> pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
        >>> ps = PinderSystem(pinder_id)
        >>> res_filter = FilterByResidueCount(min_residue_count=10, max_residue_count=500)
        >>> res_filter(ps)
        False

        """
        holo_structs = ["holo_receptor", "holo_ligand"]
        return all(self.query(getattr(ps, structure)) for structure in holo_structs)


class FilterByMissingHolo(PinderFilterBase):
    def filter(self, ps: PinderSystem) -> bool:
        holo_structs = ["holo_receptor", "holo_ligand"]
        return all(getattr(ps, structure) for structure in holo_structs)


class FilterSubByContacts(PinderFilterBase):
    def __init__(
        self,
        min_contacts: int = 5,
        radius: float = 10.0,
        calpha_only: bool = True,
        backbone_only: bool = True,
        heavy_only: bool = True,
    ) -> None:
        self.min_contacts = min_contacts
        self.radius = radius
        self.calpha_only = calpha_only
        self.backbone_only = backbone_only
        self.heavy_only = heavy_only
        self.check_contacts = CheckContacts(
            min_contacts, radius, calpha_only, backbone_only, heavy_only
        )

    def filter(self, ps: PinderSystem) -> bool:
        return self.check_contacts(ps.aligned_holo_R, ps.aligned_holo_L)


class FilterByHoloElongation(PinderFilterBase):
    def __init__(self, max_var_contribution: float = 0.92) -> None:
        self.max_var_contribution = max_var_contribution
        self.check_elongation = CheckChainElongation(max_var_contribution)

    def filter(self, ps: PinderSystem) -> bool:
        check_R = self.check_elongation(ps.holo_receptor)
        check_L = self.check_elongation(ps.holo_ligand)
        return check_R and check_L


class FilterDetachedHolo(PinderFilterBase):
    def __init__(self, radius: int = 12, max_components: int = 2) -> None:
        self.radius = radius
        self.max_components = max_components
        self.detached_chain_query = DetachedChainQuery(radius, max_components)

    def filter(self, ps: PinderSystem) -> bool:
        detached_R = self.detached_chain_query(ps.holo_receptor)
        detached_L = self.detached_chain_query(ps.holo_ligand)
        return detached_R and detached_L


class PinderFilterSubBase:
    def __call__(self, ps: PinderSystem) -> PinderSystem:
        return self.filter(ps)

    def filter(self, ps: PinderSystem) -> PinderSystem:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def filter_by_chain_query(
        ps: PinderSystem,
        chain_query: ChainQuery,
        update_monomers: bool = True,
    ) -> PinderSystem:
        for attr in structure_monomers:
            structure = getattr(ps, attr)
            if structure is not None:
                keep = chain_query(structure)
                if update_monomers and not keep:
                    setattr(ps, attr, None)
        return ps


class FilterSubLengths(PinderFilterSubBase):
    def __init__(self, min_length: int = 0, max_length: int = 1000) -> None:
        self.min_length = min_length
        self.max_length = max_length

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        for attr in structure_monomers:
            structure = getattr(ps, attr)
            if structure is not None:
                calpha = structure.atom_array[structure.calpha_mask]
                keep = self.min_length <= calpha.shape[0] <= self.max_length
                if update_monomers and not keep:
                    setattr(ps, attr, None)
        return ps


class FilterSubRmsds(PinderFilterSubBase):
    def __init__(self, rmsd_cutoff: float = 7.5) -> None:
        self.rmsd_cutoff = rmsd_cutoff

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        if ps.holo_receptor is None or ps.holo_ligand is None:
            log.warning(
                f"Skipping FilterSubRmsds on {ps.entry.id} as holo monomers have been filtered out"
            )
            return ps
        apo_rmsd = ps.unbound_rmsd("apo")
        pred_rmsd = ps.unbound_rmsd("predicted")
        pairwise_rmsd = {
            "apo_receptor": apo_rmsd["receptor_rmsd"],
            "apo_ligand": apo_rmsd["ligand_rmsd"],
            "pred_receptor": pred_rmsd["receptor_rmsd"],
            "pred_ligand": pred_rmsd["ligand_rmsd"],
        }
        for attr, rms in pairwise_rmsd.items():
            structure = getattr(ps, attr)
            if structure is not None:
                keep = rms <= self.rmsd_cutoff
                if update_monomers and not keep:
                    setattr(ps, attr, None)
        return ps


class FilterByHoloOverlap(PinderFilterSubBase):
    def __init__(self, min_overlap: int = 5) -> None:
        self.min_overlap = min_overlap

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        for attr in unbound_monomers:
            structure = getattr(ps, attr)
            if structure is not None:
                if attr.endswith("_receptor"):
                    holo = ps.holo_receptor
                else:
                    holo = ps.holo_ligand
                structure_common, holo_common = mask_common_uniprot(structure, holo)
                map_overlap = len(structure_common.residues)
                keep = map_overlap >= self.min_overlap
                if update_monomers and not keep:
                    setattr(ps, attr, None)
        return ps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_overlap={self.min_overlap})"


class FilterByHoloSeqIdentity(PinderFilterSubBase):
    def __init__(self, min_sequence_identity: float = 0.8) -> None:
        self.min_sequence_identity = min_sequence_identity

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        for attr in unbound_monomers:
            structure = getattr(ps, attr)
            if structure is not None:
                if attr.endswith("_receptor"):
                    holo = ps.holo_receptor
                else:
                    holo = ps.holo_ligand
                seq_ident = get_seq_identity(holo.sequence, structure.sequence)
                keep = seq_ident >= self.min_sequence_identity
                if update_monomers and not keep:
                    setattr(ps, attr, None)
        return ps

    def __repr__(self) -> str:
        seq_ident = f"min_sequence_identity={self.min_sequence_identity}"
        return f"{self.__class__.__name__}({seq_ident})"


class FilterSubByAtomTypes(PinderFilterSubBase):
    def __init__(self, min_atom_types: int = 4) -> None:
        self.min_atom_types = min_atom_types
        self.atom_type_count = AtomTypeCount(min_atom_type=min_atom_types)

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        return self.filter_by_chain_query(
            ps, self.atom_type_count, update_monomers=update_monomers
        )


class FilterSubByChainQuery(PinderFilterSubBase):
    def __init__(self, chain_query: ChainQuery) -> None:
        self.chain_query = chain_query

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        return self.filter_by_chain_query(
            ps, self.chain_query, update_monomers=update_monomers
        )


class FilterByElongation(PinderFilterSubBase):
    def __init__(self, max_var_contribution: float = 0.92) -> None:
        self.max_var_contribution = max_var_contribution
        self.check_elongation = CheckChainElongation(max_var_contribution)

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        return self.filter_by_chain_query(
            ps, self.check_elongation, update_monomers=update_monomers
        )


class FilterDetachedSub(PinderFilterSubBase):
    def __init__(self, radius: int = 12, max_components: int = 2) -> None:
        self.radius = radius
        self.max_components = max_components
        self.detached_chain_query = DetachedChainQuery(radius, max_components)

    def filter(self, ps: PinderSystem, update_monomers: bool = True) -> PinderSystem:
        return self.filter_by_chain_query(
            ps, self.detached_chain_query, update_monomers=update_monomers
        )
