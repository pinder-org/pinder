from __future__ import annotations
from enum import Enum

import numpy as np
import torch

from numpy.typing import NDArray
from torch_geometric.data import HeteroData

from pinder.core.index.system import PinderSystem
from pinder.core.loader.structure import Structure
from pinder.core.utils import constants as pc
from pinder.core.utils.log import setup_logger

log = setup_logger(__name__)

try:
    from torch_cluster import knn_graph

    torch_cluster_installed = True
except ImportError as e:
    log.warning(
        "torch-cluster is not installed!"
        "Please install the appropriate library for your pytorch installation."
        "See https://github.com/rusty1s/pytorch_cluster/issues/185 for background."
    )
    torch_cluster_installed = False


def structure2tensor(
    atom_coordinates: NDArray[np.double] | None = None,
    atom_types: NDArray[np.str_] | None = None,
    residue_coordinates: NDArray[np.double] | None = None,
    residue_ids: NDArray[np.int_] | None = None,
    residue_types: NDArray[np.str_] | None = None,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    property_dict = {}
    if atom_types is not None:
        types_array_ele = np.zeros(
            (len(atom_types), len(set(list(pc.ELE2NUM.values()))))
        )
        for i, name in enumerate(atom_types):
            types_array_ele[i, pc.ELE2NUM.get(name, "C")] = 1.0

        property_dict["atom_types"] = torch.tensor(types_array_ele).type(dtype)

    if residue_types is not None:
        unknown_name_idx = max(pc.AA_TO_INDEX.values()) + 1
        types_array_res = np.zeros((len(residue_types), 1))
        for i, name in enumerate(residue_types):
            types_array_res[i] = pc.AA_TO_INDEX.get(name, unknown_name_idx)
        property_dict["residue_types"] = torch.tensor(types_array_res).type(dtype)

    if atom_coordinates is not None:
        property_dict["atom_coordinates"] = torch.tensor(atom_coordinates).type(dtype)
    if residue_coordinates is not None:
        property_dict["residue_coordinates"] = torch.tensor(residue_coordinates).type(
            dtype
        )
    if residue_ids is not None:
        property_dict["residue_ids"] = torch.tensor(residue_ids).type(dtype)
    return property_dict


class NodeRepresentation(Enum):
    Surface = "surface"
    Atom = "atom"
    Residue = "residue"


class PairedPDB(HeteroData):  # type: ignore
    @classmethod
    def from_pinder_system(
        cls,
        system: PinderSystem,
        node_types: set[NodeRepresentation],
        monomer1: str = "holo_receptor",
        monomer2: str = "holo_ligand",
        add_edges: bool = True,
        k: int = 10,
    ) -> PairedPDB:
        chain1_struc = getattr(system, monomer1)
        chain1_struc.filter("element", mask=["H"], negate=True, copy=False)
        chain2_struc = getattr(system, monomer2)
        chain2_struc.filter("element", mask=["H"], negate=True, copy=False)
        return cls.from_structure_pair(
            node_types=node_types,
            ligand=chain2_struc,
            receptor=chain1_struc,
            add_edges=add_edges,
            k=k,
        )

    @classmethod
    def from_structure_pair(
        cls,
        node_types: set[NodeRepresentation],
        ligand: Structure,
        receptor: Structure,
        receptor_chain_id: str = "R",
        ligand_chain_id: str = "L",
        add_edges: bool = True,
        k: int = 10,
    ) -> PairedPDB:
        graph = cls()
        rec_calpha = receptor.filter("atom_name", mask=["CA"])
        lig_calpha = ligand.filter("atom_name", mask=["CA"])
        rec_props = structure2tensor(
            atom_coordinates=receptor.coords,
            atom_types=receptor.atom_array.element,
            residue_coordinates=rec_calpha.coords,
            residue_types=rec_calpha.atom_array.res_name,
            residue_ids=rec_calpha.atom_array.res_id,
        )
        lig_props = structure2tensor(
            atom_coordinates=ligand.coords,
            atom_types=ligand.atom_array.element,
            residue_coordinates=lig_calpha.coords,
            residue_types=lig_calpha.atom_array.res_name,
            residue_ids=lig_calpha.atom_array.res_id,
        )

        if NodeRepresentation.Residue in node_types:
            graph["ligand_residue"].residueid = lig_props["residue_types"]
            graph["ligand_residue"].pos = lig_props["residue_coordinates"]
            graph["receptor_residue"].residueid = rec_props["residue_types"]
            graph["receptor_residue"].pos = rec_props["residue_coordinates"]
            if add_edges and torch_cluster_installed:
                graph["receptor_residue"].edge_index = knn_graph(
                    graph["receptor_residue"].pos, k=k
                )
                graph["ligand_residue"].edge_index = knn_graph(
                    graph["ligand_residue"].pos, k=k
                )

        if NodeRepresentation.Atom in node_types:
            graph["ligand_atom"].x = lig_props["atom_types"]
            graph["ligand_atom"].pos = lig_props["atom_coordinates"]
            graph["receptor_atom"].x = rec_props["atom_types"]
            graph["receptor_atom"].pos = rec_props["atom_coordinates"]
            if add_edges and torch_cluster_installed:
                graph["receptor_atom"].edge_index = knn_graph(
                    graph["receptor_atom"].pos, k=k
                )
                graph["ligand_atom"].edge_index = knn_graph(
                    graph["ligand_atom"].pos, k=k
                )

        ligand_chain_id = np.array([ligand_chain_id], dtype="U4").view(np.int32)[0]
        receptor_chain_id = np.array([receptor_chain_id], dtype="U4").view(np.int32)[0]

        graph["ligand_residue"].chain = torch.tensor([ligand_chain_id]).type(
            torch.int32
        )
        graph["receptor_residue"].chain = torch.tensor([receptor_chain_id]).type(
            torch.int32
        )

        return graph
