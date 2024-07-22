import multiprocessing
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile as pdbfile
from pinder.core.loader.utils import create_nx_radius_graph


def get_num_connected_components(args: tuple[Path, float]) -> tuple[int, int, str]:
    """Get number of connected components in CA-CA distance graph.

    Find detached structures for each chain separately by detecting connected components in CA-CA distance
    graph with radius of 15 Angstroms (default)
    """
    pdb_file, radius = args
    try:
        model = pdbfile.read(pdb_file.as_posix())
        structure = model.get_structure(model=1, extra_fields=["atom_id", "b_factor"])
        structure = structure[structure.atom_name == "CA"]
        chains = list(sorted(set(structure.chain_id)))

        if len(chains) != 2:
            return -1, -1, pdb_file.stem

        # Dimers are supposed to be chain R = chain 1, chain L = chain 2.
        # Sorted will return L, R
        chain1 = structure[structure.chain_id == chains[1]]
        chain2 = structure[structure.chain_id == chains[0]]

        coords1 = chain1.coord
        graph1 = create_nx_radius_graph(coords1, radius=radius)
        number_of_components_chain1 = nx.number_connected_components(graph1)

        coords2 = chain2.coord
        graph2 = create_nx_radius_graph(coords2, radius=radius)
        number_of_components_chain2 = nx.number_connected_components(graph2)

        return (
            number_of_components_chain1,
            number_of_components_chain2,
            Path(pdb_file).name,
        )
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise e
        return -1, -1, pdb_file.stem
