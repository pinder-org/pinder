import multiprocessing
from pathlib import Path

import networkx as nx
import pandas as pd
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile as pdbfile
from pinder.core.loader.utils import create_nx_radius_graph


def get_num_connected_components(args: tuple[Path, float]) -> tuple[int, int, str]:
    """Get number of connected components in CA-CA distance graph"""
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


def find_detached_structures(
    annotation_df: pd.DataFrame, radius: int = 15
) -> pd.DataFrame:
    """Find detached structures for each chain separately
    by detecting connected components in CA-CA distance graph with radius of 15 Angstroms (default)
    """
    annotation_df["pdb_id_slim"] = annotation_df["pdb_id"].apply(
        lambda x: x.split("_")[1].lstrip("0")
    )
    annotation_df["components"] = 0
    components = []
    inputs = []
    for i, row in tqdm(annotation_df.iterrows(), total=len(annotation_df)):
        inputs.append(
            (
                Path(row["path"])
                / f"{row['pdb_id_slim']}_{row['chain1']}_{row['pdb_id_slim']}_{row['chain2']}.pdb",
                radius,
            )
        )
    with multiprocessing.get_context("spawn").Pool() as p:
        components = list(
            tqdm(p.imap(get_num_connected_components, inputs), total=len(inputs))
        )
    component_dict = {tuple(x[1].split("_")): x[0] for x in components}
    annotation_df["components"] = [
        component_dict[(x["pdb_id_slim"], x["chain1"], x["pdb_id_slim"], x["chain2"])]
        if (x["pdb_id_slim"], x["chain1"], x["pdb_id_slim"], x["chain2"])
        in component_dict
        else -1
        for _, x in annotation_df.iterrows()
    ]
    return annotation_df
