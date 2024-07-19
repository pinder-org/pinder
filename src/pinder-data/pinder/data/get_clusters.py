from __future__ import annotations
import pickle as pkl
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

from pinder.core.index.utils import setup_logger
from pinder.data import alignment_utils, graph_utils
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.config import ClusterConfig
from pinder.core.utils.timer import timeit


log = setup_logger(__name__)


def get_dimer_to_cluster_pair(
    interfaces: dict[tuple[str, str], alignment_utils.Interface],
    node_to_cluster: dict[str, int],
) -> dict[tuple[str, str], tuple[int, int]]:
    """
    Get a mapping from interface dimers to cluster pairs.

    The value (-1, -1) corresponds to the null cluster.

    Parameters
    ----------
    interfaces: Dict[Tuple[str,str], Interface]
        Mapping (dict) from dimers, represented as a tuple of monomer PINDER IDs
        to Interface objects representing the dimer interface.
    node_to_cluster: Dict[str, int]
        Mapping (dict) from monomers, represented as a single foldseek monomer ID
        to monomer cluster IDs.

    Returns
    -------
    dict[tuple[str, str], tuple[int, int]]:
        Mapping (dict) from dimers, represented as a tuple of monomer PINDER IDs
        to monomer cluster ID pairs.

    """
    # Translate interface keys to PINDER keys
    pinder_keys = [
        (
            graph_utils.interface_monomer_to_system_monomer(tup[0]),
            graph_utils.interface_monomer_to_system_monomer(tup[1]),
        )
        for tup in interfaces.keys()
    ]

    # interfaces (pinder monomers) to pairs of foldseek monomers
    return dict(
        zip(
            pinder_keys,
            [
                (
                    node_to_cluster.get(graph_utils.system_monomer_to_fsid(tup[0]), -1),
                    node_to_cluster.get(graph_utils.system_monomer_to_fsid(tup[1]), -1),
                )
                for tup in pinder_keys
            ],
        )
    )


@timeit
def add_clusters_to_index(
    index: pd.DataFrame,
    dimer_to_cluster_pair: dict[tuple[str, str], tuple[int, int]],
    name: str,
) -> None:
    """Add cluster pairs created from graph clustering to an existing index

    IMPORTANT: We define a cluster pair as a tuple of integers
        (x, y) | x <= y

    In order to preserve the ability to back out which monomer goes with
    which cluster, we also store R- and L-specific cluster IDs

    Parameters
    ----------
    index: pd.DataFrame
        The existing index, containing the "id" field, at least

    dimer_to_cluster_pair: dict[tuple[str, str], tuple[int, int]]
        A mapping from dimers: pair of monomers in pinder naming format
        to cluster id pairs.

    Returns
    -------
    None. Mutates index
    """

    # Make the nested map more readable
    def helper(system_id: str) -> tuple[int, int]:
        monomer_list = system_id.split("--")
        dimer = (monomer_list[0], monomer_list[1])
        return dimer_to_cluster_pair.get(dimer, (-1, -1))

    ## Note: because of the sorting, better to precompute and reuse,
    ## Despite adding a bit of overhead
    # We need to see all of these anyway, might as well precompute
    sysid_to_cluster_pair = dict(
        zip(index["id"].values, list(map(helper, index["id"].values)))
    )

    def sysid_to_cluster_string(tup: tuple[int, int]) -> str:
        sort_tup = sorted(tup)
        return f"cluster_{sort_tup[0]}_{sort_tup[1]}"

    # Write to index
    index[f"{name}_id_R"] = index["id"].map(
        lambda x: sysid_to_cluster_pair.get(x, (-1, -1))[0]
    )
    index[f"{name}_id_L"] = index["id"].map(
        lambda x: sysid_to_cluster_pair.get(x, (-1, -1))[1]
    )
    index[f"{name}_id"] = index["id"].map(
        lambda x: sysid_to_cluster_string(sysid_to_cluster_pair.get(x, (-1, -1)))
    )


def choose_final_clusters(index: pd.DataFrame, cluster_prefix: str) -> None:
    """Choose cluster assignments to use as the "final" cluster assignments

    Copies one set of cluster ID columns with the prefix: <cluster_prefix>
    to a new set of columns with the prefix: "cluster".

    E.g., if <cluster_prefix> is "foldseek_community", copies the columns
        ["foldseek_community_id_R",
         "foldseek_community_id_L",
         "foldseek_community_id"]
    to the columns
        ["cluster_id_R",
         "cluster_id_L",
         "cluster_id"]
    """
    index[["cluster_id_R", "cluster_id_L", "cluster_id"]] = index[
        [f"{cluster_prefix}_id_R", f"{cluster_prefix}_id_L", f"{cluster_prefix}_id"]
    ]


def load_cluster_cache(cache_pkl: Path) -> list[set[str]]:
    with open(cache_pkl, "rb") as f:
        cluster_data: list[set[str]] = pkl.load(f)
        return cluster_data


def save_cluster_cache(cluster_data: list[set[str]], cache_pkl: Path) -> None:
    with open(cache_pkl, "wb") as f:
        pkl.dump(cluster_data, f)
    log.info(f"Wrote cluster data to {cache_pkl}...")


def cluster(
    index: pd.DataFrame,
    foldseek_graph: nx.Graph,
    mmseqs_graph: nx.Graph | None,
    interfaces_clean: dict[tuple[str, str], alignment_utils.Interface],
    output_index_filename: str = "index.2.csv.gz",
    checkpoint_dir: Path = Path("/tmp/clust_chkpt"),
    config: ClusterConfig = ClusterConfig(),
    foldseek_components: list[set[str]] | None = None,
    foldseek_communities: list[set[str]] | None = None,
    mmseqs_components: list[set[str]] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Cluster interfaces based on FOLDSEEK or MMSEQS alignments-based graph

    Parameters
    ----------
    index: pd.DataFrame
        The input PINDER index file
    foldseek_graph: nx.Graph
        The foldseek-similarity-based interface graph. This graph should:
            A) contain monomer nodes
            B) contain weighted edges that indicate interface similarity
            C) have been pre-filtered or "cleaned" as desired.
    mmseqs_graph: nx.Graph | None
        Optional: The mmseqs-similarity-based interface graph. This graph should:
            A) contain monomer nodes
            B) contain weighted edges that indicate interface similarity
            C) have been pre-filtered or "cleaned" as desired.
    interfaces_clean: dict[tuple[str, str], Interface]
        Dictionary mapping dimers to min. length-filtered interfaces.
    output_index_file: str
        Name of the updated index file. For example: "index.2.csv.gz"
    checkpoint_dir: Path
        Directory in which to save checkpoints.
    seed: int | np.random.RandomState
        The random seed to use for AsynLPA clustering.
    edge_weight: str | None
        The edge attribute for nx.Graph inputs representing the weight of an edge.
        If None, uses 1 for all weights. Used for AsynLPA clustering.
        Defaults to "weights".
    foldseek_components: dict[tuple[int, int], set[tuple[tuple[str, str], bool]]]
        Mapping from component id to dimer index + flag indicating whether it
        has been sorted
    foldseek_communities: dict[tuple[int, int], set[tuple[tuple[str, str], bool]]]
        Mapping from community id to dimer index + flag indicating whether it
        has been sorted
    mmseqs_components: dict[tuple[int, int], set[tuple[tuple[str, str], bool]]]
        Mapping from component id to dimer index + flag indicating whether it
        has been sorted
    canonical_method: str
        name of the "primary" clustering method

    Returns
    -------
    pd.DataFrame
        The input index, with additional fields indicating component
        and community IDs

    Also writes this DataFrame to file
    """
    ## Housekeeping
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    foldseek_component_pkl = checkpoint_dir / "foldseek_components.pkl"
    foldseek_community_pkl = checkpoint_dir / "foldseek_communities.pkl"
    mmseqs_component_pkl = checkpoint_dir / "mmseqs_components.pkl"
    index_checkpoint = checkpoint_dir / output_index_filename
    if index_checkpoint.is_file() and use_cache:
        log.info(f"{index_checkpoint} exists, using cache...")
        index = read_csv_non_default_na(
            index_checkpoint, dtype={"pdb_id": "str", "entry_id": "str"}
        )
        return index

    # Remove graph edges with weight below config.foldseek_cluster_edge_threshold
    edges_to_remove = []
    for n1, n2, data in tqdm(
        foldseek_graph.edges(data=True), total=foldseek_graph.number_of_edges()
    ):
        if data["weight"] < config.foldseek_cluster_edge_threshold:
            edges_to_remove.append((n1, n2))
    foldseek_graph.remove_edges_from(edges_to_remove)

    ### Do foldseek clustering
    ## Compute foldseek connected components if not cached
    if foldseek_components:
        log.info(
            f"Skipping foldseek component clustering, foldseek_components provided..."
        )
        if not foldseek_component_pkl.is_file():
            save_cluster_cache(foldseek_components, foldseek_component_pkl)
    elif foldseek_component_pkl.is_file() and use_cache:
        log.info(
            f"Skipping foldseek component clustering, {foldseek_component_pkl} checkpoint exists..."
        )
        foldseek_components = load_cluster_cache(foldseek_component_pkl)
    else:
        log.info(
            (
                f"Computing components of foldseek interface graph G_fs = (V, E): "
                f"|V|={foldseek_graph.number_of_nodes()},"
                f"|E|={foldseek_graph.number_of_edges()}..."
            )
        )

        # Cluster using connected components
        foldseek_components = graph_utils.cluster_from_graph(
            foldseek_graph, community=False
        )
        # Save checkpoint
        assert foldseek_components is not None
        save_cluster_cache(foldseek_components, foldseek_component_pkl)
        log.info("Finished computing foldseek components.")

    ## Compute foldseek communities if not cached
    if foldseek_communities:
        log.info(
            f"Skipping foldseek community clustering, foldseek_communities provided..."
        )
        if not foldseek_community_pkl.is_file():
            save_cluster_cache(foldseek_communities, foldseek_community_pkl)
    elif foldseek_community_pkl.is_file() and use_cache:
        log.info(
            f"Skipping foldseek community clustering, {foldseek_community_pkl} checkpoint exists..."
        )
        foldseek_communities = load_cluster_cache(foldseek_community_pkl)
    else:
        log.info(
            (
                f"Computing communities of foldseek interface graph G_fs = (V, E): "
                f"|V|={foldseek_graph.number_of_nodes()},"
                f"|E|={foldseek_graph.number_of_edges()}..."
            )
        )

        # Cluster using community detection (LPA)
        foldseek_communities = graph_utils.cluster_from_graph(
            foldseek_graph, community=True, weight=config.edge_weight, seed=config.seed
        )
        # Save checkpoint
        assert foldseek_communities is not None
        save_cluster_cache(foldseek_communities, foldseek_community_pkl)
        log.info("Finished computing communities.")

    ### Get dimer clusters, write to index
    ## Map interfaces to pairs of components
    log.info("Mapping interfaces to component pairs...")
    fs_node_2_comp = graph_utils.get_node_to_cluster_mapping(foldseek_components)
    dimer_2_comp_pair = get_dimer_to_cluster_pair(interfaces_clean, fs_node_2_comp)
    ## Add cluster pairs to index
    add_clusters_to_index(index, dimer_2_comp_pair, name="foldseek_component")

    ## Map interfaces to pairs of communities
    log.info("Mapping interfaces to community pairs...")
    fs_node_2_comm = graph_utils.get_node_to_cluster_mapping(foldseek_communities)
    dimer_2_comm_pair = get_dimer_to_cluster_pair(interfaces_clean, fs_node_2_comm)
    ## Add cluster pairs to index
    add_clusters_to_index(index, dimer_2_comm_pair, name="foldseek_community")

    ### Do mmseqs clustering
    if mmseqs_graph is None:
        log.info("Skipping mmseqs-related filtering")
    elif mmseqs_components:
        log.info(f"Skipping mmseqs component clustering, mmseqs_components provided...")
        if not mmseqs_component_pkl.is_file():
            save_cluster_cache(mmseqs_components, mmseqs_component_pkl)
    elif mmseqs_component_pkl.is_file() and use_cache:
        log.info(
            f"Skipping mmseqs component clustering, {mmseqs_component_pkl} checkpoint exists..."
        )
        mmseqs_components = load_cluster_cache(mmseqs_component_pkl)
    else:
        log.info(
            (
                f"Computing components of mmseqs interface graph G_fs = (V, E): "
                f"|V|={mmseqs_graph.number_of_nodes()},"
                f"|E|={mmseqs_graph.number_of_edges()}..."
            )
        )
        # Cluster using connected components
        mmseqs_components = graph_utils.cluster_from_graph(
            mmseqs_graph,
            community=False,
        )
        # Save checkpoint
        assert mmseqs_components is not None
        save_cluster_cache(mmseqs_components, mmseqs_component_pkl)
        log.info("Finished computing mmseqs components.")
    if mmseqs_components is not None:
        # Here we can assume we have the components
        ### Get dimer clusters, write to index
        ## Map interfaces to pairs of components
        log.info("Mapping interfaces to mmseqs component pairs...")
        ms_node_2_comp = graph_utils.get_node_to_cluster_mapping(mmseqs_components)
        ms_dimer_2_comp_pair = get_dimer_to_cluster_pair(
            interfaces_clean, ms_node_2_comp
        )
        ## Add cluster pairs to index
        add_clusters_to_index(index, ms_dimer_2_comp_pair, name="mmseqs_component")
        log.info("Added mmseqs_component clusters to index...")

    ## Select "canonical" clusters
    log.info(f"Choosing {config.canonical_method} clusters as primary")
    choose_final_clusters(index, config.canonical_method)

    ## Write index, return
    log.info("Writing index")
    index.to_csv(index_checkpoint, index=False)
    log.info(f"Wrote new index with clusters to {index_checkpoint}")
    return index
