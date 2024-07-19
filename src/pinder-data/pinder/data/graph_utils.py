from __future__ import annotations

from copy import deepcopy
import pickle as pkl
from pathlib import Path
from string import digits
from operator import gt
from functools import reduce

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any, Callable

from pinder.core.index.utils import setup_logger
from pinder.core.utils.timer import timeit
from pinder.data.alignment_utils import Interface, load_interface_pkl
from pinder.data.config import GraphConfig, get_config_hash
from pinder.data.foldseek_utils import alignment_to_parquet

log = setup_logger(__name__)


def get_alignment_graph_with_indices(alignment_pqt: Path) -> nx.DiGraph:
    """
    Convert a foldseek alignment parquet file to a float-weighted monomer similarity graph.

    This graph contains an edge between two given monomers (X, Y) if all of the
    following conditions are met:
        0. X != Y (string equality on "{pdb_id}_{chain}")
        1. Similarity score is greater than <score_thr>
        2. Similarity score is less than <upper_threshold>
        3. Alignment length is greater than <min_length
    I will refer to this as a "valid" alignment.

    This graph contains exactly one node for a monomer in <alignment_file>
    IFF the monomer has at least one valid alignment. This means that the output
    graph does not necessarily contain all monomers.

    The edge score between X and Y is defined as the the score of the one valid
    alignment between X and Y in the alignment file, but which one?

    Parameters
    ----------
    alignment_file: Path
        Path to a pre-filtered alignment parquet file converted from original alignment format
        to parquet via foldseek_utils.alignment_to_parquet. Expects specific formatting.

    Returns
    -------
    nx.DiGraph:
        Graph containing nodes (monomers) and integer-weighted edges.


    ####################
    # FLAG
    ####################
    Possible issue: scores are not symmetric, but only one score is kept. Which
    score this is depends on alignment file order. This behavior seems to be
    inconsistent in tests...

    Possible issue: upper_threshold has potential to silently and completely
    remove monomers from graph. Ensure that this is desirable.
    """
    aln_df = pd.read_parquet(alignment_pqt)
    score_col = "pident" if "pident" in set(aln_df.columns) else "lddt"
    aln_df.rename(
        {score_col: "score", "query": "pdbid1", "target": "pdbid2"},
        axis=1,
        inplace=True,
    )
    aln_df["pdbid1"] = aln_df["pdbid1"].str.replace(".pdb", "")
    aln_df["pdbid2"] = aln_df["pdbid2"].str.replace(".pdb", "")
    query_target = aln_df[["pdbid1", "pdbid2", "score", "qstart", "qend"]].copy()
    target_query = aln_df[["pdbid2", "pdbid1", "score", "tstart", "tend"]].copy()
    query_target.loc[:, "alignment_idx"] = [
        (qstart, qend)
        for qstart, qend in zip(query_target["qstart"], query_target["qend"])
    ]
    target_query.loc[:, "alignment_idx"] = [
        (tstart, tend)
        for tstart, tend in zip(target_query["tstart"], target_query["tend"])
    ]
    query_target.drop(["qstart", "qend"], axis=1, inplace=True)
    target_query = target_query.drop(["tstart", "tend"], axis=1).rename(
        {"pdbid2": "pdbid1", "pdbid1": "pdbid2"}, axis=1
    )
    aln_df = pd.concat([query_target, target_query], ignore_index=True)
    del query_target
    del target_query
    alignment_graph: nx.DiGraph = nx.from_pandas_edgelist(
        aln_df,
        source="pdbid1",
        target="pdbid2",
        edge_attr=["alignment_idx", "score"],
        create_using=nx.DiGraph,
    )
    return alignment_graph


def sample_pairs_from_clusters(
    clusters: dict[str, list[str]], length_file: str | None = None
) -> tuple[set[str], set[str]]:
    """Sample pairs from clusters"""
    log.info(f"Sampling from {len(clusters)} clusters")

    sampled, rest = set(), set()
    if length_file is not None:
        with open(length_file, "rb") as f:
            lengths = pkl.load(f)
    for _, members in clusters.items():
        members = list(members)
        if length_file is not None:
            log.info(f"Length file {length_file}")
            try:
                sorted_member_lengths = np.argsort(
                    [lengths[(x[0], x[2])] + lengths[(x[1], x[3])] for x in members]
                )[::-1]
                sampled.add(members[sorted_member_lengths[0]])
                if len(members) > 1:
                    rest |= set([members[x] for x in sorted_member_lengths[1:]])
            except KeyError:
                log.error(f"Key error {members[0]}")
                sampled.add(members[0])
                if len(members) > 1:
                    rest |= set(members[1:])
        else:
            sampled.add(members[0])
            if len(members) > 1:
                rest |= set(members[1:])
    return sampled, rest


def system_to_pdb_chain_id(system_id: str) -> str:
    """Removes the uniprot component of a pinder monomer id

    Parameters
    ----------
    system_id: str
        The PINDER System monomer ID string

    Returns
    -------
    str
        The pdb_chain monomer ID, includes trailing chain digits
    """
    if "__" in system_id:
        pdb_id, rest = system_id.split("__")
        chain_id = rest.split("_")[0]
    else:
        pdb_id, chain_id = system_id.split("_")
    return f"{pdb_id}_{chain_id}"


def system_id_to_fsid_pair(system_id: str) -> tuple[str, str]:
    """Transform a PINDER System ID to a pair of Foldseek chain IDs

    Parameters
    ----------
    system_id: str
        The PINDER System ID string

    Returns
    -------
    tuple[str, str]
        The pair of Foldseek chain IDs
    """
    # mypy pls
    fsid_list = [system_monomer_to_fsid(e) for e in system_id.split("--")]
    return (fsid_list[0], fsid_list[1])


def system_monomer_to_fsid(system_monomer: str) -> str:
    """Transform a PINDER System monomer ID to a Foldseek chain

    Parameters
    ----------
    system_id: str
        The PINDER System monomer ID string

    Returns
    -------
    str
        The Foldseek chain ID
    """
    pdb_plus_chain = system_to_pdb_chain_id(system_monomer)
    return pdb_plus_chain.rstrip(digits)


def interface_monomer_to_system_monomer(interface_monomer: str) -> str:
    """Transform an Interface monomer ID to a PINDER System monomer ID

    This essentially ammounts to removing "-R" or "-L" from the end

    Parameters
    ----------
    system_id: str
        The Interface monomer ID string

    Returns
    -------
    str
        The PINDER System monomer ID string
    """
    if interface_monomer.endswith("-L"):
        return interface_monomer.removesuffix("-L")
    else:
        return interface_monomer.removesuffix("-R")


@timeit
def get_interface_graph(
    alignment_graph: nx.DiGraph,
    interfaces: dict[tuple[str, str], list[Interface]],
    coverage: float = 0.75,
) -> nx.Graph:
    """Create an interface_graph using an alignment graph and interface map, then
    remove nodes from the interface graph if those nodes are not in a specified
    nodeset.

    Parameters
    ----------
    alignment_graph: nx.DiGraph
        The alignment graph to use as a supergraph for interface graph
        construction. Nodes are monomer ID strings (foldseek format),
        edge (A, B) in G if A, B are foldseek-similar.
    interfaces: dict[tuple[str, str], alignment_utils.Interface]
        Mapping from PINDER dimers (pairs of PINDER monomer ID strings) to
        Interface objects
    coverage: float
        Proportion of interface that must be covered by foldseek alignment.
        Comes from GraphConfig.coverage_threshold.

    Returns
    -------
    interface_graph: nx.Graph
        Subgraph of alignment_graph where:
            Edges between A, B if A and B are similar at an interface
            Nodes are removed if not present in <used_interfaces>
    """
    # Alignment graphs have score, interface graphs have weight
    G = InterfaceGraph()  # This is an interface graph
    p1_interfaces: dict[str, set[frozenset[int]]] = {}
    p2_interfaces: dict[str, set[frozenset[int]]] = {}
    used_interfaces: set[str] = set()
    for (p1, p2), interface in tqdm(
        interfaces.items(),
        desc="Pulling alignments to interfaces",
        total=len(interfaces),
    ):
        p1_strip = system_to_pdb_chain_id(p1).rstrip(digits)
        p2_strip = system_to_pdb_chain_id(p2).rstrip(digits)
        G.add_node(p1_strip)
        G.add_node(p2_strip)
        if p1_strip not in p1_interfaces:
            p1_interfaces[p1_strip] = set()
        if p2_strip not in p2_interfaces:
            p2_interfaces[p2_strip] = set()
        p1_interfaces[p1_strip].add(frozenset(interface.indices1))
        p2_interfaces[p2_strip].add(frozenset(interface.indices2))
        used_interfaces.add(p1_strip)
        used_interfaces.add(p2_strip)

    for p1, idx_sets in tqdm(
        p1_interfaces.items(),
        desc="Pulling receptor alignments to interfaces",
        total=len(p1_interfaces),
    ):
        for al in alignment_graph.edges(p1, data=True):
            start, end = al[2]["alignment_idx"]
            aln_range = frozenset(range(start, end))
            intersections = {
                len(indices.intersection(aln_range)) / len(indices) >= coverage
                for indices in idx_sets
            }
            if any(intersections):
                G.add_edge(p1, al[1], weight=al[2]["score"])

    for p2, idx_sets in tqdm(
        p2_interfaces.items(),
        desc="Pulling ligand alignments to interfaces",
        total=len(p2_interfaces),
    ):
        for al in alignment_graph.edges(p2, data=True):
            start, end = al[2]["alignment_idx"]
            aln_range = frozenset(range(start, end))
            intersections = {
                len(indices.intersection(aln_range)) / len(indices) >= coverage
                for indices in idx_sets
            }
            if any(intersections):
                G.add_edge(p2, al[1], weight=al[2]["score"])

    G = clean_interface_graph(G, used_interfaces)
    G = G.to_undirected(reciprocal=True)
    return G


@timeit
def cluster_from_graph(
    graph: nx.Graph,
    community: bool = True,
    weight: str | None = "weight",
    seed: int | np.random.RandomState | None = None,
) -> list[set[str]]:
    """Computes clusters from a given graph using either asynchronous label propagation
    or connected_components

    Note that asynchronous label propagation is a stochastic algorithm and is therefore
    not guaranteed to return the same results if seed is not specified.

    Parameters
    ----------
    graph: nx.Graph
        An arbitrary networkX graph with edge weights <weights>. Nodes of this graph
        are expected to be monomer ids of type: str

    community: bool
        If True, use asynchronous label propagation. Else, return connected components

    weight: str | None
        The edge attribute for nx.Graph inputs representing the weight of an edge.
        If None, uses 1 for all weights. Used for AsynLPA clustering.
        Defaults to "weights".

    seed: int | np.random.RandomState | None
        The random seed for asynchronous label propagation.

    Returns
    -------
    clusters: List[Set[str]]
        A list of sets of nodes (str) corresponding to the output clusters
    """
    if community:
        return list(
            nx.algorithms.community.asyn_lpa_communities(
                graph, weight=weight, seed=seed
            )
        )
    else:
        return list(nx.algorithms.components.connected_components(graph))


def get_node_to_cluster_mapping(clusters: list[set[str]]) -> dict[str, int]:
    """Create a dictionary mapping node IDs to cluster IDs given a list of clusters.

    Parameters
    ----------
    clusters: list[set[str]]
        List of clusters, where each cluster is a set of node IDs.

    Returns
    -------
    dict:
        Dictionary mapping node IDs to cluster IDs.
    """
    return {
        node_id: cluster_id
        for cluster_id, cluster in enumerate(clusters)
        for node_id in cluster
    }


def clean_interface_graph(
    interface_graph: nx.Graph,
    used_interfaces: set[str],
) -> nx.Graph:
    """Remove nodes from the interface graph if those nodes are not in a specified
    nodeset.

    Parameters
    ----------
    interface_graph: nx.Graph
        The interface graph constructed from an alignment graph.
    used_interfaces: set[str]
        Set of unique monomer ID strings (foldseek format) in
        interfaces.keys()

    Returns
    -------
    interface_graph: nx.Graph
        Subgraph of interface_graph where:
            Edges between A, B if A and B are similar at an interface
            Nodes are removed if not present in <used_interfaces>
    """
    nodes_intgraph = set(interface_graph.nodes())
    nodes_to_delete = nodes_intgraph - used_interfaces
    interface_graph.remove_nodes_from(nodes_to_delete)
    return interface_graph


def load_graph_pickle(pkl_file: Path) -> nx.Graph:
    with open(pkl_file, "rb") as f:
        graph: nx.Graph = pkl.load(f)
        return graph


def construct_interface_graph(
    interface_pkl: Path = Path("interfaces.pkl"),
    output_dir: Path = Path("/tmp/graphs"),
    graph_config: GraphConfig = GraphConfig(),
) -> None:
    """Parameters:
    -----------
    foldseek_output: Path
        Path to foldseek output folder from the previous step
    interface_pkl: Path
        Path to the pickle file containing previously extracted interfaces.
    output_dir: str
        Path to directory in which to store generated graph pickle files.
    graph_config: GraphConfig
        Config object storing parameters used for constructing graphs.

    """
    output_hash_fp = output_dir / get_config_hash(graph_config)
    if not output_hash_fp.is_dir():
        output_hash_fp.mkdir(parents=True)

    interfaces = load_interface_pkl(interface_pkl)
    interfaces = {
        k: v
        for k, v in interfaces.items()
        if (len(v.indices1) >= graph_config.min_interface_length)
        and (len(v.indices2) >= graph_config.min_interface_length)
    }
    log.info(
        f"Found {len(interfaces)} interfaces with minimum length {graph_config.min_interface_length} residues"
    )
    with open(output_hash_fp / "min_length_interfaces.pkl", "wb") as f:
        pkl.dump(interfaces, f)


def construct_graph_from_alignment(
    alignment_file: Path = Path("/tmp/foldseek/foldseek_dbs/alignment.txt"),
    alignment_type: str = "foldseek",
    output_dir: Path = Path("/tmp/graphs"),
    graph_config: GraphConfig = GraphConfig(),
) -> nx.Graph:
    output_hash_fp = output_dir / get_config_hash(graph_config)
    if not output_hash_fp.is_dir():
        output_hash_fp.mkdir(parents=True)

    # get alignment graph with indices
    alignment_pqt = alignment_file.parent / "filtered_alignment.parquet"
    if not alignment_pqt.is_file():
        alignment_to_parquet(
            alignment_file,
            alignment_type=alignment_type,
            graph_config=graph_config,
            remove_original=True,
            use_cache=True,
        )
    aln_pqt = alignment_file.parent / "filtered_alignment.parquet"
    al_graph = get_alignment_graph_with_indices(aln_pqt)
    with open(output_hash_fp / f"{alignment_type}_alignment_graph.pkl", "wb") as f:
        pkl.dump(al_graph, f)

    log.info(
        f"Found {len(al_graph.nodes)} nodes and {len(al_graph.edges)} edges in cleaned {alignment_type} alignment graph"
    )
    return al_graph


def construct_interface_cleaned_graph(
    graph_pkl: Path | nx.Graph,
    interface_pkl: Path,
    alignment_type: str,
    graph_config: GraphConfig = GraphConfig(),
    use_cache: bool = True,
) -> None:
    interface_graph_pkl = (
        interface_pkl.parent / f"cleaned_{alignment_type}_alignment_graph.pkl"
    )
    if interface_graph_pkl.is_file() and use_cache:
        log.info(f"Interface graph {interface_graph_pkl} exists. Skipping...")
        return
    interfaces = load_interface_pkl(interface_pkl)
    if isinstance(graph_pkl, Path):
        alignment_graph = load_graph_pickle(graph_pkl)
    elif isinstance(graph_pkl, nx.Graph):
        alignment_graph = graph_pkl
    else:
        raise TypeError(
            f"graph_pkl {graph_pkl} should be a Path to pickle file or nx.Graph!"
        )
    # get interface graph
    interface_graph = get_interface_graph(
        alignment_graph, interfaces, coverage=graph_config.coverage_threshold
    )
    with open(interface_graph_pkl, "wb") as f:
        pkl.dump(interface_graph, f)


def construct_interface_alignment_graph(
    interface_pkl: Path,
    alignment_file: Path = Path("/tmp/foldseek/foldseek_dbs/alignment.txt"),
    alignment_type: str = "foldseek",
    output_dir: Path = Path("/tmp/graphs"),
    graph_config: GraphConfig = GraphConfig(),
    use_cache: bool = True,
) -> None:
    interface_graph_pkl = (
        interface_pkl.parent / f"cleaned_{alignment_type}_alignment_graph.pkl"
    )
    if interface_graph_pkl.is_file() and use_cache:
        log.info(f"Interface graph {interface_graph_pkl} exists. Skipping...")
        return
    G = construct_graph_from_alignment(
        alignment_file, alignment_type, output_dir, graph_config
    )
    construct_interface_cleaned_graph(
        graph_pkl=G,
        interface_pkl=interface_pkl,
        alignment_type=alignment_type,
        graph_config=graph_config,
        use_cache=use_cache,
    )


class InterfaceGraph(nx.MultiDiGraph):  # type: ignore
    def to_undirected_class(self) -> Callable[[], MaxGraph]:
        return MaxGraph


class MaxGraph(nx.Graph):  # type: ignore
    default_val = float("-inf")  # Default comparison value for aggregation
    agg_field = "weight"

    def __init__(self, incoming_graph_data: Any = None, **attr: Any):
        """Initialize an new MaxGraph instance."""
        super().__init__(incoming_graph_data=incoming_graph_data, **attr)

    def add_edges_from(
        self, ebunch_to_add: list[tuple[str, str, dict[str, Any]]], **attr: Any
    ) -> None:
        """Add edges, but only if the agg_field is better than previous"""
        for edge in ebunch_to_add:
            if len(edge) >= 3:
                dd = edge[-1] | attr
                self.add_edge(edge[0], edge[1], **dd)
            elif len(edge) == 2:
                self.add_edge(edge[0], edge[1], **attr)
            else:
                raise nx.NetworkXError(f"Edge tuple {edge} in unrecognized format")

    def add_edge(self, u_of_edge: str, v_of_edge: str, **attr: Any) -> None:
        """Add an edge between u and v, IFF the agg_field is better than existing."""
        if not super().has_edge(u_of_edge, v_of_edge) or gt(
            attr[self.agg_field],
            super().get_edge_data(u_of_edge, v_of_edge)[self.agg_field],
        ):
            super().add_edge(u_of_edge, v_of_edge, **attr)
