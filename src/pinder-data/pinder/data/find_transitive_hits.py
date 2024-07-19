from __future__ import annotations

import pickle
import time
from collections import defaultdict
from datetime import date
from functools import reduce
from itertools import product, chain, combinations_with_replacement
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import pinder.data.graph_utils as gu
from pinder.core.index.utils import setup_logger
from pinder.core.utils.timer import timeit
from pinder.data.csv_utils import read_csv_non_default_na
from pinder.data.config import ClusterConfig, GraphConfig, get_config_hash
from pinder.data.get_clusters import load_cluster_cache
from pinder.data.pipeline.constants import CONSIDER_LEAKED


log = setup_logger(__name__)


def get_potential_representatives(
    metadata: pd.DataFrame,
    config: ClusterConfig = ClusterConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get potential representatives from the metadata based on specified criteria.

    Args:
        metadata (pd.DataFrame): The metadata containing information about the dataset.
        config (ClusterConfig): The `ClusterConfig` object containing config for selecting the test set.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - The first DataFrame contains the potential representatives based on the specified criteria.
            - The second DataFrame contains the potential representatives after a specific date.

    """
    dataset = metadata.copy()

    dataset["date"] = dataset["date"].map(lambda x: date.fromisoformat(x))

    query_fields = [
        f"oligomeric_count == {config.oligomeric_count}",
        f"method == '{config.method}'",
        f"interface_atom_gaps_4A <= {config.interface_atom_gaps_4A}",
        f"length1 >= {config.min_chain_length}",
        f"length2 >= {config.min_chain_length}",
        f"num_atom_types >= {config.min_atom_types}",
        f"label == '{config.prodigy_label}'",
        f"resolution <= {config.resolution_thr}",
        f"number_of_components_1 == {config.number_of_components}",
        f"number_of_components_2 == {config.number_of_components}",
        f"max_var_1 <= {config.max_var_thr}",
        f"max_var_2 <= {config.max_var_thr}",
    ]
    query_str: str = " & ".join([f"({query})" for query in query_fields])
    test = dataset.query(query_str)
    af2mm_test = test[(test["date"] > date.fromisoformat(config.alphafold_cutoff_date))]
    return test, af2mm_test


def get_test_conversion_dicts(
    test_index: pd.DataFrame,
    cluster_key: str = "cluster_id",
) -> tuple[set[str], dict[tuple[int, int], str]]:
    """Convert the test index data into a set of test system IDs and a dictionary mapping cluster IDs to test system IDs.

    Parameters:
        test_index (pandas.DataFrame): The test index data.
        cluster_key (str, optional): The column name for the cluster ID. Defaults to "cluster_id".

    Returns:
        tuple: A tuple containing the set of test system IDs and the dictionary mapping cluster IDs to test system IDs.
    """
    test_system_ids = set(test_index["id"])
    cluster_to_test_systems = defaultdict(set)
    for sys_id, cluster_id in zip(test_index["id"], test_index[cluster_key]):
        cluster_to_test_systems[cluster_id].add(sys_id)
    return test_system_ids, cluster_to_test_systems  # type: ignore


def get_proto_splits_pindex(
    index: pd.DataFrame,
    metadata: pd.DataFrame,
    cluster_key: str = "cluster_id",
    config: ClusterConfig = ClusterConfig(),
) -> pd.DataFrame:
    """Get the test-train split for the index based on the cluster_id"""
    filtered_pindex = index[
        (index["cluster_id_R"] > -1) & (index["cluster_id_L"] > -1)
    ].reset_index(drop=True)
    log.info(
        (
            f"Found {filtered_pindex.shape[0]} clustered systems in "
            f"{index.shape[0]} index entries"
        )
    )
    test_meta, _ = get_potential_representatives(metadata, config=config)
    test_index = filtered_pindex[filtered_pindex["id"].isin(test_meta["id"])]

    log.info(
        (
            f"Chose {test_index.shape[0]} potential test systems "
            f"from {filtered_pindex.shape[0]} clustered systems."
        )
    )

    _, cluster_to_test_systems = get_test_conversion_dicts(test_index, cluster_key)

    filtered_pindex.loc[filtered_pindex["id"].isin(test_meta["id"]), "split"] = (
        "proto-test"
    )
    filtered_pindex.loc[~filtered_pindex["id"].isin(test_meta["id"]), "split"] = (
        "proto-train"
    )
    return filtered_pindex, cluster_to_test_systems


def get_leakage_dict(
    pinder_dir: Path,
    graph_type: str,
    config: ClusterConfig = ClusterConfig(),
    graph_config: GraphConfig = GraphConfig(),
    use_cache: bool = True,
    af2_transitive_hits: bool = False,
) -> None:
    chk_dir = pinder_dir / "cluster" / get_config_hash(config)
    if af2_transitive_hits:
        thresh_label = "{:.2f}".format(
            config.foldseek_af2_difficulty_threshold
        ).replace(".", "")
        leakage_checkpoint = (
            chk_dir / f"{graph_type}_af2_lddt{thresh_label}_leakage_dict.pkl"
        )
        potential_leaks_chkpt = (
            chk_dir / f"{graph_type}_af2_lddt{thresh_label}_potential_leaks.pkl"
        )
    else:
        leakage_checkpoint = chk_dir / f"{graph_type}_leakage_dict.pkl"
        potential_leaks_chkpt = chk_dir / f"{graph_type}_potential_leaks.pkl"

    if leakage_checkpoint.is_file() and use_cache:
        log.info(
            f"Skipping {graph_type} split leakage search, {leakage_checkpoint} checkpoint exists..."
        )
        return
    index = read_csv_non_default_na(chk_dir / "index.2.csv.gz", dtype={"pdb_id": "str"})
    metadata = read_csv_non_default_na(
        pinder_dir / "metadata.2.csv.gz", dtype={"entry_id": "str"}
    )
    filtered_pindex, cluster_to_test_systems = get_proto_splits_pindex(
        index,
        metadata,
        config=config,
    )
    systems_proto_test = set(
        filtered_pindex[filtered_pindex["split"] == "proto-test"]["id"].values.tolist()
    )
    systems_proto_train = set(
        filtered_pindex[filtered_pindex["split"] == "proto-train"]["id"].values.tolist()
    )

    foldseek_communities_cache = chk_dir / "foldseek_communities.pkl"
    graph_clusters = load_cluster_cache(foldseek_communities_cache)
    graph_fp = pinder_dir / "graphs" / get_config_hash(graph_config)
    mmseqs_graph_pkl = graph_fp / "cleaned_mmseqs_alignment_graph.pkl"
    foldseek_graph_pkl = graph_fp / "cleaned_foldseek_alignment_graph.pkl"
    if graph_type == "foldseek":
        graph_pkl = foldseek_graph_pkl
        if af2_transitive_hits:
            score = config.foldseek_af2_difficulty_threshold
        else:
            score = config.foldseek_edge_threshold
    elif graph_type == "mmseqs":
        graph_pkl = mmseqs_graph_pkl
        score = config.mmseqs_edge_threshold

    graph = gu.load_graph_pickle(graph_pkl)
    n2c = gu.get_node_to_cluster_mapping(graph_clusters)
    split_leakage = find_split_leakage(
        systems_proto_test,
        systems_proto_train,
        graph,
        n2c,
        config.depth_limit,
        edge_threshold=score,
        max_node_degree=config.max_node_degree,
        potential_leaks_chkpt=potential_leaks_chkpt,
        use_cache=use_cache,
    )
    with open(leakage_checkpoint, "wb") as f:
        pickle.dump(split_leakage, f)


def get_transitive_hits(
    pinder_dir: Path,
    config: ClusterConfig = ClusterConfig(),
    graph_config: GraphConfig = GraphConfig(),
    test_systems_output: str = "test_sys_table.csv",
    deleak_map_output: str = "transitive_hits_mapping.csv",
    use_cache: bool = True,
    af2_transitive_hits: bool = False,
) -> None:
    chk_dir = pinder_dir / "cluster" / get_config_hash(config)
    transitive_hits_checkpoint = chk_dir / test_systems_output
    if transitive_hits_checkpoint.is_file() and use_cache:
        log.info(
            f"Skipping transitive hit search, {transitive_hits_checkpoint} checkpoint exists..."
        )
        return
    index = read_csv_non_default_na(chk_dir / "index.2.csv.gz", dtype={"pdb_id": "str"})
    metadata = read_csv_non_default_na(
        pinder_dir / "metadata.2.csv.gz", dtype={"entry_id": "str"}
    )
    filtered_pindex, cluster_to_test_systems = get_proto_splits_pindex(
        index,
        metadata,
        config=config,
    )
    if af2_transitive_hits:
        graph_type = "foldseek"
        thresh_label = "{:.2f}".format(
            config.foldseek_af2_difficulty_threshold
        ).replace(".", "")
        leakage_checkpoint = (
            chk_dir / f"{graph_type}_af2_lddt{thresh_label}_leakage_dict.pkl"
        )
        leakage_checkpoints = [leakage_checkpoint]
    else:
        leakage_checkpoints = [
            chk_dir / f"{graph_type}_leakage_dict.pkl"
            for graph_type in ["foldseek", "mmseqs"]
        ]
    leakage_dicts = []
    for chk in leakage_checkpoints:
        if not chk.is_file():
            log.warning(
                f"Split leakage checkpoint {chk} not found! Generating split leakage..."
            )
            graph_type = chk.stem.split("_leakage_dict")[0].split("_af2_lddt")[0]
            get_leakage_dict(
                pinder_dir=pinder_dir,
                graph_type=graph_type,
                config=config,
                graph_config=graph_config,
                use_cache=use_cache,
                af2_transitive_hits=af2_transitive_hits,
            )
        with open(chk, "rb") as f:
            split_leakage = pickle.load(f)
            leakage_dicts.append(split_leakage)
    log.info("Finished loading leakage_dicts")
    system_hits_dict: dict[str, set[str]] = reduce(
        deep_merge_dict, leakage_dicts, dict()
    )
    system_hits_list = []
    for self_system, target_systems in system_hits_dict.items():
        system_hits_list.append(
            {
                "id": self_system,
                "neighbors": ",".join(list(target_systems)),
            }
        )
    system_hits_df = pd.DataFrame(system_hits_list)
    system_hits_df.to_csv(chk_dir / deleak_map_output, index=False)

    system_hits_dict_clean_num = {}
    for k, v in system_hits_dict.items():
        if CONSIDER_LEAKED in v:
            system_hits_dict_clean_num[k] = np.inf
        else:
            system_hits_dict_clean_num[k] = len(v)

    system_to_dg1_clust = {}
    for cluster_id, test_systems in cluster_to_test_systems.items():
        for test_system in test_systems:
            if test_system in system_hits_dict:
                if CONSIDER_LEAKED in system_hits_dict[test_system]:
                    system_to_dg1_clust[test_system] = np.inf
                else:
                    system_to_dg1_clust[test_system] = len(
                        system_hits_dict[test_system] | test_systems
                    )

    # unused
    # test_system_to_clust = {}
    # for cluster_id, test_systems in cluster_to_test_systems.items():
    #     for test_system in test_systems:
    #         test_system_to_clust[test_system] = test_systems

    filtered_pindex["depth_2_hits"] = filtered_pindex["id"].map(
        lambda x: system_hits_dict_clean_num.get(x, np.nan)
    )
    filtered_pindex["depth_2_hits_with_comm"] = filtered_pindex["id"].map(
        lambda x: system_to_dg1_clust.get(x, np.nan)
    )
    test_sys_table = filtered_pindex[filtered_pindex["split"] == "proto-test"]
    test_sys_table.to_csv(transitive_hits_checkpoint, index=False)


def cluster_leaks(
    source: str,
    graph: nx.Graph,
    node_to_cluster: dict[str, int],
    depth: int,
) -> set[str]:
    """Find all nodes x in a graph such that:
        1. there exists a path of length l <= <depth> from source to x
        2. source and x are in different clusters

    Parameters
    ----------
    source: str
        The source node
    graph: nx.Graph
        The target graph. Must contain <node>
    node_to_cluster: Dict[str, int]
        Map from nodes to cluster IDs
    depth: int
        Maximum allowed path length.

    Returns
    -------
    Set[str]
        The set of nodes in the "<depth>-neighborhood" of source
    """
    source_cluster = node_to_cluster[source]
    neighbor_nodes = nx.single_source_shortest_path_length(
        graph, source, cutoff=depth
    ).keys()
    nodes = set(neighbor_nodes)
    return nodes


@timeit
def batch_cluster_leaks(
    source_set: set[str],
    graph: nx.Graph,
    node_to_cluster: dict[str, int],
    depth: int,
    max_node_degree: int | None = 1_000,
) -> dict[str, set[str]]:
    """Find cluster leaks for all nodes in node_set.

    For each node, finds the set of all neighbors in graph within depth
    hops of node that are in different clusters.

    Applies checks to limit computation. If a node fails any
    check, the corresponding value will be CONSIDER_LEAKED.

    Parameters
    ----------
    source_set: Set[str]
        A set of target sources
    graph: nx.Graph
        The target graph. Must contain all nodes in <node_set>
    node_to_cluster: Dict[str, int]
        Map from nodes to cluster IDs
    depth: int
        Maximum allowed path length.
    max_node_degree: int
        Maximum allowed node degree. Default 1000

    Returns
    -------
    Dict[str, Set[str]]
        Mapping from a node to the set of nodes in the
        node's "<depth>-neighborhood"
    """
    # First determine for which nodes we can avoid finding paths
    if max_node_degree is not None:
        degrees = dict(graph.degree(source_set))
        failing_nodes = []
        passing_nodes = []
        for n, d in tqdm(
            degrees.items(),
            desc=f"Extracting nodes within max_node_degree={max_node_degree}",
            total=len(degrees),
        ):
            if d <= max_node_degree:
                passing_nodes.append(n)
            else:
                failing_nodes.append(n)
    else:
        passing_nodes = list(source_set)
        failing_nodes = []

    # Compute the neighbors
    matches = []
    for source in tqdm(passing_nodes):
        source_match = cluster_leaks(
            source=source, graph=graph, node_to_cluster=node_to_cluster, depth=depth
        )
        matches.append(source_match)

    # Build the dictionary
    return dict(zip(passing_nodes, matches)) | {
        n: {CONSIDER_LEAKED} for n in failing_nodes
    }


def get_leak_map_for_id(
    pure_id: str,
    pure_map_forward: dict[str, tuple[str, str]],
    pure_map_backward: dict[frozenset[str], set[str]],
    corrupt_map_backward: dict[frozenset[str], set[str]],
    all_pure_pairs: set[frozenset[str]],
    all_corrupt_pairs: set[frozenset[str]],
    potential_leaks: dict[str, set[str]],
) -> set[str]:
    """Do the multi-step mapping from:
    pure system -> pure pair of nodes
    pure pair of nodes -> possibly corrupt leaks for each node
    possibly corrupt leaks -> corrupt systems (by intersection)
    """
    v, u = pure_map_forward[pure_id]
    v_leaks = potential_leaks.get(v, set())  # Say this has size n
    u_leaks = potential_leaks.get(u, set())  # Say this has size m
    # Possible that one or more of the nodes was skipped
    if CONSIDER_LEAKED in v_leaks or CONSIDER_LEAKED in u_leaks:
        return {CONSIDER_LEAKED}
    # potential_leak_pairs = filtered_product(v_leaks, u_leaks, all_corrupt_pairs)
    check_pairs = all_corrupt_pairs.union(all_pure_pairs)
    # O(min(n*m, k)), where k = |check_pairs|
    potential_leak_pairs = intersection_unordered_product(v_leaks, u_leaks, check_pairs)
    corrupt = [corrupt_map_backward.get(k, set()) for k in potential_leak_pairs]
    corrupt_test = [pure_map_backward.get(k, set()) for k in potential_leak_pairs]
    corrupt = corrupt + corrupt_test
    pairs: set[str] = reduce(set.union, corrupt, set())
    return pairs


@timeit
def map_leak_pairs(
    pure_split: set[str],
    pure_map_forward: dict[str, tuple[str, str]],
    pure_map_backward: dict[frozenset[str], set[str]],
    corrupt_map_backward: dict[frozenset[str], set[str]],
    all_pure_pairs: set[frozenset[str]],
    all_corrupt_pairs: set[frozenset[str]],
    potential_leaks: dict[str, set[str]],
) -> dict[str, set[str]]:
    mapped_pairs: dict[str, set[str]] = {
        pure_id: get_leak_map_for_id(
            pure_id,
            pure_map_forward,
            pure_map_backward,
            corrupt_map_backward,
            all_pure_pairs,
            all_corrupt_pairs,
            potential_leaks,
        )
        for pure_id in tqdm(pure_split, desc="Mapping potential leakage pairs per ID")
    }
    return mapped_pairs


def find_split_leakage(
    pure_split: set[str],
    corrupt_split: set[str],
    graph: nx.Graph,
    node_to_cluster: dict[str, int],
    depth: int,
    edge_threshold: float = 0.65,
    max_node_degree: int = 1_000,
    potential_leaks_chkpt: Path | None = None,
    use_cache: bool = True,
) -> dict[str, set[str]]:
    """For systems in pure_split, determine whether there are systems in
    corrupt_split that leak into the pure_split.

    Note: leaking system <--> transitive hit

    A system `c:= {u, v}` in `corrupt_split` is a leaking system for system
    `p:= {s, t}` in `pure_split` iff any of the following are true:
        1. All of the following are true:
            - `u` and `s` are in different clusters
            - There exists a path between `u` and `s` of length <= `depth`
            - `v` and `t` are in different clusters
            - There exists a path between `v` and `t` of length <= `depth`
        2. All of the following are true:
            - `u` and `t` are in different clusters
            - There exists a path between `u` and `t` of length <= `depth`
            - `v` and `s` are in different clusters
            - There exists a path between `v` and `s` of length <= `depth`

    Method:
        1. Create a map from test systems to graph node pairs
        2. Do any filtering required on the graph
        3. Find cluster_leaks for all nodes in these pairs
        4. For train systems, create a nested map from graph node pairs
            to sets of train systems
        6. For each system x in test:
            a: map x to a graph test pair
            b: find all graph train pairs containing at least one member
               of the graph test pair using the "adjacency map"
            c: map from these graph train pair to train systems
            d: these are the inter_split connections
        7. Profit

    Parameters
    ----------
    pure_split: Set[str]
        The set of PINDER System IDs in the pure split (e.g., test)
    corrupt_split: Set[str]
        The set of PINDER System IDs in the corrupt split (e.g., train)
    graph: nx.Graph
        The foldseek-similarity monomer graph
    node_to_cluster: Dict[str, int]
        The map from graph nodes to cluster IDs
    depth: int
        The maximum path length to travel looking for leakage.
    max_node_degree: int
        The node degree at which we assume there is leakage. This is
        to save compute. Default: 1000

    Returns
    -------
    Dict[str, Set[str]]
        The map from pure_split IDs to corrupt_split IDs, indicating leakage.
    """
    ## We will need two maps. Ahoy!
    pure_map_forward = map_systems_to_fsid_pairs(pure_split)
    corrupt_map_backward = map_fsid_pair_to_systems(corrupt_split)
    pure_map_backward = map_fsid_pair_to_systems(pure_split)

    # Useful to also have the set of all node pairs associated with corrupt set
    all_corrupt_pairs = set(corrupt_map_backward.keys())
    all_pure_pairs = set(pure_map_backward.keys())

    # Filter the graph, if necessary
    start = time.time()
    # TODO: filtering this way does not play nicely with multiprocessing
    subgraph = nx.subgraph_view(
        graph,
        filter_edge=lambda u, v: graph[u][v]["weight"] >= edge_threshold,
    )
    log.info(f"graph filtering succeeded: {time.time() - start:.2f}s")

    # Find potential leaks
    all_pure_nodes = {n for pair in pure_map_forward.values() for n in pair}

    if (
        isinstance(potential_leaks_chkpt, Path)
        and potential_leaks_chkpt.is_file()
        and use_cache
    ):
        log.info(f"{potential_leaks_chkpt} found, skipping batch_cluster_leaks...")
        with open(potential_leaks_chkpt, "rb") as f:
            potential_leaks = pickle.load(f)
    else:
        potential_leaks = batch_cluster_leaks(
            all_pure_nodes,
            subgraph,
            node_to_cluster,
            depth,
            max_node_degree=max_node_degree,
        )
        if potential_leaks_chkpt is not None:
            with open(potential_leaks_chkpt, "wb") as f:
                pickle.dump(potential_leaks, f)

    mapped_pairs: dict[str, set[str]] = map_leak_pairs(
        pure_split=pure_split,
        pure_map_forward=pure_map_forward,
        pure_map_backward=pure_map_backward,
        corrupt_map_backward=corrupt_map_backward,
        all_pure_pairs=all_pure_pairs,
        all_corrupt_pairs=all_corrupt_pairs,
        potential_leaks=potential_leaks,
    )
    return mapped_pairs


@timeit
def map_systems_to_fsid_pairs(system_ids: set[str]) -> dict[str, tuple[str, str]]:
    """Get a map from PINDER System ID strings to pairs of Foldseek monomer ID strings"""
    return {id: gu.system_id_to_fsid_pair(id) for id in system_ids}


@timeit
def map_fsid_pair_to_systems(system_ids: set[str]) -> dict[frozenset[str], set[str]]:
    """Construct a hierarchical map from Foldseek monomer ID pairs to PINDER System IDs

    This map is insensitive to order.
    Avoid using defaultdict, which is actually quite slow
    """
    fsid_pair_to_systems: dict[frozenset[str], set[str]] = dict()
    for id in system_ids:
        # Technically, this can be a singleton, but that's ok
        unordered_pair = frozenset(gu.system_id_to_fsid_pair(id))

        if not unordered_pair in fsid_pair_to_systems.keys():
            fsid_pair_to_systems[unordered_pair] = set()

        # Add the items
        fsid_pair_to_systems[unordered_pair].add(id)
    return fsid_pair_to_systems


def deep_merge_dict(
    a: dict[str, set[str]],
    b: dict[str, set[str]],
) -> dict[str, set[str]]:
    """Merge two dictionaries in which values are Set objects using `set.update`"""
    ret = dict(a)
    for k, v in b.items():
        ret.setdefault(k, set()).update(v)
    return ret


def intersection_unordered_product(
    A: set[str],
    B: set[str],
    C: set[frozenset[str]],
) -> set[frozenset[str]]:
    """Compute the intersection between C and the unordered set product of A and B:= U.

    Attempts to do this efficiently by determining the size of U before actually
    computing U, then iterating over the smaller of U or C, while checking inclusion
    in the larger.

    Specifically:
        1. determines the size of U, the unordered cartesian product of A and B
        2. if U is smaller than C, computes U and iterates over it, checking inclusion
           in C.
        3. if U is larger than C, iterates over C and checks inclusion in A and B.

    NB: our elements are of size 1 or 2, so to check inclusion we use 0 or -1 as our
    indices. If our elements were of another size, this code would not work!

    Parameters
    ----------
    A: set[str]
        The first set in the possible product
    B: set[str]
        The second set in the possible product
    C: set[frozenset[str]]
        The set that we are intersecting with A x B

    Returns
    -------
    set[frozenset[str]
        The set intersection between C and AxB (unordered)
    """
    size = len_unordered_set_product(A, B)

    if len(C) > size:
        return {frozenset(e) for e in unordered_set_product(A, B) if frozenset(e) in C}
    else:
        return {
            frozenset(e)
            for e in C
            if (tuple(e)[0] in A and tuple(e)[-1] in B)
            or (tuple(e)[-1] in A and tuple(e)[0] in B)
        }


def unordered_set_product(A: set[str], B: set[str]) -> Iterable[tuple[str]]:
    """Compute the unordered cartesian product of sets A and B.

    We define the unordered set product U as a subset of P:=product(A, B), where if
    x:=(a,b) is in U, then y:=(b,a) is not in U, even if y is in P.

    The goal of this method is to compute the unordered set product of A and B
    in the most efficient way possible, ideally without computing the entire product.

    Given two sets, A and B, with intersection |A∩B|:=I, where |A|=n, |B|=m, and
    |I|=i. The cartesian product of A and B has size n*m, but the *unordered*
    cartesian product may be smaller.

    To compute the unordered cartesian product:
        We want the union of:
            combinations w repl., len 2 of I     (i+1C2 = (i+1)(i)/2)
            product of A-I with I                ((n-i) * i))
            product of B-I with I                ((m-i) * i))
            product of A-I with B-I              ((n-i) * (m-i))

    The size of this union is n*m - (i**2 - i)//2

    Parameters
    ----------
    A: set[str]
        The first product set
    B: set[str]
        The second product set

    Returns
    -------
    set[tuple[str]]
        The unordered cartesian product of A and B.
    """
    I = A.intersection(B)
    A_unique = A - I
    B_unique = B - I

    set_prod: Iterable[tuple[str]] = chain(
        combinations_with_replacement(I, 2),  # type: ignore
        product(A_unique, I),  # type: ignore
        product(B_unique, I),  # type: ignore
        product(A_unique, B_unique),  # type: ignore
    )
    return set_prod


def len_unordered_set_product(A: set[str], B: set[str]) -> int:
    """Compute the size of the unordered cartesian product of sets A and B.

    We define the unordered set product U as a subset of P:=product(A, B), where if
    x:=(a,b) is in U, then y:=(b,a) is not in U, even if y is in P.

    Given two sets, A and B, with intersection |A∩B|:=I, where |A|=n, |B|=m, and
    |I|=i. The cartesian product of A and B has size n*m, but the *unordered*
    cartesian product may be smaller.

    To compute the unordered cartesian product:
        We want the union of:
            combinations w repl., len 2 of I     (i+1C2 = (i+1)(i)/2)
            product of A-I with I                ((n-i) * i))
            product of B-I with I                ((m-i) * i))
            product of A-I with B-I              ((n-i) * (m-i))

    The size of this union is n*m - (i**2 - i)//2. Note that (i**2 - i) is
    always even for all integer i.

    Parameters
    ----------
    A: set[str]
        The first product set
    B: set[str]
        The second product set

    Returns
    -------
    int
        The size of AxB (unordered)
    """
    n = len(A)
    m = len(B)
    i = len(A.intersection(B))
    return n * m - (i**2 - i) // 2  # Note that i**2 - i is always even for all i
