import networkx as nx
import numpy as np
import pytest
from itertools import combinations
from unittest.mock import patch, MagicMock

from pinder.data.alignment_utils import Interface
from pinder.data.foldseek_utils import alignment_to_parquet
from pinder.data.config import get_config_hash, ContactConfig, GraphConfig
from pinder.data import graph_utils


def gen_caveman():
    graph = nx.caveman_graph(5, 5)
    nx.set_edge_attributes(graph, 1.42, "weight")
    return graph


def gen_windmill():
    graph = nx.windmill_graph(5, 5)
    nx.set_edge_attributes(graph, 1.42, "weight")
    return graph


def gen_interface_simple():
    nodes = [
        "A_a",
        "B_b",
        "C_c",
        "D_d",
        "E_e",
        "F_f",
        "G_g",
        "H_h",
    ]
    edges = [
        (nodes[0], nodes[1]),
        (nodes[2], nodes[3]),
    ]

    graph = nx.Graph()

    for node in nodes:
        graph.add_node(node)

    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    return graph


CAVEMAN = gen_caveman()
WINDMILL = gen_windmill()
INTERFACE_SIMPLE = gen_interface_simple()


@pytest.mark.parametrize(
    "pindex_id, expected_str",
    [
        ("7nsg__A1_P43005", "7nsg_A1"),
        ("6wwe__B1_A0A287AZ37", "6wwe_B1"),
        ("4wwi__D1_P01860", "4wwi_D1"),
        ("7cma__A2_A0A2X0TC55", "7cma_A2"),
    ],
)
def test_system_to_pdb_chain_id(pindex_id, expected_str):
    """
    Test the system_to_pdb_chain_id function
    """
    assert graph_utils.system_to_pdb_chain_id(pindex_id) == expected_str


@pytest.mark.parametrize(
    "graph, community, num_clust_expected",
    [
        (CAVEMAN, False, 5),
        (CAVEMAN, True, 5),
        (WINDMILL, False, 1),
        (WINDMILL, True, 5),
        (INTERFACE_SIMPLE, False, 6),
        (INTERFACE_SIMPLE, True, 6),
    ],
)
def test_cluster_from_graph(graph, community, num_clust_expected):
    clusters = graph_utils.cluster_from_graph(graph, community=community, seed=42)

    # Check clusters produced properly
    assert clusters, "no clusters produced"
    assert all(c for c in clusters), "found empty clusters"

    # Check cluster size
    assert (
        len(clusters) == num_clust_expected
    ), f"expected {num_clust_expected} clusters, got {len(clusters)}"


# Test case 1: Empty clusters
def test_get_node_to_cluster_mapping_empty_clusters():
    clusters = []
    expected_result = {}
    result = graph_utils.get_node_to_cluster_mapping(clusters)
    for k, v in result.items():
        assert expected_result[k] == v


# Test case 2: Single node in a single cluster
def test_get_node_to_cluster_mapping_single_node_single_cluster():
    clusters = [{"A"}]
    expected_result = {"A": 0}
    result = graph_utils.get_node_to_cluster_mapping(clusters)
    for k, v in result.items():
        assert expected_result[k] == v


# Test case 3: Multiple nodes in a single cluster
def test_get_node_to_cluster_mapping_multiple_nodes_single_cluster():
    clusters = [{"A", "B", "C"}]
    expected_result = {"A": 0, "B": 0, "C": 0}
    result = graph_utils.get_node_to_cluster_mapping(clusters)
    for k, v in result.items():
        assert expected_result[k] == v


# Test case 4: Multiple nodes in multiple clusters
def test_get_node_to_cluster_mapping_multiple_nodes_multiple_clusters():
    clusters = [{"A", "B"}, {"C", "D"}, {"E", "F"}]
    expected_result = {"A": 0, "B": 0, "C": 1, "D": 1, "E": 2, "F": 2}
    result = graph_utils.get_node_to_cluster_mapping(clusters)
    for k, v in result.items():
        assert expected_result[k] == v


# Test case 5: Duplicate nodes in clusters
def test_get_node_to_cluster_mapping_duplicate_nodes():
    """Currently we expect values to be overwritten."""
    clusters = [{"A", "B"}, {"B", "C"}, {"C", "D"}]
    expected_result = {"A": 0, "B": 1, "C": 2, "D": 2}
    result = graph_utils.get_node_to_cluster_mapping(clusters)
    for k, v in result.items():
        assert expected_result[k] == v


@pytest.mark.parametrize(
    "upper_thresh, expected_values",
    [  # now this is a directed graph
        (
            0.95,
            {
                "num_nodes": 4,  # upper thresh removes 2
                "num_edges": 4,  # upper thresh removes 1
                "edge_weights": {
                    ("6wwi_A", "6wwi_B"): pytest.approx(0.8339),
                    ("6wwi_B", "6wwi_A"): pytest.approx(0.8222),
                    ("4wwi_A", "4znc_A"): pytest.approx(0.9305),
                },
            },
        ),
        (
            np.inf,
            {
                "num_nodes": 6,  # orphan nodes still removed (1)
                "num_edges": 6,
                "edge_weights": {
                    ("6wwi_A", "6wwi_B"): pytest.approx(0.8339),
                    ("6wwi_B", "6wwi_A"): pytest.approx(0.8222),
                    ("4znc_A", "4wwi_A"): pytest.approx(0.9305),
                    ("4wwi_A", "4znc_A"): pytest.approx(0.9699),
                    ("4znc_D", "4wwi_D"): pytest.approx(0.9763),
                    ("4wwi_D", "4znc_D"): pytest.approx(0.9624),
                    ("4wwi_A", "4znc_A"): pytest.approx(0.9699),
                },
            },
        ),
    ],
)
def test_alignment_graph_with_indices(upper_thresh, expected_values, pinder_data_cp):
    alignment_file = pinder_data_cp / "clustering" / "3system-graph" / "alignment.txt"
    graph_config = GraphConfig(
        upper_threshold=upper_thresh, score_threshold=0.7, min_alignment_length=12
    )
    alignment_to_parquet(
        alignment_file,
        alignment_type="foldseek",
        graph_config=graph_config,
        remove_original=False,
        use_cache=False,
    )
    aln_pqt = alignment_file.parent / "filtered_alignment.parquet"
    graph = graph_utils.get_alignment_graph_with_indices(aln_pqt)
    nodes = graph.nodes
    edges = graph.edges

    assert nodes, "no nodes in graph"
    assert edges, "no edges in graph"

    assert (
        len(nodes) == expected_values["num_nodes"]
    ), f"Wrong num nodes: {len(nodes)}, expected {expected_values['num_nodes']}"

    assert (
        len(edges) == expected_values["num_edges"]
    ), f"Wrong num edges: {len(edges)}, expected {expected_values['num_edges']}"

    # weights are now alignment score
    edge2weights = {(e[0], e[1]): e[-1]["score"] for e in edges.data()}

    for k, v in expected_values["edge_weights"].items():
        assert (
            edge2weights[k] == v
        ), f"Wrong weight for {k}: {edge2weights[k]}, expected {v}"


def test_clean_interface_graph():
    # Note that we test get_interface_graph separately, so only test node removal
    nodes = [
        "7nsg_A",
        "6wwe_B",
        "4wwi_D",
        "7cma_A",
        "x1pq_B",
        "m0f9_C",
        "d3zq_D",
        "k8la_E",
        "r2yo_F",
        "e9gc_G",
        "v5hm_H",
        "i7uw_I",
        "t4ne_J",
    ]
    to_keep = [
        "6wwe_B",
        "4wwi_D",
        "7cma_A",
        "x1pq_B",
        "m0f9_C",
    ]
    graph = nx.complete_graph(nodes)
    result_graph = graph_utils.clean_interface_graph(graph, set(to_keep))
    result_nodes = list(result_graph.nodes())
    result_edges = list(result_graph.edges())

    assert len(result_nodes) == len(to_keep), "Did not keep expected number of nodes"

    assert len(result_edges) == len(
        list(combinations(to_keep, 2))
    ), "Did not keep the expected number of edges"

    for n in result_nodes:
        assert n in to_keep, f"Expected {n} in nodes, but it wasn't there!"


def test_construct_interface_graph(pinder_data_cp):
    graph_dir = pinder_data_cp / "graphs"
    graph_config = GraphConfig()
    contact_config = ContactConfig()
    contact_hash = get_config_hash(contact_config)
    graph_hash = get_config_hash(graph_config)
    contact_dir = pinder_data_cp / "foldseek_contacts" / contact_hash
    interface_pkl = contact_dir / "interfaces.pkl"
    graph_utils.construct_interface_graph(
        interface_pkl, output_dir=graph_dir, graph_config=graph_config
    )
    output_hash_fp = graph_dir / graph_hash
    assert (output_hash_fp / "min_length_interfaces.pkl").is_file()


@pytest.mark.parametrize(
    "alignment_type, alignment_file",
    [
        ("foldseek", "foldseek/foldseek_dbs/alignment.txt"),
        ("mmseqs", "foldseek/mmseqs2/alignment.txt"),
    ],
)
def test_construct_graph_from_alignment(pinder_data_cp, alignment_type, alignment_file):
    graph_dir = pinder_data_cp / "graphs"
    graph_config = GraphConfig()
    graph_hash = get_config_hash(graph_config)
    aln_path = pinder_data_cp / alignment_file
    graph_utils.construct_graph_from_alignment(
        alignment_file=aln_path,
        alignment_type=alignment_type,
        output_dir=graph_dir,
        graph_config=graph_config,
    )
    output_hash_fp = graph_dir / graph_hash
    assert (output_hash_fp / f"{alignment_type}_alignment_graph.pkl").is_file()


@pytest.mark.parametrize(
    "alignment_type, alignment_file",
    [
        ("foldseek", "foldseek/foldseek_dbs/alignment.txt"),
        ("mmseqs", "foldseek/mmseqs2/alignment.txt"),
    ],
)
def test_construct_interface_cleaned_graph(
    pinder_data_cp, alignment_type, alignment_file
):
    contact_config = ContactConfig()
    graph_config = GraphConfig()
    graph_dir = pinder_data_cp / "graphs"
    contact_hash = get_config_hash(contact_config)
    graph_hash = get_config_hash(graph_config)
    contact_dir = pinder_data_cp / "foldseek_contacts" / contact_hash
    interface_pkl = contact_dir / "interfaces.pkl"
    output_hash_fp = graph_dir / graph_hash
    interface_graph_pkl = output_hash_fp / "min_length_interfaces.pkl"
    graph_pkl = output_hash_fp / f"{alignment_type}_alignment_graph.pkl"
    aln_path = pinder_data_cp / alignment_file

    # First create interface graph pickle
    graph_utils.construct_interface_graph(
        interface_pkl, output_dir=graph_dir, graph_config=graph_config
    )
    # Next create alignment graph pickle
    alignment_graph = graph_utils.construct_graph_from_alignment(
        alignment_file=aln_path,
        alignment_type=alignment_type,
        output_dir=graph_dir,
        graph_config=graph_config,
    )
    # Now construct interface-cleaned alignment graph
    graph_utils.construct_interface_cleaned_graph(
        graph_pkl=alignment_graph,
        interface_pkl=interface_graph_pkl,
        alignment_type=alignment_type,
        graph_config=graph_config,
    )
    cleaned_pkl = graph_pkl.parent / f"cleaned_{graph_pkl.name}"
    assert cleaned_pkl.is_file()


@pytest.mark.parametrize(
    "interfaces, expect_edge",
    [
        # Test case for common interface, covered by alignment. Should result in edge
        (
            {
                ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
                    pdbid1="mono_A",
                    pdbid2="dumm_B",
                    indices1=set(range(1, 101)),
                    indices2=set(range(999, 1001)),
                ),
                ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
                    pdbid1="onom_B",
                    pdbid2="dumm_C",
                    indices1=set(range(150, 251)),
                    indices2=set(range(999, 1001)),
                ),
            },
            True,
        ),
        # Test case for distinct interfaces, both covered by alignment. Should result in edge
        (
            {
                ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
                    pdbid1="mono_A",
                    pdbid2="dumm_B",
                    indices1=set(range(1, 101)),
                    indices2=set(range(999, 1001)),
                ),
                ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
                    pdbid1="onom_B",
                    pdbid2="dumm_C",
                    indices1=set(range(400, 501)),
                    indices2=set(range(999, 1001)),
                ),
            },
            True,
        ),
        # Test case for distinct interfaces, first is covered by alignment. No edge
        (
            {
                ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
                    pdbid1="mono_A",
                    pdbid2="dumm_B",
                    indices1=set(range(1, 101)),
                    indices2=set(range(999, 1001)),
                ),
                ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
                    pdbid1="onom_B",
                    pdbid2="dumm_C",
                    indices1=set(range(1, 101)),
                    indices2=set(range(999, 1001)),
                ),
            },
            False,
        ),
        # Test case for distinct interfaces, second is covered by alignment. No edge
        (
            {
                ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
                    pdbid1="mono_A",
                    pdbid2="dumm_B",
                    indices1=set(range(400, 501)),
                    indices2=set(range(999, 1001)),
                ),
                ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
                    pdbid1="onom_B",
                    pdbid2="dumm_C",
                    indices1=set(range(400, 501)),
                    indices2=set(range(999, 1001)),
                ),
            },
            False,
        ),
        # Test case for distinct interfaces, neither is covered by alignment. No edge
        (
            {
                ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
                    pdbid1="mono_A",
                    pdbid2="dumm_B",
                    indices1=set(range(400, 501)),
                    indices2=set(range(999, 1001)),
                ),
                ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
                    pdbid1="onom_B",
                    pdbid2="dumm_C",
                    indices1=set(range(1, 101)),
                    indices2=set(range(999, 1001)),
                ),
            },
            False,
        ),
    ],
)
def test_get_interface_graph(interfaces, expect_edge):
    """
    Input alignment graph.
    Input interfaces.

    This function keeps edges for which the alignment covers BOTH interfaces.

    Interfaces can be the same: for the similar monomers, each interface is in
    the "same place".
    Or, interfaces can be distinct: for the similar monomers, the interface is not
    in the "same place".

    Separately, interfaces can be in the "region of alignment", or not.

    As written, this function doesn't care whether interfaces are common or distinct.
    As written, this function takes a logical OR over all interfaces.

    Test cases -> desired result:
        1. Common interface, alignment covers both interfaces -> edge
        X. Common interface, alignment covers exactly one interface -> impossible
        X. Common interface, alignment does not cover interface -> impossible,
            if we assume that common interface -> structural similarity
        2. Distinct interfaces, alignment covers both interfaces -> edge
        3. Distinct interfaces, alignment covers exactly one interface -> no edge
        4. Distinct interfaces, alignment covers neither interface -> no edge

    Test case description:
        Foldseek alignment: Two proteins A, B, each length 500 residues.
        A:1-350 is similar to B:150-500

        A:           |____________________----------|
        B: |----------____________________|

        "_" indicates structural similarity by foldseek

            1. Define interface: A:1-100, B:150-250 -> expect edge
            2. Define interface: A: 1-100, B: 400-500 -> expect edge
            3a: Define interface: A: 1-100, B: 1-100 -> expect no edge
            3b: Define interface: A: 400-500, B: 400-500 -> expect no edge
            4: Define interface: A:400-500, B: 1-100 -> expecet no edge
    """
    ## Update to use DiGraph and "alignment_idx"
    test_graph = nx.DiGraph()
    test_graph.add_node("mono_A")
    test_graph.add_node("onom_B")
    test_graph.add_edge(
        "mono_A",
        "onom_B",
        **{
            "alignment_idx": (1, 351),
            "score": 0.85,
        },
    )
    test_graph.add_edge(
        "onom_B",
        "mono_A",
        **{
            "alignment_idx": (150, 501),
            "score": 0.84,
        },
    )

    result_graph = graph_utils.get_interface_graph(test_graph, interfaces, coverage=1.0)
    assert (len(result_graph.edges()) == 1) is expect_edge


@pytest.mark.parametrize(
    "interfaces, expect_edge",
    [
        (
            {
                # common interface, covered by alignment. Should result in edge
                ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
                    pdbid1="mono_A",
                    pdbid2="dumm_B",
                    indices1=set(range(1, 101)),
                    indices2=set(range(999, 1001)),
                ),
                ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
                    pdbid1="onom_B",
                    pdbid2="dumm_C",
                    indices1=set(range(150, 251)),
                    indices2=set(range(999, 1001)),
                ),
                # another interface, now with a different partner
                ("mmud__E2_water", "dryy__R5_sheet"): MagicMock(
                    pdbid1="mmud_E",
                    pdbid2="dryy_R",
                    indices1=set(range(50, 101)),
                    indices2=set(range(999, 1001)),
                ),
            },
            True,
        ),
    ],
)
def test_get_interface_graph_multiple_interfaces(interfaces, expect_edge):
    """Make sure that multiple interfaces do not overwrite edges"""
    # Alignment graphs have score, interface graphs have weight
    test_graph = nx.DiGraph()
    test_graph.add_node("mono_A")
    test_graph.add_node("onom_B")
    test_graph.add_node("mmud_E")
    test_graph.add_edge(
        "mono_A",
        "onom_B",
        **{
            "alignment_idx": (1, 351),
            "score": 0.85,
        },
    )
    test_graph.add_edge(
        "onom_B",
        "mono_A",
        **{
            "alignment_idx": (150, 501),
            "score": 0.84,
        },
    )
    test_graph.add_edge(
        "mmud_E",
        "mono_A",
        **{
            "alignment_idx": (49, 200),
            "score": 0.89,
        },
    )
    test_graph.add_edge(
        "mono_A",
        "mmud_E",
        **{
            "alignment_idx": (1, 150),
            "score": 0.90,
        },
    )
    result_graph = graph_utils.get_interface_graph(test_graph, interfaces, coverage=1.0)
    assert (len(result_graph.edges()) == 2) is expect_edge

    assert result_graph.get_edge_data("mono_A", "mmud_E")["weight"] == pytest.approx(
        0.90
    )
    assert result_graph.get_edge_data("mono_A", "onom_B")["weight"] == pytest.approx(
        0.85
    )


@pytest.mark.parametrize(
    "indices, coverage, expect_edge",
    [
        # Test case for alignment that covers interface 1.0. Should result in edge.
        ([(1, 351), (150, 501)], 1.0, True),
        # Test case for common interface, covered by alignment at coverage 0.99.
        # No edge at req coverage 0.99.
        ([(2, 351), (150, 501)], 1.0, False),
        # Test case for common interface, covered by alignment at coverage 0.99.
        # No edge at req coverage 0.99.
        ([(1, 351), (151, 501)], 1.0, False),
        # Test case for common interface, covered by alignment at coverage 0.99.
        # Rescue edge at 0.95
        ([(2, 351), (150, 501)], 0.95, True),
        # Test case for common interface, covered by alignment at coverage 0.99.
        # Rescue edge at 0.95
        ([(1, 351), (151, 501)], 0.95, True),
    ],
)
def test_get_interface_graph_coverage(indices, coverage, expect_edge):
    ## Update to use DiGraph and "alignment_idx"
    interfaces = {
        ("mono__A1_foo", "dumm__B1_bar"): MagicMock(
            pdbid1="mono_A",
            pdbid2="dumm_B",
            indices1=set(range(1, 101)),
            indices2=set(range(999, 1001)),
        ),
        ("onom__B1_baz", "dumm__C1_bat"): MagicMock(
            pdbid1="onom_B",
            pdbid2="dumm_C",
            indices1=set(range(150, 251)),
            indices2=set(range(999, 1001)),
        ),
    }
    test_graph = nx.DiGraph()
    test_graph.add_node("mono_A")
    test_graph.add_node("onom_B")
    test_graph.add_edge(
        "mono_A",
        "onom_B",
        **{
            "alignment_idx": indices[0],
            "score": 0.85,
        },
    )
    test_graph.add_edge(
        "onom_B",
        "mono_A",
        **{
            "alignment_idx": indices[1],
            "score": 0.84,
        },
    )

    result_graph = graph_utils.get_interface_graph(
        test_graph, interfaces, coverage=coverage
    )
    assert (len(result_graph.edges()) == 1) is expect_edge


@pytest.fixture
def maxgraph():
    return graph_utils.MaxGraph()


def test_add_edge_with_different_weight(maxgraph):
    # Add edge with a higher weight
    maxgraph.add_edge("A", "B", weight=5)
    # Add the same edge with a lower weight
    maxgraph.add_edge("A", "B", weight=3)

    # Retrieve the weight of the edge
    edge_data = maxgraph.get_edge_data("A", "B")
    weight = edge_data["weight"]

    # Assert that the weight is the maximum value
    assert weight == 5


def test_add_edges_from_with_different_weight(maxgraph):
    # Add edges with a higher weight
    new_edges = [("A", "B", {"weight": 5}), ("B", "C", {"weight": 6})]
    # Add edges with a lower weight
    edges = [("A", "B", {"weight": 3}), ("B", "C", {"weight": 4})]

    # Add lower second
    maxgraph.add_edges_from(new_edges)
    maxgraph.add_edges_from(edges)

    # Retrieve the weight of the edges
    weight_AB = maxgraph.get_edge_data("A", "B")["weight"]
    weight_BC = maxgraph.get_edge_data("B", "C")["weight"]

    # Assert that the weights are the maximum values
    assert weight_AB == 5
    assert weight_BC == 6


@pytest.fixture
def digraph():
    return graph_utils.InterfaceGraph()


def test_merge_digraph_using_max(digraph):
    digraph.add_edge("A", "B", weight=5)
    digraph.add_edge("A", "B", weight=1)
    digraph.add_edge("B", "A", weight=3)
    digraph.add_edge("B", "A", weight=2)

    graph = digraph.to_undirected(reciprocal=True)
    weight_AB = graph.get_edge_data("A", "B")["weight"]

    assert weight_AB == 5


def test_subgraph_view_nofilter(maxgraph):
    # Add edges with a higher weight
    new_edges = [("A", "B", {"weight": 5}), ("B", "C", {"weight": 6})]
    # Add edges with a lower weight
    edges = [("A", "B", {"weight": 3}), ("B", "C", {"weight": 4})]

    # Add lower second
    maxgraph.add_edges_from(new_edges)
    maxgraph.add_edges_from(edges)

    view = nx.subgraph_view(maxgraph)

    # Retrieve the weight of the edges
    weight_AB = view.get_edge_data("A", "B")["weight"]
    weight_BC = view.get_edge_data("B", "C")["weight"]

    # Assert that the weights are the maximum values
    assert weight_AB == 5
    assert weight_BC == 6


def test_subgraph_view_filter_edge(maxgraph):
    # Add edges with a higher weight
    new_edges = [("A", "B", {"weight": 5}), ("B", "C", {"weight": 6})]
    # Add edges with a lower weight
    edges = [("A", "B", {"weight": 3}), ("B", "C", {"weight": 4})]

    # Add lower second
    maxgraph.add_edges_from(new_edges)
    maxgraph.add_edges_from(edges)

    view = nx.subgraph_view(
        maxgraph, filter_edge=lambda u, v: maxgraph.get_edge_data(u, v)["weight"] > 5
    )

    # Retrieve the weight of the edges
    weight_BC = view.get_edge_data("B", "C")["weight"]

    # Assert that the weights are the maximum values
    assert view.get_edge_data("A", "B") is None
    assert weight_BC == 6
