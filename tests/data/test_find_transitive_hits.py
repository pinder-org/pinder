from datetime import date

import networkx as nx
import pandas as pd
import pytest
from unittest import mock
from unittest.mock import patch
import pickle as pkl
from itertools import count, product
from typing import Iterable
import time
import random

from pinder.data import graph_utils
from pinder.data.config import ClusterConfig
from pinder.data.get_clusters import load_cluster_cache
from pinder.data.find_transitive_hits import (
    get_leakage_dict,
    get_potential_representatives,
    get_test_conversion_dicts,
    get_transitive_hits,
    get_proto_splits_pindex,
    cluster_leaks,
    batch_cluster_leaks,
    map_systems_to_fsid_pairs,
    map_fsid_pair_to_systems,
    find_split_leakage,
    unordered_set_product,
    len_unordered_set_product,
    intersection_unordered_product,
)
from pinder.data.pipeline.constants import CONSIDER_LEAKED


random.seed(42)


class UniqueValueDict:
    def __init__(self):
        self.counter = count()

    def __getitem__(self, key):
        return next(self.counter)


class DegenValueDict:
    def __init__(self, elements: Iterable):
        self.elements = elements

    def __getitem__(self, key):
        return int(key in self.elements)


@pytest.fixture
def mock_metadata():
    data = {
        "oligomeric_count": [2, 2, 2, 2],
        "method": [
            "X-RAY DIFFRACTION",
            "X-RAY DIFFRACTION",
            "CRYO-EM",
            "X-RAY DIFFRACTION",
        ],
        "interface_atom_gaps_4A": [0, 0, 0, 1],
        "length1": [45, 100, 30, 45],
        "length2": [45, 100, 30, 45],
        "num_atom_types": [4, 10, 2, 3],
        "label": ["BIO", "BIO", "BIO", "NON-BIO"],
        "resolution": [2.0, 3.0, 3.6, 2.5],
        "number_of_components_1": [1, 1, 1, 1],
        "number_of_components_2": [1, 1, 1, 1],
        "max_var_1": [0.97, 0.91, 0.95, 0.90],
        "max_var_2": [0.97, 0.95, 0.90, 0.97],
        "date": ["2021-09-15", "2021-11-01", "2021-12-01", "2020-01-01"],
    }
    metadata = pd.DataFrame(data)
    return metadata


def test_get_potential_representatives(mock_metadata):
    config = ClusterConfig()
    test, af2mm_test = get_potential_representatives(mock_metadata, config)

    # Check the first DataFrame contains the correct potential representatives
    assert len(test) == 2, "The number of potential representatives should be 2."
    assert all(
        test["resolution"] <= config.resolution_thr
    ), "All potential representatives should have resolution below threshold."
    assert all(test["length1"] >= config.min_chain_length) and all(
        test["length2"] >= config.min_chain_length
    ), "All potential representatives should have chain lengths above minimum."
    assert all(test["max_var_1"] <= config.max_var_thr) and all(
        test["max_var_2"] <= config.max_var_thr
    ), "All potential representatives should have variance below threshold."

    # Check the second DataFrame contains representatives after a specific date
    assert (
        len(af2mm_test) == 1
    ), "There should be 1 representative after the specific date."
    assert all(
        af2mm_test["date"] > date.fromisoformat(config.alphafold_cutoff_date)
    ), "All representatives should be after the specific date."


@pytest.fixture
def sample_node_ids():
    return [
        "5Rx1_B",
        "9Jf8_K",
        "3Gu2_N",
        "7Fb9_A",
        "6Lp3_T",
        "2Ye4_Z",
        "8Dc7_R",
        "1Hw6_S",
        "4Mq9_J",
        "0Nx5_P",
        "2Ye4_Y",
        "8Dc7_T",
        "1Hw6_E",
        "4Mq9_T",
        "0Nx5_A",
    ]


@pytest.fixture
def sample_index_data():
    index = pd.DataFrame(
        {
            "id": [1, 2, 3],  # Yes, I know indices are complicated strings.
            "pdb_chain_chain": ["1ABC_A_B", "2XYZ_C_D", "3DEF_E_F"],
        }
    )
    return index


@pytest.fixture
def sample_test_index_data():
    # TODO cluster Ids are pairs
    test_index = pd.DataFrame({"id": [1, 2, 3, 4, 5], "cluster_id": [1, 1, 2, 2, 2]})
    return test_index


def test_get_test_conversion_dicts(sample_test_index_data):
    test_system_ids, cluster_to_test_systems = get_test_conversion_dicts(
        sample_test_index_data
    )

    assert test_system_ids == {1, 2, 3, 4, 5}
    assert cluster_to_test_systems == {1: {1, 2}, 2: {3, 4, 5}}


@pytest.fixture
def sample_data():
    pdb_chain_chain = ("1ABC_A", "1ABC_B")  # One dimer
    all_systems = {
        ("1ABC_A", "1ABC_B"),
        ("2XYZ_A", "2XYZ_B"),
        ("3DEF_A", "3DEF_B"),
        ("4GHI_A", "4GHI_B"),
        ("5JKL_A", "5JKL_B"),
        ("6MNO_A", "6MNO_B"),
        ("7PQR_A", "7PQR_B"),
        ("8STU_A", "8STU_B"),
        ("9VWX_A", "9VWX_B"),
        ("0YZA_A", "0YZA_B"),
    }  # Should be systems
    cluster_members = {
        ("1ABC_A", "1ABC_B"),
        ("2XYZ_A", "2XYZ_B"),
    }  # Should be systems in the same clusters as the pdb_chain_chain system

    graph1 = nx.Graph()
    graph2 = nx.Graph()
    # Add edges with weights
    edges_with_weights = [
        ("1ABC_A", "2XYZ_A", {"weight": 0.9}),  # Not a leak, same cluster
        ("1ABC_B", "2XYZ_B", {"weight": 0.9}),
        #
        ("2XYZ_A", "3DEF_A", {"weight": 0.8}),  # This is a leak (depth 1), test sort
        ("3DEF_B", "2XYZ_B", {"weight": 0.9}),
        #
        ("1ABC_A", "3DEF_A", {"weight": 0.8}),  # Leak (depth 1),
        ("1ABC_B", "3DEF_B", {"weight": 0.9}),
        #
        ("3DEF_B", "6MNO_B", {"weight": 0.9}),  # This is a leak(depth 2)
        ("3DEF_A", "6MNO_A", {"weight": 0.9}),  # This is a leak(depth 2)
        #
        (
            "3DEF_B",
            "0YZA_B",
            {"weight": 0.9},
        ),  # This is not a leak (2) because of no A chain edge
        #
        (
            "6MNO_A",
            "7PQR_A",
            {"weight": 0.6},
        ),  # This is a leak(depth 3), only if thresh <0.6
        ("6MNO_B", "7PQR_B", {"weight": 0.6}),  # This is a leak(depth 3)
    ]
    # Add all the edges to the graph
    graph1.add_edges_from(edges_with_weights)
    graph2.add_edges_from(edges_with_weights)
    graphs = [graph1, graph2]
    graphs = [graph1]
    thresholds = [0.5, 0.7]
    return pdb_chain_chain, all_systems, cluster_members, graphs, thresholds


@pytest.fixture()
def smallgraph():
    """A small, realistic graph that contains some cycles"""
    smallgraph = nx.Graph()
    edges_with_weights = [
        # Cluster 0
        ("1ABC_A", "NUL0_A", {"weight": 0.9}),
        ("1ABC_A", "NUL1_A", {"weight": 0.9}),
        ("NUL0_A", "NUL1_A", {"weight": 0.9}),
        # Cluster 1, fully connected
        ("1ABC_A", "9XYZ_A", {"weight": 0.9}),
        ("9XYZ_A", "NUL2_A", {"weight": 0.9}),
        ("9XYZ_A", "NUL3_A", {"weight": 0.9}),
        ("9XYZ_A", "NUL4_A", {"weight": 0.9}),
        ("9XYZ_A", "NUL5_A", {"weight": 0.9}),
        ("NUL2_A", "NUL3_A", {"weight": 0.9}),
        ("NUL2_A", "NUL4_A", {"weight": 0.9}),
        ("NUL2_A", "NUL5_A", {"weight": 0.9}),
        ("NUL3_A", "NUL4_A", {"weight": 0.9}),
        ("NUL3_A", "NUL5_A", {"weight": 0.9}),
        ("NUL4_A", "NUL5_A", {"weight": 0.9}),
        # Cluster 2
        ("1ABC_B", "NUL0_B", {"weight": 0.9}),
        ("1ABC_B", "NUL6_A", {"weight": 0.9}),
        ("NUL0_B", "NUL6_A", {"weight": 0.9}),
        # Link from cluster 2 to cluster 3
        ("NUL6_A", "9XYZ_B", {"weight": 0.9}),
        # Cluster 3
        ("9XYZ_B", "NUL8_A", {"weight": 0.9}),
        ("9XYZ_B", "NUL2_B", {"weight": 0.9}),
    ]
    smallgraph.add_edges_from(edges_with_weights)

    all_systems = {
        ("1ABC_A", "1ABC_B"),
        ("9XYZ_A", "9XYZ_B"),
        ("NUL0_A", "NUL0_B"),
        ("NUL1_A", "NUL1_B"),
        ("NUL2_A", "NUL2_B"),
        ("NUL3_A", "NUL3_B"),
        ("NUL4_A", "NUL4_B"),
        ("NUL5_A", "NUL5_B"),
        ("NUL6_A", "NUL6_B"),
        ("NUL7_A", "NUL7_B"),
        ("NUL8_A", "NUL8_B"),
    }

    system_of_interest = ("1ABC_A", "1ABC_B")

    return smallgraph, system_of_interest, all_systems


@pytest.fixture
def mock_graph():
    # Create a simple mock graph
    G = nx.Graph()
    G.add_edges_from(
        [
            ("PDB1_A", "PDB2_A"),
            ("PDB1_A", "PDB4_B"),
            ("PDB1_B", "PDB2_B"),
            ("PDB2_A", "PDB4_B"),
            ("PDB3_A", "PDB4_B"),
            ("PDB3_A", "PDB5_B"),
            ("PDB3_A", "PDB6_B"),
            ("PDB3_A", "PDB7_B"),
            ("PDB3_A", "PDB2_B"),
            ("PDB2_A", "PDB7_B"),
        ]
    )
    return G


@pytest.fixture
def sample_proto_index_data():
    return pd.DataFrame(
        {
            "id": ["A", "B", "C", "D"],
            "feature": [1, 2, 3, 4],
            "cluster_id_R": [5, 6, 7, 8],
            "cluster_id_L": [5, 6, -1, 8],
        }
    )


@pytest.fixture
def sample_metadata():
    return pd.DataFrame(
        {"id": ["A", "B", "C", "D"], "meta_feature": ["X", "Y", "Z", "O"]}
    )


@pytest.fixture
def mock_dependencies(monkeypatch):
    def inner():
        monkeypatch.setattr(
            "pinder.data.find_transitive_hits.get_potential_representatives",
            mock.MagicMock(return_value=(pd.DataFrame({"id": ["A", "D"]}), None)),
        )
        monkeypatch.setattr(
            "pinder.data.find_transitive_hits.get_test_conversion_dicts",
            mock.MagicMock(return_value=(None, {"cluster_id": ["A", "D"]})),
        )

    return inner


def test_get_proto_splits_pindex(
    sample_proto_index_data, sample_metadata, mock_dependencies
):
    # Set up the dependencies:
    mock_dependencies()

    # Run the function
    result_df, cluster_to_test_systems = get_proto_splits_pindex(
        sample_proto_index_data, sample_metadata
    )

    # Check if the 'split' column is added correctly
    expected_splits = ["proto-test", "proto-train", "proto-test"]
    assert (
        result_df["split"].tolist() == expected_splits
    ), "The split column values are not as expected."

    # Check the content of cluster_to_test_systems
    expected_cluster_to_test_systems = {"cluster_id": ["A", "D"]}
    assert (
        cluster_to_test_systems == expected_cluster_to_test_systems
    ), "The cluster_to_test_systems dictionary is not as expected."


@pytest.fixture()
def foldseek_graph_1e3(test_dir):
    graph_pkl = test_dir / "pinder_data" / "graphs" / "foldseek_graph_1e3nodes.pkl"
    with open(graph_pkl, "rb") as fi:
        return pkl.load(fi)


@pytest.fixture()
def filtered_pindex_leakage(test_dir):
    pindex_csv = test_dir / "pinder_data" / "graphs" / "test_pindex.csv.gz"
    return pd.read_csv(pindex_csv)


@pytest.fixture()
def filtered_metadata_leakage(test_dir):
    metadata_csv = test_dir / "pinder_data" / "graphs" / "test_metadata.csv.gz"
    return pd.read_csv(metadata_csv)


@pytest.fixture()
def foldseek_graph_clusters_1e3(test_dir):
    communities_cache = test_dir / "pinder_data" / "graphs" / "foldseek_communities.pkl"
    return load_cluster_cache(communities_cache)


@pytest.mark.parametrize(
    "node, node2clust, depth, expected",
    [
        ("6hwf_X", DegenValueDict({"6hwf_X"}), 0, {"6hwf_X"}),  # max length 2
        (
            "6hwf_X",
            DegenValueDict({"3nzx_R", "4nnw_P", "5mpb_O"}),
            2,
            {
                "4y80_R",
                "6qm7_U",
                "4nnw_P",
                "5l67_A",
                "1iru_I",
                "5mpb_O",
                "5l54_K",
                "5fgg_R",
                "4j70_A",
                "5cz7_J",
                "4qv3_L",
                "5tho_V",
                "4y8o_G",
                "5lai_D",
                "5l5b_L",
                "3mi0_H",
                "4r00_D",
                "6hvx_V",
                "4qw1_BA",
                "6hwf_X",
                "3wxr_E",
                "5lf7_S",
                "7w3f_LA",
                "7v5g_T",
                "3nzx_R",
                "4r67_I",
            },
        ),
        (
            "6hwf_X",
            DegenValueDict(
                {
                    "4y8o_G",
                    "5lai_D",
                    "4y80_R",
                    "4qv3_L",
                    "7w3f_LA",
                    "5l5b_L",
                }
            ),
            3,
            {
                "4qv3_L",
                "4qw1_BA",
                "4y8o_G",
                "4r67_I",
                "1iru_I",
                "6hvx_V",
                "5l5b_L",
                "4nnw_P",
                "3nzx_R",
                "6hwf_X",
                "7v5g_T",
                "5lai_D",
                "3wxr_E",
                "5fgg_R",
                "5mpb_O",
                "7w3f_LA",
                "4r00_D",
                "4y80_R",
                "5cz7_J",
                "6qm7_U",
                "4j70_A",
                "5l67_A",
                "5tho_V",
                "3mi0_H",
                "5lf7_S",
                "5l54_K",
            },
        ),
        (
            "5eu5_A",
            {"5eu5_A": 0, "2mha_B": 0, "5trz_D": 1, "3sge_A": 0, "6fgb_A": 100},
            3,
            {"6fgb_A", "2mha_B", "3sge_A", "5eu5_A", "5trz_D"},
        ),  # max length 7
        ("7t9t_J", {"7t9t_J": 0, "7t9a_B": 1}, 3, {"7t9t_J", "7t9a_B"}),  # max length 2
        ("7t9t_J", {"7t9t_J": 0, "7t9a_B": 1}, 2, {"7t9t_J", "7t9a_B"}),  # max length 2
    ],
)
def test_cluster_leaks(node, node2clust, depth, expected, foldseek_graph_1e3):
    result = cluster_leaks(node, foldseek_graph_1e3, node2clust, depth)

    print(result)
    assert len(result) == len(expected)
    for n in result:
        assert n in expected


@pytest.mark.parametrize(
    "node, depth, expected",
    [
        ("6hwf_X", 0, {"6hwf_X"}),  # max length 2
        (
            "6hwf_X",
            2,
            {
                "3nzx_R",
                "4nnw_P",
                "5mpb_O",
                "3wxr_E",
                "5l54_K",
                "7v5g_T",
                "5fgg_R",
                "3mi0_H",
                "5tho_V",
                "4j70_A",
                "1iru_I",
                "6hvx_V",
                "4r00_D",
                "6hwf_X",
                "6qm7_U",
                "5lf7_S",
                "4qw1_BA",
                "4r67_I",
                "4y8o_G",
                "5lai_D",
                "4y80_R",
                "4qv3_L",
                "7w3f_LA",
                "5l5b_L",
                "5l67_A",
                "5cz7_J",
            },
        ),  # max length 2
        (
            "6hwf_X",
            3,
            {
                "3nzx_R",
                "4nnw_P",
                "5mpb_O",
                "3wxr_E",
                "5l54_K",
                "7v5g_T",
                "5fgg_R",
                "3mi0_H",
                "5tho_V",
                "4j70_A",
                "1iru_I",
                "6hvx_V",
                "4r00_D",
                "6hwf_X",
                "6qm7_U",
                "5lf7_S",
                "4qw1_BA",
                "4r67_I",
                "4y8o_G",
                "5lai_D",
                "4y80_R",
                "4qv3_L",
                "7w3f_LA",
                "5l5b_L",
                "5l67_A",
                "5cz7_J",
            },
        ),  # max length 2
        (
            "5eu5_A",
            3,
            {"2mha_B", "5trz_D", "3sge_A", "6fgb_A", "5eu5_A"},
        ),  # max length 7
        ("7t9t_J", 3, {"7t9a_B", "7t9t_J"}),  # max length 2
        ("7t9t_J", 2, {"7t9a_B", "7t9t_J"}),  # max length 2
    ],
)
def test_cluster_leaks_unique_clusters(node, depth, expected, foldseek_graph_1e3):
    result = cluster_leaks(node, foldseek_graph_1e3, UniqueValueDict(), depth)

    print(result)
    assert len(result) == len(expected)
    for n in result:
        assert n in expected


@pytest.mark.parametrize(
    "nodeset, depth, max_node_degree, expected",
    [
        ({"6hwf_X"}, 0, 1000, {"6hwf_X": {"6hwf_X"}}),
        (
            {"6hwf_X"},
            0,
            0,
            {"6hwf_X": CONSIDER_LEAKED},
        ),  # Considers all nodes as leaked
        (
            {"5eu5_A", "7t9t_J"},
            3,
            2,
            {
                "5eu5_A": {"2mha_B", "5trz_D", "3sge_A", "6fgb_A", "5eu5_A"},
                "7t9t_J": {"7t9a_B", "7t9t_J"},
            },
        ),
        (
            {"5eu5_A", "7t9t_J"},
            3,
            1,
            {
                "5eu5_A": CONSIDER_LEAKED,
                "7t9t_J": {"7t9a_B", "7t9t_J"},
            },
        ),
    ],
)
def test_batch_cluster_leaks_unique_clusters(
    nodeset,
    depth,
    max_node_degree,
    expected,
    foldseek_graph_1e3,
):
    print(foldseek_graph_1e3.degree(nodeset))
    result = batch_cluster_leaks(
        nodeset,
        foldseek_graph_1e3,
        UniqueValueDict(),
        depth=depth,
        max_node_degree=max_node_degree,
    )

    print(result)
    for k, v in result.items():
        if expected[k] is None:
            assert v is None
        else:
            for n in v:
                assert n in expected[k]


def test_find_split_leakage(
    filtered_pindex_leakage,
    filtered_metadata_leakage,
    foldseek_graph_1e3,
    foldseek_graph_clusters_1e3,
):
    filtered_pindex, cluster_to_test_systems = get_proto_splits_pindex(
        filtered_pindex_leakage, filtered_metadata_leakage
    )
    filtered_pindex.loc[:, "split"] = [
        "proto-test" if i < 20 else "proto-train"
        for i in range(filtered_pindex.shape[0])
    ]
    systems_proto_test = set(
        filtered_pindex[filtered_pindex["split"] == "proto-test"]["id"].values.tolist()
    )
    systems_proto_train = set(
        filtered_pindex[filtered_pindex["split"] == "proto-train"]["id"].values.tolist()
    )

    score = 0.6
    n2c = graph_utils.get_node_to_cluster_mapping(foldseek_graph_clusters_1e3)
    split_leakage = find_split_leakage(
        systems_proto_test,
        systems_proto_train,
        foldseek_graph_1e3,
        n2c,
        depth=2,
        edge_threshold=score,
        max_node_degree=1000,
    )
    assert isinstance(split_leakage, dict)
    assert len(split_leakage) == 20


def test_map_systems_to_fsid_pairs():
    input = {
        "7zsw__A1_G1UBD5--7zsw__B1_G1UBD5": ("7zsw_A", "7zsw_B"),
        "1ar6__B50_P03300--1ar6__D50_P03300": ("1ar6_B", "1ar6_D"),
        "7zdm__JA1_A0A6P3E975--7zdm__SA1_A0A6P9FRJ5": ("7zdm_JA", "7zdm_SA"),
        "7opx__C30_P32537--7opx__C31_P32537": (
            "7opx_C",
            "7opx_C",
        ),  # two diff behavior with frozenset
        "6z8f__A30_Q50LE5--6z8f__A4_Q50LE5": (
            "6z8f_A",
            "6z8f_A",
        ),  # two diff behavior with frozenset
        "3h6i__AA1_P9WHU1--3h6i__BA1_P9WHT9": ("3h6i_AA", "3h6i_BA"),
        "8btd__C1_A8B7H8--8btd__O1_A8B8Z6": ("8btd_C", "8btd_O"),
        "4v5z__I1_UNDEFINED--4v5z__P1_UNDEFINED": ("4v5z_I", "4v5z_P"),
        "5zbo__CB1_G0Y2B2--5zbo__PA1_G0Y2B2": ("5zbo_CB", "5zbo_PA"),
        "7tjd__I3_Q8LTE1--7tjd__N1_Q8LTE1": ("7tjd_I", "7tjd_N"),
    }

    result = map_systems_to_fsid_pairs(set(input.keys()))

    for k in result.keys():
        assert result[k] == input[k]


def test_map_fsid_pair_to_systems():
    input = {
        "7zsw__A1_G1UBD5--7zsw__B1_G1UBD5": frozenset(("7zsw_A", "7zsw_B")),
        "1ar6__B50_P03300--1ar6__D50_P03300": frozenset(("1ar6_B", "1ar6_D")),
        "7zdm__JA1_A0A6P3E975--7zdm__SA1_A0A6P9FRJ5": frozenset(("7zdm_JA", "7zdm_SA")),
        "7opx__C30_P32537--7opx__C31_P32537": frozenset(
            ("7opx_C", "7opx_C")
        ),  # two diff behavior with frozenset
        "6z8f__A30_Q50LE5--6z8f__A4_Q50LE5": frozenset(
            ("6z8f_A", "6z8f_A")
        ),  # two diff behavior with frozenset
        "3h6i__AA1_P9WHU1--3h6i__BA1_P9WHT9": frozenset(("3h6i_AA", "3h6i_BA")),
        "8btd__C1_A8B7H8--8btd__O1_A8B8Z6": frozenset(("8btd_C", "8btd_O")),
        "4v5z__I1_UNDEFINED--4v5z__P1_UNDEFINED": frozenset(("4v5z_I", "4v5z_P")),
        "5zbo__CB1_G0Y2B2--5zbo__PA1_G0Y2B2": frozenset(("5zbo_CB", "5zbo_PA")),
        "7tjd__I3_Q8LTE1--7tjd__N1_Q8LTE1": frozenset(("7tjd_I", "7tjd_N")),
        "7tjd__I5_Q8LTE1--7tjd__N42_Q8LTE1": frozenset(("7tjd_I", "7tjd_N")),
    }  # multiple

    result = map_fsid_pair_to_systems(set(input.keys()))
    print(result)

    assert len(result.keys()) == 10

    for k, v in input.items():
        assert k in result[v]

    assert (
        frozenset(
            [
                "6z8f_A",
            ]
        )
        in result.keys()
    )
    assert (
        frozenset(
            {
                "7opx_C",
            }
        )
        in result.keys()
    )
    assert frozenset(("7opx_C",)) in result.keys()


@pytest.mark.parametrize(
    "A, B",
    [
        (set(range(10)), set(range(10))),
        (set(range(10)), set(range(5, 15))),
        (set(range(10)), set(range(11, 21))),
        (set(range(10)), set(range(5, 51))),
    ],
)
def test_unordered_set_product(A, B):
    result = unordered_set_product(A, B)

    ideal_result = {frozenset(e) for e in product(A, B)}
    assert len(list(result)) == len(list(ideal_result))

    result_frozen = {frozenset(e) for e in result}
    for e in result_frozen:
        assert e in ideal_result


@pytest.mark.parametrize(
    "A, B",
    [
        (set(range(10)), set(range(10))),
        (set(range(10)), set(range(5, 15))),
        (set(range(10)), set(range(11, 21))),
        (set(range(10)), set(range(5, 51))),
    ],
)
def test_len_unordered_set_product(A, B):
    result = len_unordered_set_product(A, B)

    ideal_result = {frozenset(e) for e in product(A, B)}
    assert result == len(list(ideal_result))


@pytest.mark.parametrize(
    "A, B, C",
    [
        (
            {"ABCD_A1", "WXYZ_A1", "WXYZ_B1", "QRST_A1", "ABCD_B1", "EFGH_B1"},
            {"ABCD_C1", "ABCD_B1", "EFGH_A1", "LMNO_B1"},
            {
                frozenset(["ABCD_A1", "ABCD_B1"]),
                frozenset(["EFGH_A1", "EFGH_B1"]),
                frozenset(["LMNO_A1", "LMNO_B1"]),
                frozenset(["WXYZ_A1", "WXYZ_B1"]),
                frozenset(["QRST_A1", "QRST_B1"]),
            },
        ),
    ],
)
def test_intersection_unordered_product(A, B, C):
    result = intersection_unordered_product(A, B, C)

    ideal_result = C.intersection({frozenset(e) for e in product(A, B)})

    assert len(list(result)) == len(list(ideal_result))

    for e in result:
        assert e in ideal_result


@pytest.mark.parametrize(
    "A, B, C",
    [
        # We want to loop through C here
        (
            set(range(1000)),
            set(range(1000)),
            {frozenset((1, 2)), frozenset((998, 999))},
        ),
        # We want to loop through the product here
        (
            {1, 2},
            {2, 3},
            {frozenset(e) for e in product(set(range(3000)), set(range(3000)))},
        ),
    ],
)
def test_intersection_unordered_product_speed(A, B, C):
    # Run IUP
    start_time = time.time()
    iup_result = intersection_unordered_product(A, B, C)
    iup_time = time.time() - start_time
    # Run naively iterating through product
    start_time = time.time()
    iter_prod_result = {frozenset(e) for e in product(A, B) if frozenset(e) in C}
    iter_prod_time = time.time() - start_time
    # Run naively iterating through C
    start_time = time.time()
    iter_C_result = {
        frozenset(e)
        for e in C
        if (tuple(e)[0] in A and tuple(e)[-1] in B)
        or (tuple(e)[-1] in A and tuple(e)[0] in B)
    }
    iter_C_time = time.time() - start_time

    assert iup_time < 1e-04
    assert iter_prod_time > 1e-04 or iter_C_time > 1e-04

    assert len(iup_result) == len(iter_prod_result) and len(iup_result) == len(
        iter_C_result
    )

    for e in iup_result:
        assert e in iter_prod_result
        assert e in iter_C_result


@pytest.mark.parametrize(
    "A, B, C",
    [
        (
            set(range(10)),
            set(range(10)),
            {frozenset(e) for e in product(set(range(10)), set(range(10)))},
        ),
        (
            set(range(10)),
            set(range(5, 15)),
            {frozenset(e) for e in product(set(range(10)), set(range(5, 15)))},
        ),
        (
            set(range(10)),
            set(range(11, 21)),
            {frozenset(e) for e in product(set(range(10)), set(range(11, 21)))},
        ),
        (
            set(range(10)),
            set(range(5, 51)),
            {frozenset(e) for e in product(set(range(10)), set(range(5, 51)))},
        ),
        # Now with C as a subset of the product
        (
            set(range(10)),
            set(range(10)),
            {
                frozenset(e)
                for e in random.choices(
                    list(product(set(range(10)), set(range(10)))),
                    k=50,
                )
            },
        ),
        (
            set(range(10)),
            set(range(5, 15)),
            {
                frozenset(e)
                for e in random.choices(
                    list(product(set(range(10)), set(range(5, 15)))), k=50
                )
            },
        ),
        (
            set(range(10)),
            set(range(11, 21)),
            {
                frozenset(e)
                for e in random.choices(
                    list(product(set(range(10)), set(range(11, 21)))), k=50
                )
            },
        ),
        (
            set(range(10)),
            set(range(5, 51)),
            {
                frozenset(e)
                for e in random.choices(
                    list(product(set(range(10)), set(range(5, 51)))), k=50
                )
            },
        ),
        # Now with C as a superset of the product
        (
            set(range(10)),
            set(range(10)),
            {frozenset(e) for e in product(set(range(20)), set(range(20)))},
        ),
        (
            set(range(10)),
            set(range(5, 15)),
            {frozenset(e) for e in product(set(range(20)), set(range(5, 25)))},
        ),
        (
            set(range(10)),
            set(range(11, 21)),
            {frozenset(e) for e in product(set(range(20)), set(range(11, 31)))},
        ),
        (
            set(range(10)),
            set(range(5, 51)),
            {frozenset(e) for e in product(set(range(20)), set(range(5, 101)))},
        ),
        # particular test where the result should be empty
        (set(range(10)), set(range(10)), {frozenset((1, 1000))}),
    ],
)
def test_intersection_unordered_product_correctness(A, B, C):
    result = intersection_unordered_product(A, B, C)

    ideal_result = C.intersection({frozenset(e) for e in product(A, B)})

    assert len(list(result)) == len(list(ideal_result))
    for e in result:
        assert e in ideal_result


def test_get_leakage_dict(splits_data_cp):
    get_leakage_dict(splits_data_cp, "foldseek")


def test_get_transitive_hits(splits_data_cp):
    get_transitive_hits(splits_data_cp, use_cache=False)


@pytest.fixture
def split_leakage_test_data():
    return {
        "pure_split": {
            "HIJK__A1_ABC123--HIJK__B1_ABC456",
            "KILL__A1_BILL123--KILL__B1_BILL456",
            "ABCD__A1_BILL012--ABCD__B1_BILL345",
            "WXYZ__A1_B0L2012--WXYZ__B1_B0L345",
            "JACK__A1_QBLACK2--JACK__B1_BLACK2Q",
        },
        "corrupt_split": {
            "DILL__A1_PICKLE--DILL__B1_PICKLE",
            "DILL__A2_PICKLE--DILL__B2_PICKLE",
            "QABC__A1_ILL012--QABC__B1_ILL345",
        },
        "potential_leaks": {
            "ABCD_A": {CONSIDER_LEAKED},
            "ABCD_B": {"WXYZ_A1"},
            "WXYZ_A": {"LMNO_A", "NEMO_A", "MEMO_A", "EMOO_A"},
            "WXYZ_B": {"MEMO_B"},
            "HIJK_A": set(),
            "HIJK_B": set(),
            "KILL_A": {"QABC_A"},
            "KILL_B": {"QABC_B"},
            "JACK_A": {"DILL_B"},
            "JACK_B": {"DILL_A"},
        },
    }


def test_mock_find_split_leakage(split_leakage_test_data):
    with patch(
        "pinder.data.find_transitive_hits.batch_cluster_leaks",
        return_value=split_leakage_test_data["potential_leaks"],
    ) as mocked_func:
        result = find_split_leakage(
            pure_split=split_leakage_test_data["pure_split"],
            corrupt_split=split_leakage_test_data["corrupt_split"],
            graph=nx.Graph(),  # not used, batch_cluster_leaks is tested separately
            node_to_cluster={
                "test": 1
            },  # not used, batch_cluster_leaks is tested separately
            depth=2,  # not used, batch_cluster_leaks is tested separately
            edge_threshold=0.8,  # not used, is tested separately
            max_node_degree=3,
        )
        assert result == {
            "ABCD__A1_BILL012--ABCD__B1_BILL345": {CONSIDER_LEAKED},
            "WXYZ__A1_B0L2012--WXYZ__B1_B0L345": set(),
            "JACK__A1_QBLACK2--JACK__B1_BLACK2Q": {
                "DILL__A1_PICKLE--DILL__B1_PICKLE",
                "DILL__A2_PICKLE--DILL__B2_PICKLE",
            },
            "HIJK__A1_ABC123--HIJK__B1_ABC456": set(),
            "KILL__A1_BILL123--KILL__B1_BILL456": {"QABC__A1_ILL012--QABC__B1_ILL345"},
        }
