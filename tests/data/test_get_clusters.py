import string

import pytest
from unittest.mock import patch, MagicMock
import networkx as nx
import numpy as np

from pinder.data import graph_utils


def test_cluster(pinder_data_cp):
    import pickle as pkl
    import pandas as pd
    from pinder.data.get_clusters import cluster
    from pinder.data.config import (
        get_config_hash,
        ClusterConfig,
        ContactConfig,
        GraphConfig,
    )

    contact_config = ContactConfig()
    graph_config = GraphConfig()
    cluster_config = ClusterConfig()
    pinder_dir = pinder_data_cp / "pinder"
    graph_dir = pinder_dir / "graphs"
    contact_hash = get_config_hash(contact_config)
    graph_hash = get_config_hash(graph_config)
    contact_dir = pinder_data_cp / "foldseek_contacts" / contact_hash
    interface_pkl = contact_dir / "interfaces.pkl"
    output_hash_fp = graph_dir / graph_hash
    interface_graph_pkl = output_hash_fp / "min_length_interfaces.pkl"
    # First create interface graph pickle
    graph_utils.construct_interface_graph(
        interface_pkl, output_dir=graph_dir, graph_config=graph_config
    )

    alignment_pairs = [
        ("foldseek", "foldseek/foldseek_dbs/alignment.txt"),
        ("mmseqs", "foldseek/mmseqs2/alignment.txt"),
    ]
    for alignment_type, aln_suffix in alignment_pairs:
        alignment_file = pinder_data_cp / aln_suffix
        aln_path = pinder_data_cp / alignment_file
        graph_utils.construct_interface_alignment_graph(
            interface_pkl=interface_graph_pkl,
            alignment_file=aln_path,
            alignment_type=alignment_type,
            output_dir=graph_dir,
            graph_config=graph_config,
        )
    foldseek_cleaned_graph_pkl = output_hash_fp / "cleaned_foldseek_alignment_graph.pkl"
    mmseqs_cleaned_graph_pkl = output_hash_fp / "cleaned_mmseqs_alignment_graph.pkl"

    pindex_fname = pinder_dir / "index.1.csv.gz"
    checkpoint_dir = pinder_dir / "cluster" / get_config_hash(cluster_config)
    output_index_filename = "index.2.csv.gz"
    cluster(
        index=pd.read_csv(pindex_fname),
        foldseek_graph=pkl.load(open(foldseek_cleaned_graph_pkl, "rb")),
        mmseqs_graph=pkl.load(open(mmseqs_cleaned_graph_pkl, "rb")),
        interfaces_clean=pkl.load(open(interface_graph_pkl, "rb")),
        output_index_filename=output_index_filename,
        checkpoint_dir=checkpoint_dir,
    )
    foldseek_edge_comm_pkl = checkpoint_dir / "foldseek_communities.pkl"
    foldseek_edge_comp_pkl = checkpoint_dir / "foldseek_components.pkl"
    mmseqs_edge_comp_pkl = checkpoint_dir / "mmseqs_components.pkl"
    for edge_pkl in [
        foldseek_edge_comm_pkl,
        foldseek_edge_comp_pkl,
        mmseqs_edge_comp_pkl,
    ]:
        assert edge_pkl.is_file()

    output_index_file = checkpoint_dir / output_index_filename
    assert output_index_file.is_file()
