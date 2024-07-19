import os
from pathlib import Path

import pytest
import pandas as pd
from pinder.data import get_dimers
from pinder.data.annotation import sabdab
from pinder.data.csv_utils import read_csv_non_default_na


@pytest.mark.parametrize(
    "use_cache",
    [
        # True,
        False,
    ],
)
def test_index_dimers(use_cache, pinder_data_cp):
    data_dir = pinder_data_cp / "nextgen_rcsb"
    pinder_dir = pinder_data_cp / "pinder"
    gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT", "vantai-analysis")
    get_dimers.index_dimers(
        data_dir,
        pinder_dir,
        google_cloud_project=gcp_project,
        use_cache=use_cache,
        parallel=False,
    )


def test_download_sabdab(pinder_data_cp):
    pinder_dir = pinder_data_cp / "pinder"
    sabdab_tsv = sabdab.download_sabdab(pinder_dir)
    assert sabdab_tsv.is_file()
    df = pd.read_csv(sabdab_tsv, sep="\t")
    expected_columns = [
        "pdb",
        "Hchain",
        "Lchain",
        "model",
        "antigen_chain",
        "antigen_type",
        "antigen_het_name",
        "antigen_name",
        "short_header",
        "date",
        "compound",
        "organism",
        "heavy_species",
        "light_species",
        "antigen_species",
        "authors",
        "resolution",
        "method",
        "r_free",
        "r_factor",
        "scfv",
        "engineered",
        "heavy_subclass",
        "light_subclass",
        "light_ctype",
        "affinity",
        "delta_g",
        "affinity_method",
        "temperature",
        "pmid",
    ]
    assert list(df.columns) == expected_columns
    df.rename({"pdb": "pdb_id"}, axis=1, inplace=True)
    long = sabdab.explode_sabdab_per_chain(df)
    assert long.shape[0] > df.shape[0]
    assert long.shape[1] == df.shape[1]
    assert "NA" not in set(long.Hchain)
    assert "NA" not in set(long.Lchain)
    assert "NA" not in set(long.antigen_chain)
    assert "NA" not in set(long.antigen_het_name)
    test_pdb = "8gag"
    pdb_chains = long.query('pdb_id == "8gag"')
    assert pdb_chains.shape[0] == 3
    assert set(pdb_chains.Hchain) == {"S"}
    assert set(pdb_chains.Lchain) == {"s"}
    assert set(pdb_chains.antigen_chain) == {"A", "B", "C"}


def test_add_sabdab_annotations(pinder_data_cp):
    pinder_dir = pinder_data_cp / "sabdab"
    start_index = read_csv_non_default_na(
        pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
    )
    start_cols = set(start_index.columns)
    assert (
        len({"contains_antigen", "contains_antibody"}.intersection(start_cols)) == 0
    ), "Index contains contains_antigen or contains_antibody before sabdab annotation!"
    sabdab.add_sabdab_annotations(pinder_dir, use_cache=False)
    assert (pinder_dir / "sabdab_metadata.parquet").is_file()
    index = read_csv_non_default_na(
        pinder_dir / "index.1.csv.gz", dtype={"pdb_id": "str"}
    )
    assert index.contains_antibody.sum() == 7721
    assert index.contains_antigen.sum() == 5426


def test_add_predicted_monomers_to_index(pinder_data_cp):
    pinder_dir = pinder_data_cp / "pinder"
    get_dimers.add_predicted_monomers_to_index(pinder_dir, use_cache=False)
    assert (pinder_dir / "index_with_pred.parquet").is_file()
    pred_index = pd.read_parquet(pinder_dir / "index_with_pred.parquet")
    no_pred = pred_index.query("~predicted_R and ~predicted_L")
    assert pred_index.query("predicted_R or predicted_L").shape[0] == 7
    assert set(no_pred.predicted_R_pdb) == {""}
    assert "" not in set(pred_index.query("predicted_R").predicted_R_pdb)
    assert set(no_pred.id) == {"7cma__A1_A0A2X0TC55--7cma__B2_A0A2X0TC55"}


def test_summarize_putative_apo_pred_counts(pinder_data_cp):
    pinder_dir = pinder_data_cp / "pinder"
    output_file = pinder_dir / "apo_pred_counts_by_uniprot.parquet"
    get_dimers.summarize_putative_apo_pred_counts(pinder_dir)
    assert output_file.is_file()
    counts = pd.read_parquet(output_file)
    assert counts.putative_apo_L_count.sum() == 2
    assert counts.putative_apo_R_count.sum() == 2
    assert counts.pred_R_count.sum() == 7
    assert counts.pred_L_count.sum() == 7
