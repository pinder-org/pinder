import os
from pathlib import Path

import pytest
import pandas as pd
from pinder.data import get_dimers
from pinder.data.annotation import sabdab
from pinder.data.csv_utils import read_csv_non_default_na


requires_live_sabdab1 = pytest.mark.skip(
    reason="The legacy SAbDab1 summary endpoint now redirects to SAbDab2 HTML. "
    "These integration tests require the original SAbDab1 table; "
    "SAbDab2 annotations must not silently replace it."
)


@requires_live_sabdab1
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


@requires_live_sabdab1
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


@requires_live_sabdab1
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


def test_explode_sabdab1_chains():
    # Minimal legacy-format example; no live database is needed for expansion.
    summary = pd.DataFrame(
        [
            {
                "pdb_id": "8gag",
                "Hchain": "S",
                "Lchain": "s",
                "antigen_chain": "A | B | C",
                "antigen_het_name": "NA | NA | NA",
            }
        ]
    )

    chains = sabdab.explode_sabdab_per_chain(summary)

    assert list(chains.columns) == list(summary.columns)
    assert chains.shape[0] == 3
    assert set(chains.pdb_id) == {"8gag"}
    assert set(chains.Hchain) == {"S"}
    assert set(chains.Lchain) == {"s"}
    assert set(chains.antigen_chain) == {"A", "B", "C"}
    assert chains.antigen_het_name.isna().all()


def test_add_sabdab1_annotations_offline(tmp_path, monkeypatch):
    # Synthetic SAbDab1 records exercise annotation semantics independently of
    # the retired service; these are not a replacement for the published data.
    summary = tmp_path / "legacy_summary.tsv"
    pd.DataFrame(
        [{"pdb": "8gag", "Hchain": "S", "Lchain": "s", "antigen_chain": "A | B"}]
    ).to_csv(summary, sep="\t", index=False)
    pairs = [("S", "A"), ("S", "s"), ("A", "B"), ("X", "Y")]
    ids = [f"8gag__{receptor}1_U1--8gag__{ligand}1_U2" for receptor, ligand in pairs]
    pd.DataFrame({"id": ids, "pdb_id": "8gag"}).to_csv(
        tmp_path / "index.1.csv.gz", index=False
    )
    pd.DataFrame(
        [
            {
                "id": identifier,
                "asym_id_R": receptor,
                "asym_id_L": ligand,
                "pdb_strand_id_R": receptor,
                "pdb_strand_id_L": ligand,
            }
            for identifier, (receptor, ligand) in zip(ids, pairs)
        ]
    ).to_parquet(tmp_path / "chain_metadata.parquet", index=False)
    monkeypatch.setattr(sabdab, "download_sabdab", lambda **kwargs: summary)

    sabdab.add_sabdab_annotations(tmp_path, use_cache=False)

    annotated = pd.read_csv(tmp_path / "index.1.csv.gz").set_index("id").loc[ids]
    assert annotated.contains_antibody.tolist() == [True, True, False, False]
    assert annotated.contains_antigen.tolist() == [True, False, True, False]
    metadata = pd.read_parquet(tmp_path / "sabdab_metadata.parquet")
    heavy_ids = set(";".join(metadata.pinder_Hchain_ids).split(";")) - {""}
    assert heavy_ids == {ids[0], ids[1]}
    light_ids = set(";".join(metadata.pinder_Lchain_ids).split(";")) - {""}
    assert light_ids == {ids[1]}
    antigen_ids = set(";".join(metadata.pinder_antigen_chain_ids).split(";")) - {""}
    assert antigen_ids == {ids[0], ids[2]}

    def unexpected_download(**kwargs):
        raise AssertionError("Existing annotation cache should be reused")

    monkeypatch.setattr(sabdab, "download_sabdab", unexpected_download)
    sabdab.add_sabdab_annotations(tmp_path, use_cache=True)
