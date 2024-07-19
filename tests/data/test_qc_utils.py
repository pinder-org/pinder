import pandas as pd
from pinder.core import (
    get_index,
    get_pinder_bucket_root,
    get_supplementary_data,
)
from pinder.data.qc import utils as qu


def test_download_pdbfam_db(tmp_path):
    pfam_pqt = f"{get_pinder_bucket_root()}/PDBfam.parquet"
    pfam_data = qu.load_pfam_db(pfam_pqt)
    assert list(pfam_data.index.names) == ["PdbID", "AuthChain"]


def test_load_index():
    loaded_index = qu.load_index()
    assert loaded_index.shape == get_index().shape


def test_load_metadata():
    load_metadata = qu.load_metadata()
    assert "chain_1_residues" in load_metadata.columns
    assert "chain_2_residues" in load_metadata.columns


def test_load_entity_metadata():
    loaded_entities = qu.load_entity_metadata()
    assert loaded_entities.shape == get_supplementary_data("entity_metadata").shape


def test_view_potential_leaks(test_dir, tmp_path):
    ialign_dir = test_dir / "pinder_data/splits/ialign_metrics"
    pair_pqt = ialign_dir / "potential_alignment_leaks.parquet"
    potential_leak_pairs = pd.read_parquet(pair_pqt)
    potential_leak_pairs.rename({"id": "query_id"}, axis=1, inplace=True)
    pml_file = tmp_path / "test_view_potential_leaks.pml"
    qu.view_potential_leaks(
        potential_leak_pairs,
        pml_file=pml_file,
    )
    assert pml_file.is_file()
