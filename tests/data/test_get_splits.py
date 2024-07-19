import pandas as pd
import pytest
from pinder.data import get_alignment_similarity, get_splits, get_test_set


def test_get_splits(splits_data_cp):
    pinder_dir = splits_data_cp
    get_splits.get_splits(pinder_dir)


def test_get_alignment_similarity(splits_data_cp):
    get_alignment_similarity.get_alignment_similarity(splits_data_cp)


def test_construct_final_index(splits_data_cp):
    get_splits.construct_final_index(splits_data_cp)


def test_curate_test_split(splits_data_cp):
    get_test_set.curate_test_split(splits_data_cp, parallel=False)


def test_assign_pinder_s_subset(splits_data_cp):
    index = pd.read_parquet(splits_data_cp / "index.parquet", engine="pyarrow")
    index = get_test_set.assign_pinder_s_subset(splits_data_cp, index)
    assert index.pinder_s.sum() > 0


def test_create_normalized_test_monomers(splits_data_cp):
    get_test_set.create_normalized_test_monomers(splits_data_cp, parallel=False)


def test_create_transformed_holo_monomer(splits_data_cp):
    pdb_dir = splits_data_cp / "pdbs"
    test_pdb_dir = splits_data_cp / "test_set_pdbs"
    pdb_file = next(pdb_dir.glob("*.pdb"))
    pdb_paths = (pdb_file, test_pdb_dir / pdb_file.name)
    get_test_set.create_transformed_holo_monomer(pdb_paths)


@pytest.mark.parametrize(
    "feat,n_samples",
    [
        ("resolution", 20),
        ("resolution", 50),
        (["polar_polar_contacts", "apolar_apolar_contacts"], 20),
    ],
)
def test_get_stratified_sample(feat, n_samples, splits_data_cp):
    df = pd.read_csv(splits_data_cp / "metadata.2.csv.gz")
    sampled = get_test_set.get_stratified_sample(
        df,
        feat=feat,
        n_samples=n_samples,
    )
    assert (
        sampled.shape[0] == n_samples,
        f"Expected {n_samples} samples, got {sampled.shape[0]}",
    )
    if not isinstance(feat, list):
        feat = [feat]
    df_mean = []
    for f in feat:
        sampled_min, sampled_max = sampled[f].min(), sampled[f].max()
        df_mean = df[f].mean()
        assert (
            sampled_min < df_mean < sampled_max,
            f"Expected column {f} samples to span min-max range containing overall mean!",
        )


def test_construct_sequence_database(splits_data_cp):
    get_test_set.construct_sequence_database(splits_data_cp, parallel=False)
    seq_db = splits_data_cp / "sequence_database.parquet"
    assert seq_db.is_file()
    db = pd.read_parquet(seq_db)
    assert db.shape == (30, 3)
