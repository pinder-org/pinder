import pytest
from tempfile import TemporaryDirectory
from pathlib import Path
from pinder.data.annotation import graphql


@pytest.fixture(scope="module")
def graphql_temp_dir():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


EXPECTED_FEATURE_TYPES = {
    "ASA",
    "UNASSIGNED_SEC_STRUCT",
    "CIS-PEPTIDE",
    "SCOP",
    "UNOBSERVED_RESIDUE_XYZ",
    "SCOP2B_SUPERFAMILY",
    "UNOBSERVED_ATOM_XYZ",
    "CATH",
    "SHEET",
    "HELIX_P",
    "ECOD",
    "BINDING_SITE",
}
EXPECTED_ANNOTATION_IDS = {
    "3.30.70.330",
    "8051194",
    "8061923",
    "e6q0rA3",
    "e6q0rA2",
    "8052530",
    "d6q0rd1",
    "e6q0rA1",
    "e6q0rD1",
    "d6q0rd2",
}


@pytest.mark.parametrize(
    "pdb_id",
    [
        "6Q0R",
    ],
)
def test_fetch_entry_annotations(pdb_id, tmp_path):
    data_json = tmp_path / f"{pdb_id}.json"
    graphql.fetch_entry_annotations(pdb_id, data_json, use_cache=False)
    assert data_json.is_file()
    pfam_df, feature_df, annotation_df, ec_df = graphql.parse_annotation_data(data_json)
    # Check for number of columns staying the same, there may be more annotation rows over time.
    assert pfam_df.shape[1] == 19
    assert feature_df.shape[1] == 14
    assert annotation_df.shape[1] == 12
    assert ec_df.shape == (0, 0)
    assert {"PF00076", "PF14939", "PF03178", "PF10172"}.issubset(set(pfam_df.rcsb_id))
    assert {"DCAF15_WD40", "RRM_1", "CPSF_A", "DDA1"}.issubset(
        set(pfam_df.rcsb_pfam_identifier)
    )
    assert set(feature_df.type) == EXPECTED_FEATURE_TYPES
    assert set(feature_df.asym_id) == {"C", "A", "E", "B", "D"}
    assert set(annotation_df.annotation_id) == EXPECTED_ANNOTATION_IDS


def test_populate_rcsb_annotations(graphql_temp_dir):
    pdb_ids = ["6Q0R", "2A79"]
    graphql.populate_rcsb_annotations(
        pinder_dir=graphql_temp_dir,
        pdb_ids=pdb_ids,
        parallel=True,
        max_workers=2,
    )
    annotation_fp = graphql_temp_dir / "rcsb_annotations"
    pfam_fp = annotation_fp / "pfam"
    feat_fp = annotation_fp / "features"
    annot_fp = annotation_fp / "annotations"
    assert all([d.is_dir() for d in [pfam_fp, feat_fp, annot_fp]])
    for pdb_id in pdb_ids:
        for fp in [pfam_fp, annot_fp, feat_fp]:
            expected_file = fp / f"{pdb_id}.csv.gz"
            assert expected_file.is_file(), f"{expected_file} not found"


def test_collect_rcsb_annotations(graphql_temp_dir):
    annotation_fp = graphql_temp_dir / "rcsb_annotations"
    graphql.collect_rcsb_annotations(graphql_temp_dir)
    annot_names = [
        "pfam",
        "annotations_ecod",
        "annotations_scop",
        "annotations_cath",
        "features_ecod",
        "features_scop",
        "features_cath",
    ]
    for name in annot_names:
        expected_file = annotation_fp / f"{name}.csv.gz"
        assert expected_file.is_file(), f"{expected_file} not found"
