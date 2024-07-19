import pandas as pd
import pytest
import shutil
from pinder.data.qc import ialign


query_id = "3bof__A1_Q9WYA5--3bof__B1_Q9WYA5"
hit_id = "3bol__A1_Q9WYA5--3bol__B1_Q9WYA5"

expected = {
    "query_id": query_id,
    "hit_id": hit_id,
    "IS-score": pytest.approx(0.96407),
    "P-value": pytest.approx(4.376e-35),
    "Z-score": pytest.approx(79.114),
    "Number of aligned residues": 124,
    "Number of aligned contacts": 567,
    "RMSD": pytest.approx(0.32),
    "Seq identity": pytest.approx(1.0),
}


@pytest.mark.skipif(
    shutil.which("ialign.pl") is None,
    reason="could not find ialign.pl, make sure it's installed and in $PATH!",
)
def test_ialign(test_dir):
    ialign_dir = test_dir / "pinder_data/ialign"
    query_pdb = ialign_dir / f"{query_id}.pdb"
    hit_pdb = ialign_dir / f"{hit_id}.pdb"
    results = ialign.ialign(query_id, query_pdb, hit_id, hit_pdb)
    for k, v in results.items():
        assert expected[k] == v, f"ialign key {k} expected value {expected[k]} != {v}"


@pytest.mark.skipif(
    shutil.which("ialign.pl") is None,
    reason="could not find ialign.pl, make sure it's installed and in $PATH!",
)
def test_ialign_all(test_dir):
    ialign_dir = test_dir / "pinder_data/ialign"
    df = pd.DataFrame([{"id": query_id, "hit_id": hit_id}])
    result_df = ialign.ialign_all(df, pdb_root=ialign_dir, n_jobs=1)
    assert result_df.shape[0] == 1
    results = result_df.to_dict(orient="records")[0]
    for k, v in results.items():
        assert expected[k] == v, f"ialign key {k} expected value {expected[k]} != {v}"
