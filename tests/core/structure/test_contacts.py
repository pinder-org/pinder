import pytest
from pinder.eval.clashes import count_clashes, count_pinder_clashes


def test_count_clashes(pinder_temp_dir, pinder_eval_dir, pinder_method_test_dir):
    id = "2e31__A1_Q80UW2--2e31__B1_P63208"
    pdb_file = pinder_method_test_dir / f"geodock/{id}/{id}.pdb"
    expected1 = {
        "atom_clashes": 0,
        "residue_clashes": 0,
        "min_dist": 2.738283256207756,
        "min_dist_vdw_ratio": 0.894863809218221,
        "vdw_sum": 3.0599999999999996,
        "radius": 1.2,
        "pdb_file": str(pdb_file),
    }
    clashes1 = count_clashes(pdb_file)
    for k, v in expected1.items():
        if isinstance(v, float):
            assert clashes1[k] == pytest.approx(v, abs=1e-6)
        else:
            assert clashes1[k] == v

    id2 = "1b8m__B1_P34130--1b8m__A1_P23560"
    pdb_name = "af__P23560-R--af__P34130-L.model_1.pdb"
    pdb_file = pinder_eval_dir / f"some_method/{id2}/predicted_decoys/{pdb_name}"
    expected2 = {
        "atom_clashes": 16,
        "residue_clashes": 9,
        "min_dist": 0.5731739049933875,
        "min_dist_vdw_ratio": 0.1873117336579698,
        "vdw_sum": 3.0599999999999996,
        "radius": 1.2,
        "pdb_file": str(pdb_file),
    }
    clashes2 = count_clashes(pdb_file)
    for k, v in expected2.items():
        if isinstance(v, float):
            assert clashes2[k] == pytest.approx(v, abs=1e-6)
        else:
            assert clashes2[k] == v


def test_count_pinder_clashes(pinder_temp_dir):
    id1 = "2e31__A1_Q80UW2--2e31__B1_P63208"
    id2 = "1b8m__B1_P34130--1b8m__A1_P23560"
    id1_clashes = count_pinder_clashes(id1)
    id2_clashes = count_pinder_clashes(id2)
    assert id1_clashes.shape == (5, 9)
    assert id2_clashes.shape == (5, 9)
    assert id1_clashes.atom_clashes.sum() == 23
    assert id2_clashes.atom_clashes.sum() == 36
