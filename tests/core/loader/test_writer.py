from pinder.core import PinderSystem
from pinder.core.loader.writer import PinderDefaultWriter, PinderClusteredWriter


def test_default_writer(pinder_temp_dir):
    pinder_id = "3s9d__B1_P48551--3s9d__A1_P01563"
    dimer = PinderSystem(pinder_id)
    output_path = pinder_temp_dir / "test_writer"
    writer = PinderDefaultWriter(
        output_path=output_path,
    )
    writer.write(dimer)
    assert output_path.is_dir()
    entry_path = output_path / pinder_id
    assert entry_path.is_dir()
    assert len(list(entry_path.glob("*.pdb"))) == 6


def test_clustered_writer(pinder_temp_dir):
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    dimer = PinderSystem(pinder_id)
    output_path = pinder_temp_dir / "test_writer"
    writer = PinderClusteredWriter(
        output_path=output_path,
    )
    writer.write(dimer)
    cluster_path = output_path / "cluster_1030_1030"
    expected_paths = [
        output_path,
        cluster_path,
        cluster_path / pinder_id,
        cluster_path / pinder_id / "apo",
        cluster_path / pinder_id / "holo",
        cluster_path / pinder_id / "predicted",
    ]
    for path in expected_paths:
        assert path.is_dir()
