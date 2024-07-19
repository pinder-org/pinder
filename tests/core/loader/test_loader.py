import pytest
from pinder.core import get_systems, PinderLoader, PinderSystem
from pinder.core.loader import filters
from pinder.core.loader.writer import PinderDefaultWriter


def test_get_systems(pinder_temp_dir):
    pinder_id = "3s9d__B1_P48551--3s9d__A1_P01563"

    local_paths = {}
    for system in get_systems([pinder_id]):
        assert isinstance(system, PinderSystem)
        local_paths[system.entry.id] = system.filepaths
    assert list(local_paths.keys()) == [pinder_id]
    expected_keys = {
        "apo_ligand",
        "apo_receptor",
        "holo_ligand",
        "holo_receptor",
        "pred_ligand",
        "pred_receptor",
    }
    assert set(local_paths[pinder_id].keys()) == expected_keys
    for path in local_paths[pinder_id].values():
        assert path.is_file()


def test_load_split(pinder_temp_dir):
    loader = PinderLoader()
    loader.load_split(split="test", subset="pinder_af2")
    assert not isinstance(loader.dimers, list)
    count = 0
    for dimer in loader.dimers:
        assert dimer.entry.pinder_af2
        assert dimer.entry.split == "test"
        count += 1
        if count > 0:
            break


def test_load_split_invalid(pinder_temp_dir):
    loader = PinderLoader()
    with pytest.raises(ValueError) as exc_info:
        loader.load_split(split="foo")
    assert exc_info.type is ValueError
    assert exc_info.value.args[0] == "No systems found matching foo None"


def test_pinder_system_loader(pinder_temp_dir):
    base_filters = [
        filters.FilterByMissingHolo(),
        filters.FilterSubByContacts(min_contacts=5, radius=10.0, calpha_only=True),
        filters.FilterByHoloElongation(max_var_contribution=0.92),
        filters.FilterDetachedHolo(radius=12, max_components=2),
    ]
    sub_filters = [
        filters.FilterSubByAtomTypes(min_atom_types=4),
        filters.FilterByHoloOverlap(min_overlap=5),
        filters.FilterByHoloSeqIdentity(min_sequence_identity=0.8),
        filters.FilterSubLengths(min_length=0, max_length=1000),
        filters.FilterSubRmsds(rmsd_cutoff=7.5),
        filters.FilterByElongation(max_var_contribution=0.92),
        filters.FilterDetachedSub(radius=12, max_components=2),
    ]
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    loader = PinderLoader(base_filters=base_filters, sub_filters=sub_filters)
    loader.load_systems([pinder_id])
    assert not isinstance(loader.dimers, list)
    for dimer in loader.dimers:
        assert dimer.entry.id == pinder_id

    loader.load_systems([pinder_id])
    dimers = list(loader.dimers)
    assert len(dimers) == 1
    assert isinstance(dimers[0], PinderSystem)


def test_pinder_loader_load(pinder_temp_dir):
    base_filters = [
        filters.FilterByMissingHolo(),
        filters.FilterSubByContacts(min_contacts=5, radius=10.0, calpha_only=True),
        filters.FilterByHoloElongation(max_var_contribution=0.92),
        filters.FilterDetachedHolo(radius=12, max_components=2),
    ]
    sub_filters = [
        filters.FilterSubByAtomTypes(min_atom_types=4),
        filters.FilterByHoloOverlap(min_overlap=5),
        filters.FilterByHoloSeqIdentity(min_sequence_identity=0.8),
        filters.FilterSubLengths(min_length=0, max_length=1000),
        filters.FilterSubRmsds(rmsd_cutoff=7.5),
        filters.FilterByElongation(max_var_contribution=0.92),
        filters.FilterDetachedSub(radius=12, max_components=2),
    ]
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    loader = PinderLoader(base_filters=base_filters, sub_filters=sub_filters)
    loader.load_systems([pinder_id])
    loaded = loader.load(n_cpu=8, batch_size=1)
    for dimer in loaded:
        assert isinstance(dimer, PinderSystem)
        assert dimer.entry.id == pinder_id

    loader.load_systems([pinder_id])
    loaded = loader.load(n_cpu=8, batch_size=10)
    for batch in loaded:
        dimer_list = list(batch)
        assert isinstance(dimer_list[0], PinderSystem)
        assert dimer_list[0].entry.id == pinder_id

    loader.load_systems([pinder_id])
    loaded = loader.load(n_cpu=1, batch_size=10)
    for batch in loaded:
        dimer_list = list(batch)
        assert isinstance(dimer_list[0], PinderSystem)
        assert dimer_list[0].entry.id == pinder_id

    loader = PinderLoader(
        base_filters=base_filters,
        sub_filters=sub_filters,
        writer=PinderDefaultWriter(pinder_temp_dir),
    )
    loader.load_systems([pinder_id])
    loaded = loader.load(n_cpu=1, batch_size=1)
    expect_none = list(loaded)
    assert expect_none == [None]
    assert loader.writer.output_path.is_dir()
