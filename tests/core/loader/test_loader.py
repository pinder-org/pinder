import pytest
from pinder.core import get_systems, PinderLoader, PinderSystem
from pinder.core.loader import filters
from pinder.core.loader.writer import PinderDefaultWriter
from pinder.core.loader.structure import Structure


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
    loader = PinderLoader(split="test", subset="pinder_af2")
    count = 0
    for item in loader:
        assert isinstance(item, tuple)
        assert len(item) == 3
        dimer, feature_complex, target_complex = item
        assert dimer.entry.pinder_af2
        assert dimer.entry.split == "test"
        count += 1
        if count > 0:
            break


@pytest.mark.parametrize(
    "split, ids",
    [
        # invalid split is selected
        ("foo", None),
        # valid ids and split provided, but id is in a different split
        ("test", ["7mbm__I1_P84233--7mbm__J1_P62799"]),
    ],
)
def test_load_split_invalid(split, ids, pinder_temp_dir):
    with pytest.raises(ValueError) as exc_info:
        loader = PinderLoader(split=split, ids=ids)
    assert exc_info.type is ValueError
    assert (
        exc_info.value.args[0]
        == f"No systems found matching split={split}, ids={ids}, subset=None"
    )


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
    loader = PinderLoader(
        ids=[pinder_id],
        base_filters=base_filters,
        sub_filters=sub_filters,
    )
    assert len(loader) == 1
    dimer_iterator = iter(loader)
    dimer, feature_complex, target_complex = next(dimer_iterator)
    assert isinstance(dimer, PinderSystem)
    assert isinstance(feature_complex, Structure)
    assert isinstance(target_complex, Structure)
    for item in loader:
        assert isinstance(item, tuple)
        assert len(item) == 3
        assert item[0].entry.id == pinder_id

    items = list(loader)
    assert len(items) == 1
    assert isinstance(items[0][0], PinderSystem)


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
    loader = PinderLoader(
        ids=[pinder_id],
        base_filters=base_filters,
        sub_filters=sub_filters,
    )
    assert loader.index.shape[0] == 1
    system, feature, target = loader[0]
    assert isinstance(system, PinderSystem)
    assert isinstance(target, Structure)
    assert isinstance(feature, Structure)
    assert system.entry.id == pinder_id

    loader = PinderLoader(
        ids=[pinder_id],
        base_filters=base_filters,
        sub_filters=sub_filters,
        writer=PinderDefaultWriter(pinder_temp_dir),
    )
    item_dir = loader.writer.output_path / pinder_id
    assert loader.writer.output_path.is_dir()
    assert not item_dir.is_dir()
    _ = next(iter(loader))
    assert item_dir.is_dir()
    written_pdbs = list(item_dir.glob("*.pdb"))
    assert len(written_pdbs) == 4


@pytest.mark.parametrize(
    "monomer_priority, use_canonical_apo, expected_sample_id",
    [
        # canonical apo is selected
        ("apo", True, "1n6v__A1_P48551--2lms__A1_P01563"),
        # alternate apo is selected (system contains two receptor and two ligand apo monomers)
        ("apo", False, "1n6u__A1_P48551--1itf__A1_P01563"),
        # Predicted monomers are selected for both receptor and ligand
        ("pred", True, "af__P48551--af__P01563"),
    ],
)
def test_monomer_selection(
    monomer_priority, use_canonical_apo, expected_sample_id, pinder_temp_dir
):
    pinder_id = "3s9d__B1_P48551--3s9d__A1_P01563"
    loader = PinderLoader(
        ids=[pinder_id],
        monomer_priority=monomer_priority,
        use_canonical_apo=use_canonical_apo,
    )
    item = next(iter(loader))
    feature_complex = item[1]
    assert feature_complex.pinder_id == expected_sample_id


def test_load_pre_specified_monomers(pinder_temp_dir):
    import pandas as pd

    ids = [
        "1df0__A1_Q07009--1df0__B1_Q64537",
        "7mbm__I1_P84233--7mbm__J1_P62799",
        "3s9d__B1_P48551--3s9d__A1_P01563",
    ]
    pre_specified_df = pd.DataFrame([{"id": pid, "monomer": "pred"} for pid in ids])
    loader = PinderLoader(
        ids=ids,
        monomer_priority="holo",  # verify that this gets ignored
        pre_specified_monomers=pre_specified_df,
    )
    item = next(iter(loader))
    feature_complex = item[1]
    assert feature_complex.pinder_id.startswith("af__")
    pre_specified_dict = {pid: "pred" for pid in ids}
    loader = PinderLoader(
        ids=ids,
        monomer_priority="holo",  # verify that this gets ignored
        pre_specified_monomers=pre_specified_dict,
    )
    item = next(iter(loader))
    feature_complex = item[1]
    assert feature_complex.pinder_id.startswith("af__")


@pytest.mark.parametrize(
    "index_query, metadata_query, expected_len",
    [
        # full test split is loaded
        ("", "", 1955),
        # test split where both apo are defined
        ("(apo_R and apo_L)", "", 342),
        # No test systems have resolution > 3.5
        ("", "resolution <= 3.5", 1955),
        # pinder_xl is a boolean column that should match split = "test"
        ("pinder_xl", "resolution <= 3.5", 1955),
    ],
)
def test_load_with_index_and_metadata_query(
    index_query, metadata_query, expected_len, pinder_temp_dir
):
    loader = PinderLoader(
        split="test",
        index_query=index_query,
        metadata_query=metadata_query,
    )
    assert len(loader) == expected_len
