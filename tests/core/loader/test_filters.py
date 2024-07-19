from pinder.core import PinderSystem

from pinder.core.loader import filters
from pinder.core.index.utils import MetadataEntry, SupplementaryData, get_metadata


def test_pinder_system_passes_filters(pinder_temp_dir):
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    dimer = PinderSystem(entry=pinder_id)
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
    dimers = [dimer]
    for sub_filter in sub_filters:
        dimers = [sub_filter(dimer) for dimer in dimers]

    for base_filter in base_filters:
        dimers = [dimer for dimer in dimers if base_filter(dimer)]

    assert len(dimers) == 1


def test_pinder_system_fails_filters(pinder_temp_dir):
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    dimer = PinderSystem(entry=pinder_id)
    base_filters = [
        filters.FilterByMissingHolo(),
        filters.FilterSubByContacts(min_contacts=1000, radius=10.0, calpha_only=True),
        filters.FilterByHoloElongation(max_var_contribution=0.92),
        filters.FilterDetachedHolo(radius=12, max_components=2),
    ]
    dimers = [dimer]
    for base_filter in base_filters:
        dimers = [dimer for dimer in dimers if base_filter(dimer)]

    assert len(dimers) == 0


def test_pinder_meta_filters(pinder_temp_dir):
    pinder_id = "1df0__A1_Q07009--1df0__B1_Q64537"
    ps = PinderSystem(entry=pinder_id)
    assert not ps.entry.contains_antibody
    failing_filter = filters.FilterMetadataFields(
        length2=(">=", 200),
        oligomeric_details=("==", "dimeric"),
        contains_antibody=("is", False),
    )
    assert failing_filter(ps) is False

    passing_filter = filters.FilterMetadataFields(
        oligomeric_details=("==", "dimeric"), contains_antibody=("is", False)
    )
    assert passing_filter(ps) is True

    list_of_meta_filters = [
        filters.FilterMetadataFields(contains_antibody=("", False)),
        filters.FilterMetadataFields(contains_enzyme=("is not", True)),
        filters.FilterMetadataFields(resolution=("<=", 2.75)),
        filters.FilterMetadataFields(method=("!=", "X-RAY DIFFRACTION")),
    ]
    expected_filter_results = [True, False, True, False]
    for i, meta_filter in enumerate(list_of_meta_filters):
        assert meta_filter(ps) == expected_filter_results[i]


def test_pinder_meta_filters_extrafields(pinder_data_copy, pinder_data_dir):
    metadata = get_metadata("metadata.1.csv.gz")
    meta_list = metadata.apply(lambda row: MetadataEntry(**row.to_dict()), axis=1)
    meta_dict = dict(zip([m.id for m in meta_list], meta_list))

    ids = [
        "7nsg__A1_P43005--7nsg__B1_P43005",
        "7nsg__A1_P43005--7nsg__C1_P43005",
        "7nsg__B1_P43005--7nsg__C1_P43005",
        "6wwe__B1_A0A287AZ37--6wwe__C1_L0N7N1",
    ]
    pss = [PinderSystem(entry=e, metadata=meta_dict[e]) for e in ids]

    meta_filters = [
        filters.FilterMetadataFields(is_true=("is", True)),
        filters.FilterMetadataFields(is_true=("is", False)),
        filters.FilterMetadataFields(foo=("is not", None)),
        filters.FilterMetadataFields(foo=("==", "bar")),
        filters.FilterMetadataFields(foo=("==", "baz")),
        filters.FilterMetadataFields(K_d=("is not", None)),
        filters.FilterMetadataFields(K_d=("<", 25)),
        filters.FilterMetadataFields(K_d=(">", 0.01)),
    ]

    expected_results = [
        [True, False, False, False],
        [False, True, False, False],
        [True, False, True, False],
        [True, False, False, False],
        [False, False, True, False],
        [True, True, False, False],
        [True, False, False, False],
        [False, True, False, False],
    ]

    for result_list, filter in zip(expected_results, meta_filters):
        for result, ps in zip(result_list, pss):
            assert (
                filter(ps) == result
            ), f"{ps.entry.id}: {filter.extra_meta_fields} did not pass"


def test_pinder_meta_filters_transient():
    metadata = get_metadata(extra_data=SupplementaryData.transient_interface_metadata)
    ids = [
        "3o7o__A1_Q9S5X8--3o7o__B1_Q9S5X8",
        "7x4b__A1_A0A2D0TCG3--7x4b__B1_A0A2D0TCG3",
    ]
    metadata = metadata[metadata["id"].isin(ids)].reset_index(drop=True)
    meta_list = metadata.apply(lambda row: MetadataEntry(**row.to_dict()), axis=1)
    meta_dict = dict(zip([m.id for m in meta_list], meta_list))

    pss = [PinderSystem(entry=e, metadata=meta_dict[e]) for e in ids]

    meta_filters = [
        filters.FilterMetadataFields(potential_transient=("is", False)),
    ]

    expected_results = [
        [False, False],
    ]

    for result_list, filter in zip(expected_results, meta_filters):
        for result, ps in zip(result_list, pss):
            assert (
                filter(ps) == result
            ), f"{ps.entry.id}: {filter.extra_meta_fields} did not pass"
