from __future__ import annotations
from pathlib import Path
from typing import Iterator
from pinder.core.index.system import PinderSystem
from pinder.core.index.utils import (
    setup_logger,
    IndexEntry,
    MetadataEntry,
    get_index,
    get_metadata,
)
from pinder.core.loader import filters
import pandas as pd


log = setup_logger(__name__)


def get_dev_systems(
    dev_index: Path,
    dev_metadata: Path,
    dataset_path: Path | None = None,
) -> Iterator[PinderSystem]:
    """
    Loads a list of PinderSystem objects from a local (development) Pinder dataset.

    Use case example
    ----------------
    from pinder.core.loader import filters
    from pinder.data.system import get_dev_systems

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
    dimers = get_dev_systems(path_to_dev_pinder)

    for sub_filter in sub_filters:
        dimers = (dimer for dimer in dimers if sub_filter(dimer))

    for base_filter in base_filters:
        dimers = (dimer for dimer in dev_dimers if base_filter(dimer))

    """
    pindex_dev_df = get_index(dev_index)
    metadata_dev_df = get_metadata(dev_metadata)
    pindex_dev_df = pindex_dev_df.set_index("id", drop=False).sort_index()
    metadata_dev_df = metadata_dev_df.set_index("id", drop=False).sort_index()

    for id in pindex_dev_df.id:
        try:
            pindex_entry = IndexEntry(**pindex_dev_df.loc[id].to_dict())
            metadata_entry = MetadataEntry(**metadata_dev_df.loc[id].to_dict())
            dimer = PinderSystem(
                entry=pindex_entry, metadata=metadata_entry, dataset_path=dataset_path
            )
        except Exception as e:
            log.error(f"Error loading {id}: {e}")
            continue
        yield dimer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="""
        Example:
        PINDER_DATA_DIR=tests/test_data/pinder_data/pinder python -m pinder.data.system --index pindex.1.csv.gz --metadata metadata.1.csv.gz
    """
    )
    parser.add_argument("--index", type=Path, required=False, default="index.csv.gz")
    parser.add_argument(
        "--metadata", type=Path, required=False, default="metadata.csv.gz"
    )
    args = parser.parse_args()

    base_filters = [
        filters.FilterByMissingHolo(),
        filters.FilterSubByContacts(min_contacts=5, radius=10.0, calpha_only=True),
        filters.FilterByHoloElongation(max_var_contribution=0.92),
        filters.FilterDetachedHolo(radius=12, max_components=2),
    ]

    for dimer in get_dev_systems(args.index, args.metadata):
        if all(f(dimer) for f in base_filters):
            assert isinstance(dimer, PinderSystem)
