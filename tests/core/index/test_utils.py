import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from pinder.core.index.utils import (
    MetadataEntry,
    SupplementaryData,
    get_metadata,
    get_extra_metadata,
)

# Clear cache, because these functions get called on 'import pinder'
get_metadata.cache_clear()
get_extra_metadata.cache_clear()

extra_fields = [
    {},
    {
        "is_true": True,
    },
    {
        "K_d": 1.34e-6,
    },
    {
        "foo": "bar",
    },
    {
        "is_true": True,
        "foo": "bar",
    },
    {
        "is_true": True,
        "K_d": 1.34e-6,
    },
    {"is_true": True, "K_d": 1.34e-6, "foo": "bar"},
]


@pytest.fixture(scope="module")
def metadata_entry(pinder_data_copy, pinder_data_dir):
    metadata = get_metadata(pinder_data_copy / "pinder/metadata.1.csv.gz")
    return metadata.iloc[0, :]


@pytest.mark.parametrize("extra_fields", extra_fields)
def test_extra_metadata(metadata_entry, extra_fields):
    modified_entry = metadata_entry.to_dict() | extra_fields
    meta = MetadataEntry(**modified_entry)

    for key, value in extra_fields.items():
        assert getattr(meta, key) is value, f"{key} field is incorrect"


def test_read_extra_metadata(pinder_data_copy, pinder_data_dir):
    metadata = get_metadata("metadata.1.csv.gz")
    meta_list = metadata.apply(lambda row: MetadataEntry(**row.to_dict()), axis=1)
    meta_dict = dict(zip([m.id for m in meta_list], meta_list))

    assert meta_dict["7nsg__A1_P43005--7nsg__B1_P43005"].is_true is True
    assert meta_dict["7nsg__A1_P43005--7nsg__B1_P43005"].K_d < 1e-5
    assert meta_dict["7nsg__A1_P43005--7nsg__B1_P43005"].foo == "bar"
    assert meta_dict["7nsg__A1_P43005--7nsg__C1_P43005"].is_true is False
    assert meta_dict["7nsg__A1_P43005--7nsg__C1_P43005"].K_d == 50
    assert np.isnan(meta_dict["7nsg__A1_P43005--7nsg__C1_P43005"].foo)
    assert np.isnan(meta_dict["7nsg__B1_P43005--7nsg__C1_P43005"].is_true)
    assert np.isnan(meta_dict["7nsg__B1_P43005--7nsg__C1_P43005"].K_d)
    assert meta_dict["7nsg__B1_P43005--7nsg__C1_P43005"].foo == "baz"


def test_read_extra_metadata_from_bucket(
    pinder_data_copy, pinder_data_dir, monkeypatch
):
    def local_ls(self, bucket_path: str, *args, **kwargs):
        """Patch to let the Gsutil class just list the local dir"""
        return list(Path(bucket_path).iterdir())

    from pinder.core.utils.cloud import Gsutil as gs

    monkeypatch.setattr(gs, "ls", local_ls)

    extra_metadata = get_extra_metadata(
        "/this/path/dne",
        str(pinder_data_copy / "pinder"),
        glob_pattern="metadata-*.csv.gz",
        update=True,
    )

    assert all(
        [col in extra_metadata.columns for col in ["K_d", "foo", "is_true"]]
    ), "Did not find all metadata files"


def test_get_metadata_extra_data():
    meta = get_metadata(extra_data=SupplementaryData.paired_neff)
    assert "neff" in meta.columns
    modified_entry = meta.to_dict(orient="records")[0]
    meta = MetadataEntry(**modified_entry)
    assert isinstance(meta, MetadataEntry)
