import os
import subprocess
from tempfile import TemporaryDirectory
import pytest
from pathlib import Path
from pinder.core import get_index, get_metadata
from pinder.core.index.utils import (
    IndexEntry,
    MetadataEntry,
    get_index_location,
    get_pinder_bucket_root,
)


pindex = get_index()
metadata = get_metadata()


def test_get_pinder_bucket_root():
    expected = "gs://pinder/2024-02"
    assert get_pinder_bucket_root() == expected
    os.environ["PINDER_RELEASE"] = "new-release"
    expected = "gs://pinder/new-release"
    assert get_pinder_bucket_root() == expected
    os.environ["PINDER_RELEASE"] = "2024-02"


def test_get_index_location():
    index_loc = get_index_location()
    assert isinstance(index_loc, Path)
    assert index_loc.name == "index.parquet"

    remote_index_loc = get_index_location(remote=True)
    assert isinstance(remote_index_loc, str)
    assert remote_index_loc == "gs://pinder/2024-02/index.parquet"


def test_index_entry():
    entry_id = "1zop__A1_P20701--1zop__B1_P20701"
    row = pindex.query(f'id == "{entry_id}"')
    entry = IndexEntry(**row.to_dict(orient="records")[0])
    assert entry.homodimer
    assert entry.pinder_id == entry_id


def test_metadata_schema():
    print(f"metadata columns: {list(metadata.columns)}")
    for rec in metadata.sample(50).to_dict(orient="records"):
        meta = MetadataEntry(**rec)
        assert isinstance(meta, MetadataEntry)


def test_metadata_memory_use():
    mem_usage = metadata.memory_usage(deep=True).sum() / 1024**2
    assert (
        mem_usage < 1000.0
    ), f"Expected metadata memory use to be <1Gb, got {mem_usage}"


def test_index_memory_use():
    mem_usage = pindex.memory_usage(deep=True).sum() / 1024**2
    assert mem_usage < 1000.0, f"Expected index memory use to be <1Gb, got {mem_usage}"


def test_update_index():
    from subprocess import Popen, PIPE

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        os.environ["PINDER_BASE_DIR"] = str(tmp_dir_path)
        cmd = f"pinder_update_index --pinder_base_dir {tmp_dir_path}"
        proc = Popen(cmd, shell=True, stderr=PIPE, stdout=PIPE)
        stdout, stderr = proc.communicate()
        result = stdout.decode().strip().split("\n")
        if proc.returncode != 0:
            for ln in stderr.decode().splitlines():
                print(f"ERROR: {ln.strip()}")
            for ln in stdout.decode().splitlines():
                print(f"ERROR: {ln.strip()}")
        else:
            for ln in stdout.decode().splitlines():
                print(ln.strip())
        index_file = tmp_dir_path / "pinder/2024-02/index.parquet"
        assert (
            index_file.is_file(),
            f"Expected index file {index_file} not found. "
            f"Pinder dir contains {list(tmp_dir_path.glob('*/*/*'))}\n"
            f"Pinder release contains {list(tmp_dir_path.glob('*/*'))}\n"
            f"tmp_dir contains {list(tmp_dir_path.glob('*'))}\n",
        )
