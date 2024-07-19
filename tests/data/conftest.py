import gzip
import shutil
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from pinder.data.alignment_utils import Interface, ContactConfig


@pytest.fixture
def pinder_data_cp(tmp_path, test_dir):
    src_dir = test_dir / "pinder_data"
    tmp_dir = tmp_path / "pinder_data"
    if tmp_dir.is_dir():
        shutil.rmtree(str(tmp_dir))

    data_dir = Path(shutil.copytree(src_dir, tmp_dir))
    return data_dir


def unzip_pdb(gz_file: Path):
    with gzip.open(gz_file, "rb") as f_in:
        with open(gz_file.parent / gz_file.stem, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    gz_file.unlink()


@pytest.fixture(scope="module")
def stateful_data_cp():
    test_asset_fp = Path(__file__).absolute().parent.parent / "test_data"
    src_dir = test_asset_fp / "pinder_data"
    with TemporaryDirectory() as tmp_dir:
        data_dir = Path(shutil.copytree(src_dir, Path(tmp_dir) / "pinder_data"))
        pdb_dir = data_dir / "apo_holo/pdbs"
        _ = [unzip_pdb(gz) for gz in pdb_dir.glob("*.pdb.gz")]
        yield data_dir


@pytest.fixture(scope="module")
def splits_data_cp():
    test_asset_fp = Path(__file__).absolute().parent.parent / "test_data"
    src_dir = test_asset_fp / "pinder_data/splits"
    with TemporaryDirectory() as tmp_dir:
        data_dir = Path(shutil.copytree(src_dir, Path(tmp_dir) / "splits"))
        pdb_dir = data_dir / "pdbs"
        _ = [unzip_pdb(gz) for gz in pdb_dir.glob("*.pdb.gz")]
        yield data_dir
