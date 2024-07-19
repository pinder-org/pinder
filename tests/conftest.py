import json
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

test_asset_fp = Path(__file__).absolute().parent / "test_data"


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getoption("--slow"):
        pytest.skip("need --slow option to run this test")


@pytest.fixture
def test_dir():
    return test_asset_fp


@pytest.fixture(scope="session")
def pdb_5cq2():
    return test_asset_fp / "5cq2__A2_Q96J02--5cq2__C1_Q9H3M7.pdb"


@pytest.fixture(scope="session")
def example_atoms():
    from pinder.core.structure import atoms

    pdb_file = test_asset_fp / "5cq2__A2_Q96J02--5cq2__C1_Q9H3M7.pdb"
    arr = atoms.atom_array_from_pdb_file(pdb_file)
    return arr


@pytest.fixture(scope="session")
def fasta_O14727():
    return test_asset_fp / "O14727.fasta"


@pytest.fixture(scope="session")
def dockq_directory():
    return test_asset_fp / "4n7h"


@pytest.fixture(scope="session")
def superimpose_directory():
    return test_asset_fp / "superimpose"


@pytest.fixture(scope="session")
def pisa_assembly_json():
    pisa_dir = test_asset_fp / "pinder_data/pisa/7cm8"
    with open(pisa_dir / "7cm8-pisa-lite-assembly.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def pisa_interface_json():
    pisa_dir = test_asset_fp / "pinder_data/pisa/7cm8"
    with open(pisa_dir / "7cm8-pisa-lite-interfaces.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def pinder_base_dir():
    os.environ["PINDER_BASE_DIR"] = str(test_asset_fp)


@pytest.fixture(scope="session")
def pinder_eval_dir():
    return test_asset_fp / "eval_example"


@pytest.fixture(scope="session")
def pinder_method_test_dir():
    return test_asset_fp / "method_eval"


@pytest.fixture(scope="session")
def pinder_temp_dir():
    with TemporaryDirectory() as tmp_dir:
        os.environ["PINDER_BASE_DIR"] = str(tmp_dir)
        yield Path(tmp_dir)


@pytest.fixture(scope="session")
def pinder_data_copy(pinder_temp_dir):
    src_dir = test_asset_fp / "pinder_data"
    tmp_dir = pinder_temp_dir / "pinder_data"
    if tmp_dir.is_dir():
        shutil.rmtree(str(tmp_dir))

    data_dir = Path(shutil.copytree(src_dir, tmp_dir))
    return data_dir


@pytest.fixture(scope="session")
def pinder_data_dir(pinder_data_copy):
    os.environ["PINDER_DATA_DIR"] = str(pinder_data_copy / "pinder")


@pytest.fixture
def bucket():
    class _blob:
        size: int = 1

        def __init__(self, name):
            self.name = name

        def upload_from_filename(self, path, **kws):
            pass

        def download_to_filename(self, path, **kws):
            pass

        def delete(self, path, **kws):
            pass

    class bkt:
        _blob_cls = _blob

        def __init__(self, name):
            self.name = name

        def list_blobs(self, prefix, **kws):
            return []

        def blob(self, name):
            blob = self._blob_cls(name)
            blob.bucket = self
            return blob

        def copy_blob(self, *args, **kws):
            pass

    return bkt("test")


@pytest.fixture
def client(bucket):
    class _client:
        def create_anonymous_client(self):
            return self

        def bucket(self, name):
            return bucket

        def __call__(self):
            return self

    return _client()


@pytest.fixture
def gs(client):
    from pinder.core.utils.cloud import Gsutil

    return Gsutil(client=client)


@pytest.fixture()
def foldseek_utils_data(test_dir):
    return test_dir / "pinder_data" / "foldseek_utils_data"
