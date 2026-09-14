import base64
import builtins
import gzip
import hashlib
import json
import socket
from functools import partial
from http.server import (
    BaseHTTPRequestHandler,
    SimpleHTTPRequestHandler,
    ThreadingHTTPServer,
)
from io import BytesIO
from threading import Lock, Thread
from types import SimpleNamespace
from urllib.error import HTTPError

import pandas as pd
import pytest

from pinder.core.utils import dataset


def write_manifest(root, group, keys):
    path = root / "manifests" / (group + ".jsonl.gz")
    path.parent.mkdir(exist_ok=True)
    with gzip.open(path, "wt") as stream:
        for key in keys:
            data = (root / key).read_bytes() if (root / key).exists() else b""
            stream.write(
                json.dumps(
                    {
                        "key": key,
                        "size": len(data),
                        "md5": base64.b64encode(hashlib.md5(data).digest()).decode(),
                    }
                )
                + "\n"
            )


@pytest.fixture
def mirror(tmp_path, monkeypatch):
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PINDER_MIRROR_URL", f"http://127.0.0.1:{server.server_port}")
    yield tmp_path
    server.shutdown()
    thread.join()
    server.server_close()


def test_download_preserves_bytes_and_failure_preserves_cache(mirror, tmp_path):
    source = mirror / "2024-02/pdbs/a #?.pdb"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ATOM\n")
    dest = tmp_path / "cache/a.pdb"
    dataset.download_files(["https://pinderdata.org/2024-02/pdbs/a #?.pdb"], [dest])
    assert dest.read_bytes() == b"ATOM\n"
    with pytest.raises(Exception):
        dataset.download_files(["https://pinderdata.org/2024-02/missing.pdb"], [dest])
    assert dest.read_bytes() == b"ATOM\n"
    assert list(dest.parent.iterdir()) == [dest]


def test_dataframe_and_manifest(mirror):
    root = mirror / "2024-02"
    root.mkdir()
    (root / "metadata-extra.csv.gz").write_bytes(gzip.compress(b"id,value\na,2\n"))
    manifests = root / "manifests"
    manifests.mkdir()
    write_manifest(root, "root", ["metadata-extra.csv.gz"])
    assert list(dataset.list_files("https://pinderdata.org/2024-02")) == [
        "https://pinderdata.org/2024-02/metadata-extra.csv.gz"
    ]
    df = dataset.read_dataframe("https://pinderdata.org/2024-02/metadata-extra.csv.gz")
    assert df.to_dict("records") == [{"id": "a", "value": 2}]


@pytest.mark.parametrize("key", ["../escape", "/absolute", "pdbs/../../escape"])
def test_rejects_manifest_traversal(mirror, key):
    manifests = mirror / "2024-02/manifests"
    manifests.mkdir(parents=True)
    (manifests / "root.jsonl.gz").write_bytes(
        gzip.compress((json.dumps({"key": key}) + "\n").encode())
    )
    with pytest.raises(ValueError):
        list(dataset.list_files("https://pinderdata.org/2024-02"))


def test_other_release_fails_without_gcs_fallback(mirror):
    with pytest.raises(ValueError, match="2024-02"):
        dataset.read_dataframe("https://pinderdata.org/2023-11/index.parquet")


def test_empty_mirror_cannot_enable_gcs(monkeypatch):
    monkeypatch.setenv("PINDER_MIRROR_URL", "")
    with pytest.raises(ValueError, match="nonempty HTTP"):
        dataset.read_dataframe("https://pinderdata.org/2024-02/index.parquet")


@pytest.mark.parametrize(
    "uri", ["gs://pinder/2024-02/index.parquet", "gs://custom/index.parquet"]
)
def test_gcs_sources_are_rejected(uri):
    with pytest.raises(ValueError, match="GCS is unsupported"):
        dataset.read_dataframe(uri)


def test_local_dataframe_still_supported(tmp_path):
    path = tmp_path / "custom.csv"
    path.write_text("id,value\na,2\n")
    assert dataset.read_dataframe(path).to_dict("records") == [{"id": "a", "value": 2}]


def test_supplementary_data_and_sync_use_mirror(mirror, tmp_path, monkeypatch):
    from pinder.core.index import utils

    monkeypatch.setenv("PINDER_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PINDER_RELEASE", "2024-02")
    root = mirror / "2024-02"
    root.mkdir()
    pd.DataFrame({"id": ["a"], "neff": [2.0]}).to_parquet(
        root / "test_split_paired_neffs.parquet"
    )
    manifests = root / "manifests"
    manifests.mkdir()
    write_manifest(root, "pdbs", ["pdbs/a.pdb"])
    write_manifest(root, "root", ["test_split_paired_neffs.parquet"])
    utils.get_supplementary_data.cache_clear()
    try:
        assert utils.get_supplementary_data(
            utils.SupplementaryData.paired_neff
        ).neff.tolist() == [2.0]
        remote, local = utils.get_missing_blobs("pdbs")
        assert remote == [utils.get_pinder_bucket_root() + "/pdbs/a.pdb"]
        assert local == [tmp_path / "cache/pdbs/a.pdb"]
    finally:
        utils.get_supplementary_data.cache_clear()


def test_structure_download_uses_mirror(mirror, tmp_path, monkeypatch):
    from pinder.core.index.system import PinderSystem

    root = mirror / "2024-02/pdbs"
    root.mkdir(parents=True)
    (root / "a.pdb").write_bytes(b"ATOM\n")
    system = PinderSystem.__new__(PinderSystem)
    system.entry = SimpleNamespace(
        pdb_paths={"apo_R": "pdbs/a.pdb"}, mapping_paths={}, test_system=False
    )
    system.pinder_root = tmp_path / "cache"
    system.pdbs_path = system.pinder_root / "pdbs"
    monkeypatch.setenv("PINDER_RELEASE", "2024-02")
    system.download_entry()
    assert (system.pdbs_path / "a.pdb").read_bytes() == b"ATOM\n"


def test_extra_metadata_discovery_uses_mirror(mirror, tmp_path, monkeypatch):
    from pinder.core.index import utils

    root = mirror / "2024-02"
    (root / "manifests").mkdir(parents=True)
    (root / "metadata-extra.csv.gz").write_bytes(gzip.compress(b"id,value\na,2\n"))
    write_manifest(root, "root", ["metadata-extra.csv.gz"])
    utils.get_extra_metadata.cache_clear()
    try:
        df = utils.get_extra_metadata(
            str(tmp_path / "cache"), "https://pinderdata.org/2024-02", update=True
        )
        assert df.to_dict("records") == [{"id": "a", "value": 2}]
    finally:
        utils.get_extra_metadata.cache_clear()


def test_short_response_retries_before_replacing_cache(tmp_path, monkeypatch):
    class Response(BytesIO):
        headers = {"Content-Length": "5"}

    responses = iter([Response(b"ab"), Response(b"abcde")])
    monkeypatch.setattr(dataset, "urlopen", lambda *a, **kw: next(responses))
    monkeypatch.setattr(dataset.time, "sleep", lambda seconds: None)
    destination = tmp_path / "object"
    destination.write_bytes(b"old")
    dataset._download("https://example.invalid/object", destination)
    assert destination.read_bytes() == b"abcde"
    assert list(tmp_path.iterdir()) == [destination]


def test_archives_download_from_mirror(mirror, tmp_path, monkeypatch):
    from pinder.core.index import utils

    root = mirror / "2024-02"
    root.mkdir()
    for name in ["pdbs", "test_set_pdbs", "mappings"]:
        (root / f"{name}.zip").write_bytes(b"archive")
    write_manifest(
        root, "root", [name + ".zip" for name in ["pdbs", "test_set_pdbs", "mappings"]]
    )
    monkeypatch.setenv("PINDER_DATA_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("PINDER_RELEASE", "2024-02")
    utils.download_dataset(skip_inflation=True)
    assert {p.name: p.read_bytes() for p in (tmp_path / "cache").glob("*.zip")} == {
        f"{name}.zip": b"archive" for name in ["pdbs", "test_set_pdbs", "mappings"]
    }


def test_checksum_mismatch_does_not_replace_cache(tmp_path, monkeypatch):
    class Response(BytesIO):
        headers = {"Content-Length": "3"}
        status = 200

    monkeypatch.setattr(dataset, "urlopen", lambda *a, **kw: Response(b"bad"))
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    target = tmp_path / "file"
    target.write_bytes(b"old")
    with pytest.raises(ValueError, match="checksum"):
        dataset._download(
            "https://example.invalid/file",
            target,
            expected_size=3,
            expected_md5="kAFQmDzST7DWlj99KOF/cg==",
        )
    assert target.read_bytes() == b"old"


def test_interrupted_download_resumes_with_if_range(tmp_path, monkeypatch):
    requests = []

    class Response(BytesIO):
        def __init__(self, data, status, headers):
            super().__init__(data)
            self.status = status
            self.headers = headers

    def open_response(request, **kwargs):
        requests.append(request)
        if len(requests) == 1:
            return Response(b"ab", 200, {"Content-Length": "5", "ETag": '"version1"'})
        assert request.get_header("Range") == "bytes=2-"
        assert request.get_header("If-range") == '"version1"'
        return Response(
            b"cde",
            206,
            {
                "Content-Length": "3",
                "Content-Range": "bytes 2-4/5",
                "ETag": '"version1"',
            },
        )

    monkeypatch.setattr(dataset, "urlopen", open_response)
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    target = tmp_path / "file"
    dataset._download("https://example.invalid/file", target, expected_size=5)
    assert target.read_bytes() == b"abcde"
    assert len(requests) == 2


@pytest.mark.parametrize(
    "status,etag,content_range,expected",
    [
        (200, '"v2"', None, b"abcde"),
        (206, '"v1"', "bytes 1-4/5", None),
        (206, '"v2"', "bytes 2-4/5", None),
    ],
)
def test_resume_restarts_or_rejects_invalid_response(
    tmp_path, monkeypatch, status, etag, content_range, expected
):
    class Response(BytesIO):
        def __init__(self, data, status, headers):
            super().__init__(data)
            self.status = status
            self.headers = headers

    headers = {"Content-Length": "5" if status == 200 else "3", "ETag": etag}
    if content_range:
        headers["Content-Range"] = content_range
    responses = iter(
        [
            Response(b"ab", 200, {"Content-Length": "5", "ETag": '"v1"'}),
            Response(b"abcde" if status == 200 else b"cde", status, headers),
        ]
    )
    monkeypatch.setattr(dataset, "urlopen", lambda *a, **kw: next(responses))
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    dest = tmp_path / "file"
    dest.write_bytes(b"old")
    if expected is None:
        with pytest.raises(ValueError, match="resumed"):
            dataset._download("https://example.invalid/file", dest)
        assert dest.read_bytes() == b"old"
    else:
        dataset._download("https://example.invalid/file", dest)
        assert dest.read_bytes() == expected


def test_manifest_is_parsed_lazily(mirror):
    root = mirror / "2024-02"
    root.mkdir()
    (root / "a.csv").write_bytes(b"id\na\n")
    write_manifest(root, "root", ["a.csv"])
    with gzip.open(root / "manifests/root.jsonl.gz", "at") as stream:
        stream.write("not-json\n")
    records = dataset.iter_manifest("https://pinderdata.org/2024-02")
    assert next(records)["key"] == "a.csv"
    with pytest.raises(json.JSONDecodeError):
        next(records)


def test_sync_consumes_bounded_batches(tmp_path, monkeypatch):
    state = {"produced": 0, "consumed": 0}
    lock = Lock()

    def records(*args):
        for i in range(1000):
            with lock:
                state["produced"] += 1
                assert state["produced"] - state["consumed"] <= 64
            yield {"key": f"pdbs/{i}.pdb", "size": 1, "md5": "checksum"}

    def download(url, path, size, md5):
        assert size == 1 and md5 == "checksum"
        with lock:
            state["consumed"] += 1

    monkeypatch.setenv("PINDER_MIRROR_URL", "https://example.invalid")
    monkeypatch.setattr(dataset, "iter_manifest", records)
    monkeypatch.setattr(dataset, "_download", download)
    dataset.sync_directory("https://pinderdata.org/2024-02", "pdbs", tmp_path)
    assert state["consumed"] == 1000


def test_resume_over_real_http(tmp_path, monkeypatch):
    payload = b"abcdef" * 500000
    checksum = base64.b64encode(hashlib.md5(payload).digest()).decode()
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            ranges.append(requested)
            start = int(requested.split("=")[1].split("-")[0]) if requested else 0
            self.send_response(206 if requested else 200)
            self.send_header("ETag", '"stable"')
            self.send_header("Content-Length", str(len(payload) - start))
            if requested:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            if len(ranges) == 1:
                self.wfile.write(payload[:1500000])
                self.wfile.flush()
                self.connection.shutdown(socket.SHUT_WR)
            else:
                self.wfile.write(payload[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(dataset.time, "sleep", lambda _: None)
    try:
        dest = tmp_path / "archive.zip"
        dataset._download(
            f"http://127.0.0.1:{server.server_port}/file", dest, len(payload), checksum
        )
        assert dest.read_bytes() == payload
        assert len(ranges) == 2 and ranges[1] == "bytes=1500000-"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_default_routes_only_published_release_to_r2(monkeypatch):
    monkeypatch.delenv("PINDER_MIRROR_URL", raising=False)
    assert dataset.mirror_url("https://pinderdata.org/2024-02/index.parquet") == (
        "https://pinderdata.org/2024-02/index.parquet"
    )
    with pytest.raises(ValueError, match="2024-02"):
        dataset.mirror_url("https://pinderdata.org/2023-11/index.parquet")
    with pytest.raises(ValueError, match="GCS is unsupported"):
        dataset.mirror_url("gs://custom/2024-02/index.parquet")


def test_default_archive_uses_manifest_checksum(monkeypatch, tmp_path):
    monkeypatch.delenv("PINDER_MIRROR_URL", raising=False)
    roots = []

    def manifest(root):
        roots.append(root)
        yield {"key": "pdbs.zip", "size": 7, "md5": "checksum"}

    calls = []
    monkeypatch.setattr(dataset, "iter_manifest", manifest)
    monkeypatch.setattr(dataset, "_download", lambda *args: calls.append(args))
    destination = tmp_path / "pdbs.zip"
    dataset.download_files(["https://pinderdata.org/2024-02/pdbs.zip"], [destination])
    assert roots == ["https://pinderdata.org/2024-02"]
    assert calls == [
        ("https://pinderdata.org/2024-02/pdbs.zip", destination, 7, "checksum")
    ]


@pytest.fixture(autouse=True)
def forbid_gcs_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(
            ("gcsfs", "google.cloud.storage", "pinder.core.utils.cloud")
        ):
            raise AssertionError("Distribution code imported GCS: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("local_name", ["archive.zip", "archive.bin"])
def test_root_download_verifies_checksum_independent_of_destination(
    mirror, tmp_path, local_name
):
    root = mirror / "2024-02"
    root.mkdir()
    archive = root / "pdbs.zip"
    archive.write_bytes(b"correct")
    write_manifest(root, "root", ["pdbs.zip"])
    archive.write_bytes(b"corrupt")
    destination = tmp_path / "cache" / local_name
    destination.parent.mkdir()
    destination.write_bytes(b"cached")

    with pytest.raises(ValueError, match="checksum mismatch") as error:
        dataset.download_files(
            ["https://pinderdata.org/2024-02/pdbs.zip"], [destination]
        )

    assert "2024-02/pdbs.zip" in str(error.value)
    assert str(destination) in str(error.value)
    assert destination.read_bytes() == b"cached"
    assert list(destination.parent.iterdir()) == [destination]


def test_root_object_name_is_encoded_once(mirror, tmp_path):
    root = mirror / "2024-02"
    root.mkdir()
    name = "metadata #?.csv"
    (root / name).write_bytes(b"id,value\na,2\n")
    write_manifest(root, "root", [name])
    destination = tmp_path / "cache" / "renamed.bin"

    dataset.download_files([f"https://pinderdata.org/2024-02/{name}"], [destination])

    assert destination.read_bytes() == b"id,value\na,2\n"
    assert dataset.read_dataframe(f"https://pinderdata.org/2024-02/{name}").to_dict(
        "records"
    ) == [{"id": "a", "value": 2}]


@pytest.mark.parametrize("status,attempts", [(404, 1), (503, 3)])
def test_http_failure_retry_policy_preserves_cache(
    monkeypatch, tmp_path, caplog, status, attempts
):
    monkeypatch.setattr(dataset.LOG, "propagate", True)
    calls = []
    delays = []
    url = "https://example.invalid/object"

    def fail(request, **kwargs):
        calls.append(request)
        raise HTTPError(url, status, "unavailable", {}, None)

    monkeypatch.setattr(dataset, "urlopen", fail)
    monkeypatch.setattr(dataset.time, "sleep", delays.append)
    destination = tmp_path / "object"
    destination.write_bytes(b"cached")

    with pytest.raises(HTTPError) as error:
        dataset._download(url, destination)

    assert error.value.code == status
    assert len(calls) == attempts
    assert delays == ([1, 2] if status == 503 else [])
    assert url in caplog.text
    assert str(destination) in caplog.text
    assert destination.read_bytes() == b"cached"
    assert list(tmp_path.iterdir()) == [destination]
