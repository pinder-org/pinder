from pathlib import Path

import pytest
import os
from pinder.core.utils import cloud

try:
    from google.cloud import storage  # noqa: F401
except ImportError:
    pytest.skip(allow_module_level=True)


@pytest.mark.parametrize(
    "name, method, path",
    [
        ("test", "upload_from_filename", "foo"),
        ("test", "download_to_filename", "foo"),
    ],
)
def test_retry_blob_method(name, method, path, tmp_path, bucket):
    blob = bucket.blob(name)
    cloud.retry_blob_method(blob, method, (tmp_path / path).as_posix())


@pytest.mark.parametrize(
    "src_blob, tgt_blob, method",
    [
        ("testa", "testb", "copy_blob"),
    ],
)
def test_retry_bucket_method(src_blob, tgt_blob, method, bucket):
    cloud.retry_bucket_method(bucket.blob(src_blob), bucket.blob(tgt_blob), method)


@pytest.mark.parametrize("nfails", [1, 5])
def test_retry_bucket_method_retry(nfails, tmp_path, bucket, monkeypatch):
    monkeypatch.setattr("pinder.core.utils.cloud.sleep", lambda x: None)

    def copy_blob(self, **kws):
        if self._count < nfails:
            self._count += 1
            raise Exception

    bucket.__class__.copy_blob = copy_blob
    bucket._count = 0
    src_blob = bucket.blob("src_blob")
    tgt_blob = bucket.blob("tgt_blob")
    if nfails == 5:
        with pytest.raises(Exception):
            cloud.retry_bucket_method(src_blob, tgt_blob, "copy_blob")
    else:
        cloud.retry_bucket_method(src_blob, tgt_blob, "copy_blob")


@pytest.mark.parametrize("nfails", [1, 5])
def test_retry_blob_method_retry(nfails, tmp_path, bucket, monkeypatch):
    monkeypatch.setattr("pinder.core.utils.cloud.sleep", lambda x: None)

    def upload_from_filename(self, path, **kws):
        if self._count < nfails:
            self._count += 1
            raise Exception

    bucket._blob_cls.upload_from_filename = upload_from_filename
    bucket._blob_cls._count = 0
    blob = bucket.blob("name")
    if nfails == 5:
        with pytest.raises(Exception):
            cloud.retry_blob_method(
                blob, "upload_from_filename", (tmp_path / "foo").as_posix()
            )
    else:
        cloud.retry_blob_method(
            blob, "upload_from_filename", (tmp_path / "foo").as_posix()
        )


@pytest.mark.parametrize(
    "names, method",
    [
        (["test", "name"], "upload_from_filename"),
        (["test", "name"], "download_to_filename"),
        (["test", "name"], "delete"),
        (["test", "name"], "unknown"),
        ([], "upload_from_filename"),
    ],
)
def test_process_many(names, method, bucket, tmp_path):
    pairs = [(name, bucket.blob(name)) for name in names]
    io = cloud.Gsutil()
    if method in cloud.BLOB_ACTIONS:
        cloud.process_many(pairs, method)
        io.process_many(pairs, method)
    else:
        with pytest.raises(Exception):
            cloud.process_many(pairs, method)
        with pytest.raises(Exception):
            io.process_many(pairs, method)


@pytest.mark.parametrize(
    "names, method",
    [
        (["test", "name"], "copy_blob"),
        (["test", "name"], "unknown"),
        ([], "copy_blob"),
    ],
)
def test_bucket_process_many(names, method, bucket, tmp_path):
    try:
        srcs = [bucket.blob(names[0])]
        tgts = [bucket.blob(names[1])]
    except IndexError:
        srcs = []
        tgts = []
    io = cloud.Gsutil()
    if method in cloud.BUCKET_ACTIONS:
        cloud.bucket_process_many(srcs, tgts, method)
        io.bucket_process_many(srcs, tgts, method)
    else:
        with pytest.raises(Exception):
            cloud.process_many(srcs, tgts, method)
        with pytest.raises(Exception):
            io.process_many(srcs, tgts, method)
        with pytest.raises(Exception):
            cloud.bucket_process_many(srcs, tgts, method)
        with pytest.raises(Exception):
            io.bucket_process_many(srcs, tgts, method)


def test_process_many_future_exception(monkeypatch, bucket):
    class FutureException(Exception):
        pass

    class _condition:
        def acquire(self):
            pass

        def release(self):
            pass

    class future:
        def __init__(self, *args, **kws):
            self.__name = args[-1]
            self._condition = _condition()
            self._state = "FINISHED"

        def exception(self):
            if self.__name in ["name", "copy_blob"]:
                return FutureException

    monkeypatch.setattr(
        "pinder.core.utils.cloud.ThreadPoolExecutor.submit",
        lambda *args, **kws: future(*args, **kws),
    )

    names = ["test", "name"]
    pairs = [(name, bucket.blob(name)) for name in names]
    with pytest.raises(FutureException):
        cloud.process_many(pairs, "upload_from_filename")
    with pytest.raises(FutureException):
        cloud.bucket_process_many(
            [bucket.blob(names[0])],
            [bucket.blob(names[1])],
            "copy_blob",
        )


@pytest.mark.parametrize(
    "sources, targets, local_first, raises",
    [
        ([Path("/path/to/file")], [Path("gs://path/to/file")], True, False),
        ([Path("gs://path/to/file")], [Path("/path/to/file")], False, False),
        ([Path("/path/to/file")], [Path("/path/to/file")], True, True),
        ([Path("gs://path/to/file")], [Path("gs://path/to/file")], False, True),
    ],
)
def test_method_context(sources, targets, local_first, raises):
    if raises:
        with pytest.raises(Exception):
            cloud.method_context(sources, targets)
    else:
        local, remote, action = cloud.method_context(sources, targets)
        if local_first:
            assert local == sources
            assert remote == targets
            assert action == "upload_from_filename"
        else:
            assert remote == sources
            assert local == targets
            assert action == "download_to_filename"


@pytest.mark.parametrize(
    "local_paths, remote_keys",
    [
        (
            [Path("/path/to/file"), Path("/path/to/other")],
            [Path("remote/file"), Path("remote/other")],
        ),
        (
            [Path("/path/to/file"), Path("/path/to/other")],
            ["remote/file", "remote/other"],
        ),
    ],
)
def test_make_path_pairs(local_paths, remote_keys, bucket):
    result = cloud.make_path_pairs(local_paths, remote_keys, bucket)
    for (path_str, blob), key in zip(result, remote_keys):
        if isinstance(key, Path):
            key = key.as_posix()
        assert isinstance(path_str, str)
        assert blob.name == key


def test_make_path_pairs_blobs(bucket):
    paths = [Path("/path/to/file"), Path("/path/to/other")]
    blobs = [bucket.blob(path.as_posix()) for path in paths]
    result = cloud.make_path_pairs(paths, blobs, bucket)
    for (_, blob), orig_blob in zip(result, blobs):
        assert blob.name == orig_blob.name


def test_gsutil_get_bucket_no_client(monkeypatch, client):
    monkeypatch.setattr("pinder.core.utils.cloud.Client", client)
    gs = cloud.Gsutil()
    assert repr(gs)
    got = gs.get_bucket("test")
    assert bool(got)


def test_gsutil_get_bucket_from_client(client):
    gs = cloud.Gsutil(client=client)
    got = gs.get_bucket("test")
    assert got == client.bucket("test")
    assert repr(gs)


def test_gsutil_get_bucket_pass_through(bucket):
    got = cloud.Gsutil().get_bucket("test", bucket=bucket)
    assert got == bucket


def test_gsutil_get_bucket_from_path(client):
    got = cloud.Gsutil(client=client).get_bucket("gs://test")
    assert got.name == "test"


def test_gsutil_get_bucket_cached(client):
    io = cloud.Gsutil(client=client)
    got = io.get_bucket("gs://test")
    assert got.name == "test"
    got = io.get_bucket("gs://test")
    assert got.name == "test"


@pytest.mark.parametrize(
    "sources, target, anchor",
    [
        (
            ["path/to/things", "path/to/stuff"],
            "gs://dne",
            "path/to",
        ),
        (
            ["gs://dne/to/things", "gs://dne/to/stuff"],
            "path/to",
            "gs://dne",
        ),
    ],
)
def test_gsutil_cp_paths(sources, target, anchor, client):
    io = cloud.Gsutil(client=client)
    io.cp_paths(sources, target, anchor)


def test_gsutil_cp_paths_empty(client):
    io = cloud.Gsutil(client=client)
    io.cp_paths([], "test", "test")


def test_gsutil_cp_paths_raises(client):
    io = cloud.Gsutil(client=client)
    with pytest.raises(ValueError):
        io.cp_paths(["foo"], ["bar"], "test")


@pytest.mark.parametrize(
    "source, target",
    [
        (
            "path/to/things",
            "gs://dne",
        ),
        (
            "gs://dne/to/stuff",
            "path/to",
        ),
    ],
)
def test_gsutil_cp_dir(source, target, client):
    io = cloud.Gsutil(client=client)
    io.cp_dir(source, target)


def test_gsutil_cp_dir_no_glob(client):
    io = cloud.Gsutil(client=client)
    with pytest.raises(Exception):
        io.cp_dir("path/to/glob*", "gs://dne")


def test_gsutil_cp_dir_upload_trailing_filter_characters(bucket, monkeypatch):
    source = "path/to/things"
    target = "gs://dne"
    monkeypatch.setattr(
        "pinder.core.utils.cloud.paths.expand_local_path",
        lambda _: [Path(source + "foo"), Path(source + "bar")],
    )

    def list_blobs(self, prefix, **kws):
        return [bucket.blob("hi"), bucket.blob("ho")]

    bucket.list_blobs = lambda *args, **kws: list_blobs(bucket, *args, **kws)
    cloud.Gsutil().cp_dir(source, target, bucket=bucket)


def test_gsutil_cp_dir_download_trailing_filter_characters(bucket):
    source = "gs://dne/h"
    target = "path/to/things"

    def list_blobs(self, prefix, **kws):
        return [bucket.blob("hi"), bucket.blob("ho")]

    bucket.list_blobs = lambda *args, **kws: list_blobs(bucket, *args, **kws)
    cloud.Gsutil().cp_dir(source, target, bucket=bucket)


def test_gsutil_cp_dir_with_remote(bucket):
    def list_blobs(self, prefix, **kws):
        return [bucket.blob("hi"), bucket.blob("ho")]

    bucket.list_blobs = lambda *args, **kws: list_blobs(bucket, *args, **kws)
    source = "gs://dne"
    target = "path/to/things"

    io = cloud.Gsutil()
    io.cp_dir(source, target, bucket=bucket)


@pytest.mark.parametrize("path, num", [("", 0), ("gs://test", 101)])
def test_gsutil_ls(path, num, bucket):
    def list_blobs(self, prefix, **kws):
        return [bucket.blob(f"hi_{i}") for i in range(101)]

    bucket.list_blobs = lambda *args, **kws: list_blobs(bucket, *args, **kws)

    io = cloud.Gsutil()
    ret = io.ls(path, bucket=bucket)
    assert len(ret) == num


def test_gsutil_get_bucket_raises(client):
    def bucket(self, name):
        raise Exception

    client.bucket = lambda *args, **kws: bucket(*args, **kws)
    with pytest.raises(Exception):
        cloud.Gsutil(client=client).get_bucket("dne")


def test_gsutil_ls_blobs(client):
    cloud.Gsutil(client=client).ls_blobs("gs://test")


@pytest.mark.parametrize(
    "max_cpu_fraction, expected_cpu",
    [
        (0.0001, 1),
        (1.0, os.cpu_count()),
        (0.9, os.cpu_count() - 1),
    ],
)
def test_get_container_cpu_frac(max_cpu_fraction, expected_cpu):
    n_cpu = cloud.get_container_cpu_frac(max_cpu_fraction)
    assert n_cpu == expected_cpu


@pytest.mark.parametrize(
    "prefix, result_len",
    [
        ("gs://bucket/foo", 3),
        ("gs://bucket/foo/", 3),
        ("gs://bucket/foo/bar", 1),
        ("gs://bucket/foo/bar/", 1),
        ("gs://bucket/foo/baz", 1),
        ("gs://bucket/foo/baz/", 1),
        ("gs://bucket/foo/ligase.pdb", 1),
    ],
)
def test_gsutil_ls_nonrecursive(bucket, prefix, result_len):
    """Test behavior when passing recursive=False

    google cloud will sometimes have empty folder blobs, but this
    doesn't matter too much for our implementation.
    AFAICT, these empty folder blobs always end in "/"
    """
    blobs = [
        bucket.blob("foo/ligase.pdb"),
        bucket.blob("foo/bar/ubiquitin.pdb"),
        bucket.blob("foo/baz/"),
    ]

    def list_blobs(prefix, delimiter=None, **kws):
        if not delimiter:
            raise Exception

        if prefix.endswith("//") or prefix.endswith(".pdb/"):
            raise Exception

        if not prefix.endswith("/"):
            return [b for b in blobs if b.name == prefix]

        return [b for b in blobs if b.name.startswith(prefix)]

    def exists(self, **kwargs):
        return any([b.name == self.name for b in blobs])

    bucket.list_blobs = list_blobs
    bucket._blob_cls.exists = exists

    gs = cloud.Gsutil()
    result = gs.ls(prefix, recursive=False, bucket=bucket)

    assert len(result) == result_len
