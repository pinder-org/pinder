from pathlib import Path

import pytest
from pinder.core.utils import paths

PATHS = [
    ("/path/to/local", True),
    ("gs://path/to/remote", False),
    (Path("/path/to/local"), True),
    (Path("gs://path/to/remote"), False),
    (".", True),
]


@pytest.fixture
def with_files(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    files = [
        tmp_path / "afile",
        tmp_path / "bfile",
        subdir / "afile",
        subdir / "bfile",
    ]
    for fl in files:
        with fl.open("w") as f:
            f.write("")


@pytest.mark.parametrize("path, local", PATHS)
def test_is_local_path(path, local):
    assert paths.is_local_path(path) == local


@pytest.mark.parametrize("path", ["/", "/root", "/path/to/file"])
def test_is_local_path_edge_case(path):
    assert paths.is_local_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "/path/to/local",
        "/path/to/*",
        "/path/to/partial*",
        "gs://path/to/local",
        "gs://path/to/*",
        "gs://path/to/partial*",
    ],
)
def test_strip_glob(path):
    result = paths.strip_glob(path)
    assert "*" not in result.as_posix()


@pytest.mark.parametrize("path, local", PATHS)
def test_remote_key(path, local):
    if not local:
        assert isinstance(paths.remote_key(Path(path)), str)
    else:
        with pytest.raises(Exception):
            paths.remote_key(path)


def test_expand_local_path_file(tmp_path):
    path = tmp_path / "afile"
    with path.open("w") as f:
        f.write("")
    ps = paths.expand_local_path(path)
    assert len(ps) == 1
    assert ps[0] == path


def test_expand_local_path_file_edge_case(tmp_path):
    path = tmp_path / "afile"
    with path.open("w") as f:
        f.write("")
    ps = paths.expand_local_path(path)  # Path(path.as_posix() + "/"))
    assert len(ps) == 1
    assert ps[0] == path


@pytest.mark.parametrize("num_files", [0, 1, 2])
def test_expand_local_path_dir(num_files, tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    for _ in range(num_files):
        with (subdir / f"file{_}").open("w") as f:
            f.write("")
    ps = paths.expand_local_path(subdir)
    assert len(ps) == num_files


@pytest.mark.parametrize(
    "glob, match",
    [
        ("**", 4),
        ("**/afile", 2),
        ("**/bfile", 2),
        ("af**", 1),
        ("bf**", 1),
        ("af*", 1),
        ("bf*", 1),
        ("*", 2),
        ("dne", 0),
        ("af", 1),  # if strip_glob was called before expand_local_path
    ],
)
def test_expand_local_path_glob(glob, match, tmp_path, with_files):
    ps = paths.expand_local_path(tmp_path / glob)
    assert len(ps) == match


def test_rmdir(with_files, tmp_path):
    paths.rmdir(tmp_path)
    assert not tmp_path.is_dir()


def test_tree(with_files, tmp_path):
    tree = list(paths.tree(tmp_path))
    assert len(tree)


@pytest.mark.parametrize("gs", [None, ""])
def test_blob_tree(gs, client):
    from pinder.core.utils import cloud

    bucket = client.bucket("hi")

    def ls_blobs(self, target, **kws):
        return [
            bucket.blob("afile"),
            bucket.blob("bfile"),
            bucket.blob("subdir/afile"),
            bucket.blob("subdir/bfile"),
        ]

    orig = cloud.Gsutil.ls_blobs
    cloud.Gsutil.ls_blobs = ls_blobs
    if gs is not None:
        gs = cloud.Gsutil(client=client)
    tree = list(paths.blob_tree("gs://blah", gs=gs))
    assert len(tree)
    cloud.Gsutil.ls_blobs = orig


@pytest.mark.parametrize("args", [["gs://blah"], ["malformed"], []])
def test_blob_tree_cmd(args, client):
    from pinder.core.utils import cloud

    bucket = client.bucket("hi")

    def ls_blobs(self, target, **kws):
        return [
            bucket.blob("afile"),
            bucket.blob("bfile"),
            bucket.blob("subdir/afile"),
            bucket.blob("subdir/bfile"),
        ]

    orig = cloud.Gsutil.ls_blobs
    cloud.Gsutil.ls_blobs = ls_blobs
    gs = cloud.Gsutil(client=client)
    if not len(args) or not args[0].startswith("gs"):
        with pytest.raises(SystemExit):
            paths.blob_tree_cmd(argv=args, gs=gs)
    else:
        paths.blob_tree_cmd(argv=args, gs=gs)
    cloud.Gsutil.ls_blobs = orig
