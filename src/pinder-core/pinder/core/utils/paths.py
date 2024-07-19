"""Useful path utilities

See https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
for the source of the tree function.
"""

from __future__ import annotations
import shutil
from collections import defaultdict
from glob import glob
from itertools import repeat
from pathlib import Path
from typing import Any, Generator, List, Union
from pinder.core.utils.log import setup_logger
from pinder.core.utils.process import process_starmap

LOG = setup_logger(__name__)

_StrPath = Union[str, Path]
_ListStrPath = Union[List[str], List[Path]]
_space = "    "
_branch = "│   "
_tee = "├── "
_last = "└── "


def is_local_path(path: _StrPath) -> bool:
    if not isinstance(path, Path):
        path = Path(path)
    try:
        root, *_ = path.parts
    except ValueError:
        return True
    return root != "gs:"


def remote_key(path: _StrPath) -> str:
    if not isinstance(path, Path):
        path = Path(path)
    if is_local_path(path):
        raise Exception(f"{path} does not start with gs://!")
    _, _, *rest = path.parts
    return "/".join(rest)


def strip_glob(path: _StrPath) -> Path:
    if not isinstance(path, Path):
        path = Path(path)
    return Path(path.as_posix().split("*")[0])


def expand_local_path(path: Path) -> list[Path]:
    """Support glob path expansion from path glob shorthand.

    Parameters
    ----------
    path : Path
        path, potentially including asterisks for globbing

    Returns
    -------
    paths : list[Path]
        paths that match the input path
    """
    pstr = path.as_posix()
    if "**" in pstr:
        paths = [Path(name) for name in glob(pstr, recursive=True)]
    elif "*" in pstr:
        paths = [Path(name) for name in glob(pstr)]
    elif path.is_dir():
        paths = [Path(name) for name in glob(f"{pstr}/**", recursive=True)]
    elif path.is_file():
        paths = [path]
    elif len(glob(f"{pstr}*")):
        paths = [Path(name) for name in glob(f"{pstr}*")]
    else:
        paths = []
        print(f"found no files for {path=}")
    return [path for path in paths if path.exists() and not path.is_dir()]


def rmdir(path: Path) -> None:
    """pathlib native rmdir"""
    while len(list(path.rglob("*"))):
        for p in path.rglob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                try:
                    p.rmdir()
                except Exception:
                    pass
    path.rmdir()


def tree(dir_path: Path, prefix: str = "") -> Generator[str, str, None]:
    """Recurse a directory path and build a unix tree
    analog to be printed for debugging purposes.

    Parameters
    ----------
    dir_path : Path
        directory to tree
    prefix : str, default=''
        exposed for recursion

    Returns
    -------
    gen : generator
        the formatted directory contents
    """
    if not prefix:
        yield dir_path.name
    contents = list(sorted(dir_path.iterdir()))
    pointers = [_tee] * (len(contents) - 1) + [_last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():
            extension = _branch if pointer == _tee else _space
            yield from tree(path, prefix=prefix + extension)


def blob_tree(gs_path: Path, gs: Any = None) -> Generator[str, str, None]:
    """Like tree, but for blobs."""
    from pinder.core.utils.cloud import Gsutil

    def recurse(obj: dict[str, Any], prefix: str = "") -> Generator[str, str, None]:
        contents = dict(sorted(obj.items())).items()
        pointers = [_tee] * (len(contents) - 1) + [_last]
        for pointer, (name, sub) in zip(pointers, contents):
            yield name if not prefix else prefix[4:] + pointer + name
            if isinstance(sub, dict):
                extension = _branch if pointer == _tee else _space
                yield from recurse(sub, prefix=prefix + extension)

    if not isinstance(gs, Gsutil):
        gs = Gsutil()

    blobs = gs.ls_blobs(gs_path)
    root = remote_key(gs_path)
    parent = Path(root).parent

    def recursive_defaultdict() -> defaultdict:  # type: ignore
        return defaultdict(recursive_defaultdict)

    grouped = recursive_defaultdict()

    for blob in blobs:
        path = Path(blob.name)
        *nested, name = path.relative_to(parent).parts
        sub = grouped
        for item in nested:
            sub = sub[item]
        sub[name] = None

    yield from recurse(grouped)


def blob_tree_cmd(argv: list[str] | None = None, gs: Any = None) -> None:
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="blob-tree",
        usage="blob-tree gs://mybucket",
        description="like UNIX tree, for GS blobs!",
    )
    parser.add_argument("path", help="gs://path (must start with gs://)")
    argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    if not args.path.startswith("gs://"):
        parser.print_help()
        exit()
    for line in blob_tree(Path(args.path), gs=gs):
        print(line)


def empty_file(file: Path) -> bool:
    if file.is_file():
        return file.stat().st_size == 0
    return True


def _safe_copy_file(
    src: Path,
    dest: Path,
    use_cache: bool = True,
) -> None:
    # Assumes src is a file! Avoids extra is_file if it is known.
    try:
        if not empty_file(dest):
            if use_cache:
                return
            dest.unlink()
        shutil.copy(src, dest)
    except Exception as e:
        LOG.error(f"Failed to copy {src}->{dest}: {e}")


def parallel_copy_files(
    src_files: list[Path],
    dest_files: list[Path],
    use_cache: bool = True,
    max_workers: int | None = None,
    parallel: bool = True,
) -> None:
    """Safely copy list of source files to destination filepaths.

    Operates in parallel and assumes that source files all exist. In case any
    NFS errors cause stale file stat or any other issues are encountered, the
    copy operation is retried up to 10 times before silently exiting.

    Parameters:
    src_files (list[Path]): List of source files to copy. Assumes source is a valid file.
    dest_files (list[Path]): List of fully-qualified destination filepaths. Assumes target directory exists.
    use_cache (bool): Whether to skip copy if destination exists.
    max_workers (int, optional): Limit number of parallel processes spawned to `max_workers`.

    """
    _ = process_starmap(
        _safe_copy_file,
        zip(src_files, dest_files, repeat(use_cache)),
        parallel=parallel,
        max_workers=max_workers,
    )
