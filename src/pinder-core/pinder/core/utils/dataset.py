"""Download published Pinder data over HTTP from R2.

Release manifests supply listings and checksums for root files and directory
sync. Individual structures use MD5 ETags so fetching one structure does not
require downloading a manifest with millions of entries.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from http.client import HTTPException, HTTPMessage
from itertools import islice
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

import pandas as pd

from pinder.core.utils import setup_logger

LOG = setup_logger(__name__)
DEFAULT_MIRROR_URL = "https://pinderdata.org"
_SUPPORTED_RELEASE = "2024-02"
_DOWNLOAD_WORKERS = 8
_BATCH_SIZE = 64
_CHUNK_SIZE = 1024 * 1024
_MAX_ATTEMPTS = 3
_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class ManifestEntry(TypedDict):
    """An object key relative to its release, byte size, and base64 MD5."""

    key: str
    size: int
    md5: str


def mirror_base() -> str:
    """Return the HTTP bucket base, allowing PINDER_MIRROR_URL overrides."""
    base = os.environ.get("PINDER_MIRROR_URL", DEFAULT_MIRROR_URL).rstrip("/")
    parsed = urlsplit(base)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError("PINDER_MIRROR_URL must be a nonempty HTTP(S) bucket base URL")
    return base


def release_url(release: str) -> str:
    """Return the URL of a supported published release."""
    if release != _SUPPORTED_RELEASE:
        raise ValueError("R2 downloads support only PINDER_RELEASE=2024-02")
    return mirror_base() + "/" + release


def mirror_url(uri: str | Path) -> str:
    """Map an unescaped published object URL to the configured mirror.

    Object names are URL-encoded here, including spaces and reserved characters.
    URLs outside the default or configured mirror are rejected.
    """
    return mirror_base() + "/" + quote(_object_key(uri), safe="/")


def _object_key(uri: str | Path) -> str:
    uri = str(uri)
    for base in (mirror_base(), DEFAULT_MIRROR_URL):
        prefix = base + "/"
        if uri.startswith(prefix):
            key = uri[len(prefix) :]
            _validate_key(key)
            release_url(key.split("/", 1)[0])
            return key
    raise ValueError("Dataset downloads require an R2 HTTP URL; GCS is unsupported")


def _validate_key(key: str) -> None:
    if (
        not key
        or key.startswith("/")
        or "\\" in key
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise ValueError(f"Invalid dataset object key: {key!r}")


def _response_size(
    status: int,
    headers: HTTPMessage | Mapping[str, str],
    offset: int,
    etag: str | None,
) -> int | None:
    """Validate a full or resumed response and return the complete object size."""
    length = headers.get("Content-Length")
    if status == 200:
        return int(length) if length is not None else None
    if status != 206:
        raise ValueError(f"Unexpected dataset response status: {status}")

    content_range = re.fullmatch(
        r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", "")
    )
    if not offset or content_range is None:
        raise ValueError("Unexpected partial dataset response")
    start, end, total = map(int, content_range.groups())
    if start != offset or end != total - 1 or headers.get("ETag") != etag:
        raise ValueError("Invalid resumed dataset response")
    if length is not None and int(length) != total - offset:
        raise ValueError("Invalid resumed response length")
    return total


def _verify_checksum(path: Path, expected_md5: str | None, etag: str | None) -> None:
    """Check a manifest MD5 or, if absent, a single-part object's MD5 ETag."""
    checksum = expected_md5
    if checksum is None and etag and re.fullmatch(r'"[0-9a-fA-F]{32}"', etag):
        checksum = base64.b64encode(bytes.fromhex(etag.strip('"'))).decode()
    # Multipart and opaque ETags cannot supply a content checksum.
    if checksum is None:
        return
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = base64.b64encode(digest.digest()).decode()
    if actual != checksum:
        raise ValueError(
            f"Dataset checksum mismatch: expected {checksum}, got {actual}"
        )


def _download(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    expected_md5: str | None = None,
) -> None:
    """Stream to a sibling temporary file and atomically replace on success.

    Interrupted transfers resume within this call using a strong ETag. A server
    returning 200 to a range request restarts the transfer. Failed calls remove
    partial bytes and leave any existing destination intact.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=".part", delete=False
    ) as output:
        temporary = Path(output.name)
    etag: str | None = None
    try:
        for attempt in range(_MAX_ATTEMPTS):
            offset = temporary.stat().st_size if etag else 0
            headers = {
                "User-Agent": "pinder (dataset downloader)",
                "Accept-Encoding": "identity",
            }
            if offset and etag:
                headers.update({"Range": f"bytes={offset}-", "If-Range": etag})
                LOG.debug(f"Resuming {url} at byte {offset}")
            try:
                with urlopen(Request(url, headers=headers), timeout=60) as response:
                    status = getattr(response, "status", 200)
                    total = _response_size(status, response.headers, offset, etag)
                    if (
                        expected_size is not None
                        and total is not None
                        and total != expected_size
                    ):
                        raise ValueError(
                            f"Dataset size differs from manifest: expected {expected_size}, got {total}"
                        )
                    if status == 200:
                        offset = 0
                    response_etag = response.headers.get("ETag")
                    etag = (
                        response_etag
                        if response_etag and not response_etag.startswith("W/")
                        else None
                    )
                    with temporary.open("ab" if offset else "wb") as output:
                        shutil.copyfileobj(response, output, length=_CHUNK_SIZE)
                    size = temporary.stat().st_size
                    if total is not None and size != total:
                        raise OSError(
                            f"Incomplete dataset download: expected {total} bytes, got {size}"
                        )
                    if expected_size is not None and size != expected_size:
                        raise ValueError(
                            f"Dataset size differs from manifest: expected {expected_size}, got {size}"
                        )
                _verify_checksum(temporary, expected_md5, etag)
                os.replace(temporary, destination)
                return
            except (OSError, URLError, HTTPException) as exc:
                retryable = (
                    not isinstance(exc, HTTPError)
                    or exc.code in _RETRYABLE_STATUS_CODES
                )
                if not retryable or attempt == _MAX_ATTEMPTS - 1:
                    LOG.error(f"Download failed: {url} -> {destination}: {exc}")
                    raise
                delay = 2**attempt
                LOG.warning(
                    f"Download interrupted: {url} -> {destination}: {exc}. "
                    f"Retrying in {delay}s (attempt {attempt + 2}/{_MAX_ATTEMPTS})"
                )
                time.sleep(delay)
    except ValueError as exc:
        raise ValueError(f"{url} -> {destination}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def download_files(sources: list[str], destinations: list[Path]) -> None:
    """Download paired object URLs and local paths with bounded concurrency.

    Root objects use manifest checksums regardless of their local filename.
    Individual directory objects use MD5 ETags when available.
    """
    if len(sources) != len(destinations):
        raise ValueError("Each source must have one destination")
    if not sources:
        return
    for source in sources:
        _object_key(source)
    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        # Consume results so any transfer failure reaches the caller.
        pairs = iter(zip(sources, map(Path, destinations)))
        while batch := list(islice(pairs, _BATCH_SIZE)):
            batch_sources, batch_destinations = zip(*batch)
            list(pool.map(_download_file, batch_sources, batch_destinations))


def _read_local_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe extension: {path.suffix}")


def read_dataframe(uri: str | Path) -> pd.DataFrame:
    """Read a local CSV/Parquet file or a verified release-root object.

    Remote files are streamed to temporary disk storage before pandas reads
    them. This avoids retaining an additional copy of the response in memory.
    """
    uri = str(uri)
    if not urlsplit(uri).scheme:
        return _read_local_dataframe(Path(uri))
    record = _root_manifest_entry(uri)
    with TemporaryDirectory() as directory:
        local = Path(directory) / record["key"]
        _download(mirror_url(uri), local, record["size"], record["md5"])
        return _read_local_dataframe(local)


def _root_manifest_entry(uri: str | Path) -> ManifestEntry:
    release, key = _object_key(uri).split("/", 1)
    # Close the iterator when returning early so its temporary file is removed.
    with closing(iter_manifest(release_url(release))) as records:
        for record in records:
            if record["key"] == key:
                return record
    raise ValueError(f"File absent from release manifest: {uri}")


def iter_manifest(
    root: str, directory: str = ""
) -> Generator[ManifestEntry, None, None]:
    """List one flat dataset directory via a release-relative JSONL manifest.

    Manifests live at <release>/manifests/{root,pdbs,mappings,test_set_pdbs}.jsonl.gz
    Entries contain a release-relative key, byte size, and base64 MD5. Records
    are validated and yielded individually to bound memory use.
    """
    if directory not in {"", "pdbs", "mappings", "test_set_pdbs"}:
        raise ValueError(f"Unsupported dataset directory: {directory}")
    manifest = f"{root}/manifests/{directory or 'root'}.jsonl.gz"
    url = mirror_url(manifest)
    with TemporaryDirectory() as temporary:
        local = Path(temporary) / "manifest.jsonl.gz"
        _download(url, local)
        with gzip.open(local, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                key = record["key"]
                _validate_key(key)
                if str(PurePosixPath(key).parent) != (directory or "."):
                    raise ValueError(f"Object outside manifest directory: {key!r}")
                size = record.get("size")
                if type(size) is not int or size < 0:
                    raise ValueError("Invalid manifest size")
                try:
                    checksum = base64.b64decode(record["md5"], validate=True)
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError("Invalid manifest checksum") from exc
                if len(checksum) != 16:
                    raise ValueError("Invalid manifest checksum")
                yield ManifestEntry(key=key, size=size, md5=record["md5"])


def list_files(root: str, directory: str = "") -> Iterator[str]:
    """Yield object URLs from one directory manifest without loading it into memory."""
    for record in iter_manifest(root, directory):
        yield root + "/" + record["key"]


def _download_file(source: str, destination: Path) -> None:
    url = mirror_url(source)
    _, key = _object_key(source).split("/", 1)
    if "/" not in key:
        record = _root_manifest_entry(source)
        _download(url, destination, record["size"], record["md5"])
    else:
        _download(url, destination)


def sync_directory(root: str, directory: str, local: Path) -> None:
    """Consume a manifest in bounded batches, verifying each missing file."""
    records = (
        record
        for record in iter_manifest(root, directory)
        if not (local / PurePosixPath(record["key"]).name).is_file()
    )

    def download(record: ManifestEntry) -> None:
        _download(
            mirror_url(root + "/" + record["key"]),
            local / PurePosixPath(record["key"]).name,
            record["size"],
            record["md5"],
        )

    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        while batch := list(islice(records, _BATCH_SIZE)):
            list(pool.map(download, batch))
