from __future__ import annotations
import logging
import http.client
import sys
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait
from functools import lru_cache
from gzip import GzipFile
from math import floor
from os import cpu_count
from pathlib import Path
from time import sleep
from typing import Any, ClassVar, Iterator, List, Optional, Tuple, Union
from tqdm.std import tqdm

import gcsfs
import pandas as pd
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client

from pinder.core.utils import paths
from pinder.core.utils.timer import timeit
from pinder.core.utils import setup_logger


logging.getLogger("urllib3").setLevel(logging.CRITICAL)

UPLOAD = "upload_from_filename"
DOWNLOAD = "download_to_filename"
BLOB_ACTIONS = [UPLOAD, DOWNLOAD]

REMOTE_COPY = "copy_blob"
BUCKET_ACTIONS = [REMOTE_COPY]
LOG = setup_logger(__name__)


def _safe_decode(text: str, incoming: str | None = None, errors: str = "strict") -> str:
    """Decodes incoming text/bytes string using `incoming` if they're not
       already unicode.

    Parameters
    ----------
    incoming
        Text's current encoding
    errors
        Errors handling policy. See here for valid values
        http://docs.python.org/2/library/codecs.html

    Returns
    -------
    str
        text or a unicode `incoming` encoded representation of it.

    Raises
    ------
    TypeError
        If text is not an instance of str
    """
    if not isinstance(text, (str, bytes)):
        raise TypeError("%s can't be decoded" % type(text))

    if isinstance(text, str):
        return text

    if not incoming:
        incoming = getattr(sys.stdin, "encoding", None) or sys.getdefaultencoding()

    try:
        return text.decode(incoming, errors)
    except UnicodeDecodeError:
        return text.decode("utf-8", errors)


def _format_with_unicode_kwargs(msg_format: str, kwargs: dict[str, str]) -> str:
    try:
        return msg_format % kwargs
    except UnicodeDecodeError:
        try:
            kwargs = {k: _safe_decode(v) for k, v in kwargs.items()}
        except UnicodeDecodeError:
            return msg_format
        return msg_format % kwargs


class Error(Exception):
    """Base error class

    Child classes have a title and a message format.
    """

    title: str | None = None
    code: int | None = None
    message_format: str | None = None

    def __init__(self, message: str | None = None, **kwargs: str) -> None:
        try:
            message = self._build_message(message, **kwargs)
        except KeyError:
            LOG.warning("missing exception kwargs")
            message = self.message_format

        super(Error, self).__init__(message)

    def _build_message(self, message: str | None, **kwargs: str) -> str:
        if message:
            return message
        assert isinstance(self.message_format, str)
        return _format_with_unicode_kwargs(self.message_format, kwargs)


class GCPError(Error):
    message_format = "Unexpected GCP Error"
    code = int(http.client.BAD_GATEWAY)
    title = http.client.responses[http.client.BAD_GATEWAY]


class GCPTimeOutError(GCPError):
    message_format = (
        "The GCP request you have made timed out after attempting to reconnect"
    )
    code = int(http.client.GATEWAY_TIMEOUT)
    title = http.client.responses[http.client.GATEWAY_TIMEOUT]


@lru_cache
def get_cpu_limit(
    reserve_cpu: int = 0,
    quota_path: Path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"),
    period_path: Path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"),
    shares_path: Path = Path("/sys/fs/cgroup/cpu/cpu.shares"),
) -> int:
    """See https://bugs.python.org/issue36054 for more details.
    Attempt to correctly obtain the available CPU resources
    allocated to a given container since os.cpu_count always
    returns the machine resources, not the resources allocated
    to the container.
    """
    cfs_quota_us = None
    if quota_path.is_file():
        with quota_path.open() as f:
            cfs_quota_us = int(f.read().strip())
    avail_cpu = None
    if period_path.is_file():
        with period_path.open() as f:
            cfs_period_us = int(f.read().strip())
        if cfs_quota_us is not None:
            avail_cpu = cfs_quota_us // cfs_period_us
    elif shares_path.is_file():
        with shares_path.open() as f:
            cpu_shares = int(f.read().strip())
        avail_cpu = int(cpu_shares / 1024)
    if avail_cpu is None:
        LOG.debug("could not find cgroup cpu limits")
        avail_cpu = -1
    avail_cpu -= reserve_cpu
    if avail_cpu > 0:
        return avail_cpu
    avail_cpu = cpu_count()
    if avail_cpu is None:
        avail_cpu = 1
    return max([avail_cpu - reserve_cpu, 1])


def get_container_cpu_frac(max_cpu_fraction: float = 0.9) -> int:
    cpu_limit = get_cpu_limit()
    cpu = floor(max_cpu_fraction * cpu_limit) or 1
    return cpu


def retry_bucket_method(
    source_blob: Blob,
    target_blob: Blob,
    method: str,
    timeout: Tuple[int, int] = (5, 120),
    retries: int = 5,
) -> None:
    """Retry wrapper around bucket.copy_blob to
    tolerate intermittent network outages. Can be generalized
    to other Bucket methods which are not also provided as
    convenience methods on the Blob object.

    Parameters
    ----------
    source_blob : Blob
        blob to copy
    target_blob : Blob
        new location of source blob
    timeout : tuple, default=(5, 120)
        timeout forwarded to Bucket method
    retries : int, default=5
        how many times to attempt the method
    """
    exc = None
    func = getattr(source_blob.bucket, method)
    for i in range(1, retries + 1):
        try:
            func(
                blob=source_blob,
                destination_bucket=target_blob.bucket,
                new_name=target_blob.name,
                timeout=timeout,
            )
            return None
        except Exception as e:
            # exponential backoff on retry
            name = Path(source_blob.name).name
            msg = f"{method} {name} hit {repr(e)}, sleeping {2 ** i}s"
            LOG.error(msg)
            exc = e
            sleep(2**i)
            continue
    LOG.error(f"failed to copy blob to {target_blob.name}: {repr(exc)}")
    raise GCPTimeOutError(f"Timeout error for {source_blob.name}")


@timeit
def bucket_process_many(
    sources: List[Blob],
    targets: List[Blob],
    method: str,
    global_timeout: int = 600,
    timeout: Tuple[int, int] = (5, 120),
    retries: int = 5,
    threads: int = 4,
) -> None:
    """Use a ThreadPoolExecutor to execute multiple bucket
    operations concurrently using the python storage API.

    Parameters
    ----------
    sources : List[Blob]
        source blobs
    targets : List[Blob]
        target blobs
    method : str
        name of Bucket method to call
    global_timeout : int=600
        timeout to wait for all operations to complete
    timeout : tuple, default=(5, 120)
        timeout forwarded to Bucket method
    retries : int, default=5
        how many times to attempt the method
    threads : int, default=4
        how many threads to use in the thread pool
    """
    if not len(sources) or not len(targets):
        LOG.warning("Gsutil bucket_process_many was called with no files")
        return
    if method not in BUCKET_ACTIONS:
        raise Exception(f"{method=} not in {BUCKET_ACTIONS=}")
    threads = min(threads, len(sources))
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                retry_bucket_method,
                source,
                target,
                method,
                timeout=timeout,
                retries=retries,
            )
            for source, target in zip(sources, targets)
        ]
        wait(
            futures,
            timeout=global_timeout,
            return_when=ALL_COMPLETED,
        )
        for future in futures:
            if future.exception():
                raise future.exception()  # type: ignore


def retry_blob_method(
    blob: Blob,
    method: str,
    path: str,
    retries: int = 5,
    timeout: Tuple[int, int] = (5, 120),
) -> None:
    f"""Thin wrapper around Blob methods to add a retry
    loop. Supports the following actions: {BLOB_ACTIONS}.

    Parameters
    ----------
    blob : Blob
        the blob to process
    method : str
        "{UPLOAD}" | "{DOWNLOAD}"
    path : str
        the path string to upload to or download to
    retries : int, default=5
        how many times to attempt the method
    timeout : tuple, default=(5, 120)
        forwarded to the Blob method itself
    """
    kws = {UPLOAD: {"timeout": timeout}}.get(method, {})
    exc = None
    func = getattr(blob, method)
    # bucket.blob doesn't return blob metadata - bucket.get_blob does.
    blob_name = blob.name
    if not blob.size:
        blob = blob.bucket.get_blob(blob.name)
    # In case blob.size was not defined, bucket.get_blob may return a NoneType.
    if blob is None:
        raise FileNotFoundError(
            f"Requested storage blob <{blob_name}> does not exist! "
            "Verify that the provided GCS path points to a valid file!"
        )

    # Only show pbar at 500Mb
    large_file = blob.size / 10**6 > 500
    if method == UPLOAD:
        LOG.debug(f"{path} -> {blob.name}")
    elif method == DOWNLOAD:
        LOG.debug(f"{blob.name} -> {path}")
    else:
        LOG.debug(f"remote rm {blob.name}")
    for i in range(1, retries + 1):
        try:
            if method == DOWNLOAD:
                Path(path).parent.mkdir(exist_ok=True, parents=True)
            if method == DOWNLOAD and large_file:
                with open(Path(path), "wb") as f:
                    with tqdm.wrapattr(
                        f, "write", total=blob.size, desc=f"Downloading {blob.name}"
                    ) as file_obj:
                        blob.download_to_file(file_obj)
                return None
            func(path, **kws)
            return None
        except Exception as e:
            # exponential backoff on retry
            LOG.error(f"{Path(blob.name).name} hit {repr(e)}, sleeping {2 ** i}s")
            exc = e
            sleep(2**i)
            continue
    LOG.error(f"failed on {method} for {path}: {repr(exc)}")
    raise Exception(f"Timeout error {path}")


@timeit
def process_many(
    path_pairs: List[Tuple[str, Blob]],
    method: str,
    global_timeout: int = 600,
    timeout: Tuple[int, int] = (5, 120),
    retries: int = 5,
    threads: int = 4,
) -> None:
    f"""Use a ThreadPoolExecutor to execute multiple uploads
    or downloads concurrently using the python storage API.

    Parameters
    ----------
    path_pairs : list
        tuples of (path_str, Blob)
    method : str
        "{UPLOAD}" | "{DOWNLOAD}"
    global_timeout : int=600
        timeout to wait for all operations to complete
    timeout : tuple, default=(5, 120)
        timeout forwarded to Blob method
    retries : int, default=5
        how many times to attempt the method
    threads : int, default=4
        how many threads to use in the thread pool
    """
    if not len(path_pairs):
        LOG.warning("Gsutil process_many was called with no files")
        return
    if method not in BLOB_ACTIONS:
        raise Exception(f"{method=} not in {BLOB_ACTIONS=}")
    threads = min(threads, len(path_pairs))
    LOG.info(
        f"Gsutil process_many={method}, threads={threads}, items={len(path_pairs)}"
    )
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [
            executor.submit(
                retry_blob_method,
                blob,
                method,
                path,
                retries=retries,
                timeout=timeout,
            )
            for path, blob in path_pairs
        ]
        wait(
            futures,
            timeout=global_timeout,
            return_when=ALL_COMPLETED,
        )
        for future in futures:
            if future.exception():
                raise future.exception()  # type: ignore


def method_context(
    sources: List[Path],
    targets: List[Path],
) -> Tuple[List[Path], List[Path], str]:
    f"""Determine by inspection of source and target
    whether the action to perform is an upload or
    a download. Assumes all paths in sources and
    targets are self-consistent, i.e., all remote
    or local paths, respectively.

    Parameters
    ----------
    source : List[Path]
        source paths
    target : List[Path]
        target paths

    Returns
    -------
    tup : tuple
        (local, remote, "{UPLOAD}" | "{DOWNLOAD}")
    """
    source = sources[0]
    target = targets[0]
    source_local = paths.is_local_path(source.parent)
    target_local = paths.is_local_path(target.parent)
    if source_local and target_local:
        raise Exception(f"{source=} and {target=} are both local paths!")
    if not source_local and not target_local:
        raise Exception(f"{source=} and {target=} are both remote paths!")
    if source_local:
        return sources, targets, UPLOAD
    return targets, sources, DOWNLOAD


def make_path_pairs(
    local_paths: List[Path],
    remote_keys: Union[List[str], List[Path], List[Blob]],
    bucket: Bucket,
) -> List[Tuple[str, Blob]]:
    """Create pairs of (str, Blob) for threaded processing.

    Parameters
    ----------
    local_paths : List[Path]
        local file paths
    remote_keys : List[str] | List[Path] | List[Blob]
        remote key locations
    bucket : Bucket
        client provider

    Returns
    -------
    pairs : List[Tuple[str, Blob]]
        destinations and blobs
    """
    if not len(local_paths) or not len(remote_keys):
        return []

    def get_blob(blob: Union[str, Path, Blob]) -> Blob:
        if isinstance(blob, Path):
            return bucket.blob(blob.as_posix())
        if isinstance(blob, str):
            return bucket.blob(blob)
        return blob

    return [
        (
            path.as_posix(),
            get_blob(key),
        )
        for path, key in zip(local_paths, remote_keys)
    ]


class Gsutil:
    """Attempt to achieve the efficiency of "gsutil -m" commands
    using the python storage API. This involves using a simple
    ThreadPoolExecutor to circumvent blocking network I/O and
    retain an ergonomic high-level user interface.
    """

    DOWNLOAD: ClassVar[str] = DOWNLOAD
    UPLOAD: ClassVar[str] = UPLOAD
    REMOTE_COPY: ClassVar[str] = REMOTE_COPY

    @staticmethod
    def process_many(
        *args: List[Tuple[str, Blob]], **kwargs: int | tuple[int, int]
    ) -> None:
        process_many(*args, **kwargs)
        return None

    @staticmethod
    def bucket_process_many(
        *args: list[Blob] | str, **kwargs: int | tuple[int, int]
    ) -> None:
        bucket_process_many(*args, **kwargs)
        return None

    def __repr__(self) -> str:
        name = self.__class__.__name__
        if self._name is not None:
            return f"{name}({self._name})"
        return f"{name}()"

    def __init__(self, client: Optional[Client] = None) -> None:
        self._client = client
        self._bucket = None
        self._name = None

    def get_bucket(
        self, value: paths._StrPath, bucket: Optional[Bucket] = None
    ) -> Bucket:
        if bucket is not None:
            return bucket
        name = value.parts[1] if isinstance(value, Path) else value
        if name == self._name and self._bucket is not None:
            return self._bucket
        if self._client is None:
            self._client = Client.create_anonymous_client()
        try:
            self._bucket = self._client.bucket(name)
        except Exception:
            LOG.error(f"could not create bucket from {name}!")
            raise
        self._name = name
        return self._bucket

    def cp_paired_paths(
        self,
        sources: paths._ListStrPath,
        targets: paths._ListStrPath,
        bucket: Optional[Bucket] = None,
        threads: Optional[int] = None,
    ) -> None:
        """Copy a collection of sources to collection of targets.

        Parameters
        ----------
        sources : List[str | Path]
            source files to copy
        targets : List[str | Path]
            target destinations
        bucket : Any
            optionally pass existing client
        threads : int, default=None
            explicit number of threads passed to ThreadPoolExecutor
        """
        if not isinstance(targets, list):
            raise ValueError(
                f"Only support target as list of str or Paths, got {targets=}"
            )
        try:
            local_paths, remote_paths, method = method_context(
                [Path(path) for path in sources],
                [Path(path) for path in targets],
            )
        except IndexError:
            LOG.warning("Gsutil.cp_paths received no files")
            return
        bucket = self.get_bucket(remote_paths[0], bucket)
        if method == UPLOAD:
            raise ValueError(f"Uploading paired paths not supported! See cp_paths")

        remote_keys = [paths.remote_key(path) for path in remote_paths]
        pairs = make_path_pairs(local_paths, remote_keys, bucket)
        process_many(pairs, method, threads=threads or get_cpu_limit())

    def cp_paths(
        self,
        sources: paths._ListStrPath,
        target: paths._StrPath,
        anchor: paths._StrPath,
        bucket: Optional[Bucket] = None,
        threads: Optional[int] = None,
    ) -> None:
        """Copy a collection of sources to target.
        target must be a single path.

        Parameters
        ----------
        sources : List[str | Path]
            source files to copy
        target : str | Path
            target destination
        anchor : str | Path
            make source paths relative to anchor
        bucket : Any
            optionally pass existing client
        threads : int, default=None
            explicit number of threads passed to ThreadPoolExecutor
        """
        if isinstance(target, list):
            raise ValueError(
                f"Only support target as single str or Path, got {target=}"
            )
        try:
            local_paths, remote_paths, method = method_context(
                [Path(path) for path in sources],
                [Path(path) for path in [target]],
            )
        except IndexError:
            LOG.warning("Gsutil.cp_paths received no files")
            return
        anchor = Path(anchor)
        bucket = self.get_bucket(remote_paths[0], bucket)
        if method == UPLOAD:
            remote = paths.strip_glob(remote_paths[0])
            rkey = paths.remote_key(remote)
            remote_keys = [rkey / path.relative_to(anchor) for path in local_paths]
        else:
            local = paths.strip_glob(local_paths[0])
            local_paths = [local / path.relative_to(anchor) for path in remote_paths]
            remote_keys = [paths.remote_key(path) for path in remote_paths]
        pairs = make_path_pairs(local_paths, remote_keys, bucket)
        process_many(pairs, method, threads=threads or get_cpu_limit())

    def cp_dir(
        self,
        source: paths._StrPath,
        target: paths._StrPath,
        bucket: Optional[Any] = None,
        threads: Optional[int] = None,
    ) -> None:
        """Copy an entire source directory to target. Assumes
        everything processed will be relative to source and target.

        Parameters
        ----------
        source : str | Path
            source directory to copy
        target : str | Path
            target destination
        bucket : Any
            optionally pass existing client
        threads : int, default=None
            explicit number of threads passed to ThreadPoolExecutor

        Examples
        --------
        For downloading flat directories like:

        .. code-block:: text

            gs://{bucket}/{flat_path}
            ├── data.csv
            ├── other_data.csv
            └── ...

        .. code-block:: python

            >>> Gsutil().cp_dir(
            ...     f"gs://{bucket}/{flat_path}",
            ...     "/path/to/local_dir",
            ... ) # doctest: +SKIP

        .. code-block:: text

            # should result in
            /path/to/local_dir
            ├── data.csv
            ├── other_data.csv
            └── ...

        For downloading nested directories like:

        .. code-block:: text

            gs://{bucket}/{nested_path}
            ├── {nested_path}-0
            │   └── item.json
            ├── {nested_path}-1
            │   └── item.json
            └── ...

        .. code-block:: python

            >>> Gsutil().cp_dir(
            ...     f"gs://{bucket}/{nested_path}",
            ...     "/path/to/local_dir",
            ... ) # doctest: +SKIP

        .. code-block:: text

            # should result in
            /path/to/local_dir
            ├── {nested_path}-0
            │   └── item.json
            ├── {nested_path}-1
            │   └── item.json
            └── ...

        For uploading flat directories like:

        .. code-block:: text

            /path/to/local_dir
            ├── data.csv
            ├── other_data.csv
            └── ...

        .. code-block:: python

            >>> Gsutil().cp_dir(
            ...     "/path/to/local_dir",
            ...     f"gs://{bucket}/{target_path}",
            ... ) # doctest: +SKIP

        .. code-block:: text

            # should result in
            gs://{bucket}/{target_path}
            ├── data.csv
            ├── other_data.csv
            └── ...

        For uploading nested directories like:

        .. code-block:: text

            /path/to/local_dir
            ├── {nested_path}-0
            │   └── item.json
            ├── {nested_path}-1
            │   └── item.json
            └── ...

        .. code-block:: python

            >>> Gsutil().cp_dir(
            ...     "/path/to/local_dir",
            ...     f"gs://{bucket}/{nested_path}",
            ... ) # doctest: +SKIP

        .. code-block:: text

            # should result in
            gs://{bucket}/{nested_path}
            ├── {nested_path}-0
            │   └── item.json
            ├── {nested_path}-1
            │   └── item.json
            └── ...

        Advanced Examples
        -----------------
        For downloading files using partial matches:

        .. code-block:: text

            gs://{bucket}/{prefix}/{partial_match}
            ├── {partial_match}-0.csv
            ├── {partial_match}-1.csv
            └── ...

        .. code-block:: python

            >>> Gsutil().cp_dir(
            ...     "gs://{bucket}/{prefix}/{partial_match}",
            ...     "test_dir/subdir",
            ... ) # doctest: +SKIP

        .. code-block:: text

            # should result in
            test_dir
            └── subdir
                ├── {partial_match}-0.csv
                ├── {partial_match}-1.csv
                └── ...

        For uploading files using partial matches:

        .. code-block:: text

            test_dir
            └── subdir
                ├── {partial_match}-0.csv
                ├── {partial_match}-1.csv
                └── ...

        .. code-block:: python

            >>> Gsutil().cp_dir(
            ...     "test_dir/subdir/{partial_match}",
            ...     "gs://{bucket}/{prefix}/test_upload",
            ... ) # doctest: +SKIP

        .. code-block:: text

            # should result in
            gs://{bucket}/{prefix}/test_upload
            ├── {partial_match}-0.csv
            ├── {partial_match}-1.csv
            └── ...

        """
        if "*" in str(source) or "*" in str(target):
            raise Exception("Gsutil will not resolve glob expressions")
        [local], [remote], method = method_context([Path(source)], [Path(target)])
        local = paths.strip_glob(local)
        remote = paths.strip_glob(remote)
        rkey = paths.remote_key(remote)
        bucket = self.get_bucket(remote, bucket)
        if method == UPLOAD:
            local_paths = paths.expand_local_path(local)
            try:
                # when a clean directory is provided
                remote_keys = [rkey / path.relative_to(local) for path in local_paths]
            except ValueError:
                # when trailing filter characters are provided
                remote_keys = [
                    rkey / path.relative_to(local.parent) for path in local_paths
                ]
        else:
            local_paths = []
            # assume a clean directory was provided
            remote_keys = list(bucket.list_blobs(prefix=f"{rkey}/"))
            if not len(remote_keys):
                # but allow for trailing filter characters
                remote_keys = list(bucket.list_blobs(prefix=f"{rkey}"))
            if len(remote_keys):
                if len(remote_keys) > 1:
                    local.mkdir(exist_ok=True, parents=True)
                try:
                    # the case where clean directory was provided
                    local_paths = [
                        local / Path(blob.name).relative_to(rkey)
                        for blob in remote_keys
                    ]
                except ValueError:
                    rkey = Path(rkey).parent
                    # the case where we have trailing filter characters
                    local_paths = [
                        local / Path(blob.name).relative_to(rkey)
                        for blob in remote_keys
                    ]
        pairs = make_path_pairs(local_paths, remote_keys, bucket)
        process_many(pairs, method, threads=threads or get_cpu_limit())

    def _ls(
        self,
        target: paths._StrPath,
        bucket: Bucket,
        versions: bool = False,
        recursive: bool = True,
    ) -> Iterator[Blob]:
        """Bucket search"""
        remote = Path(target)
        if paths.is_local_path(remote):
            LOG.info(f"Gsutil.ls will skip {remote=}")
            return iter(())
        remote_key = paths.remote_key(remote)
        ## Logic to allow mimicking ls in a non-recursive manner
        # The remote key will *never* have a trailing slash as currently
        # implemented. If we are passing delimiter (non-recursive),
        # we want to append the delimiter iff the remote_key is not a blob.
        extra_kwargs = {}
        if not recursive:
            extra_kwargs["delimiter"] = "/"
            if not bucket.blob(remote_key).exists():
                remote_key = f"{remote_key}{extra_kwargs['delimiter']}"
        # Call to list_blobs
        blob_iter: Iterator[Blob] = bucket.list_blobs(
            prefix=remote_key,
            versions=versions,
            **extra_kwargs,
        )
        return blob_iter

    def ls(
        self,
        target: paths._StrPath,
        bucket: Optional[Any] = None,
        recursive: bool = True,
    ) -> List[Path]:
        """List the contents of a remote directory, returning full paths
        to files including gs://{bucket_name}.

        Parameters
        ----------
        target : str | Path
            root path for remote files to list
        bucket : Any
            optionally pass existing client
        recursive : bool
            recursively list files in sub-directories

        Returns
        -------
        paths : List[Path]
            full remote paths to listed files
        """
        bucket = self.get_bucket(Path(target), bucket)
        blobs = self._ls(target, bucket, recursive=recursive)
        return [Path(f"gs://{bucket.name}") / blob.name for blob in blobs]

    def ls_blobs(
        self,
        target: paths._StrPath,
        bucket: Optional[Any] = None,
        versions: bool = False,
    ) -> List[Blob]:
        """Remote ls returning blob objects instead of paths

        Parameters
        ----------
        target : str | Path
            root path for remote files to list
        bucket : Any
            optionally pass existing client
        versions : bool, default=False
            if True, include blob versions

        Returns
        -------
        paths : List[Path]
            full remote paths to listed files
        """
        bucket = self.get_bucket(Path(target), bucket)
        return list(self._ls(target, bucket, versions))

    def remote_cp(
        self,
        sources: List[paths._StrPath],
        targets: List[paths._StrPath],
        bucket: Optional[Any] = None,
    ) -> None:
        """Transfer remote files from one location to another without
        downloading them to a local context first. Assumes all source
        files reside in the same bucket, and all target files reside in
        the same bucket, but the source bucket can be different from
        the target bucket.

        Parameters
        ----------
        sources : List[str | Path]
            full remote paths to copy from
        targets : List[str | Path]
            full remote paths to copy to
        bucket : Any
            optionally pass existing bucket (assumes source and target
            buckets are the same bucket)

        """
        sources = [Path(source) for source in sources]
        targets = [Path(target) for target in targets]
        if any((paths.is_local_path(path) for path in sources)) or any(
            (paths.is_local_path(path) for path in targets)
        ):
            LOG.info("Gsutil.remote_cp is only meant for remote->remote copy")
            return

        source_bucket = self.get_bucket(sources[0], bucket)
        target_bucket = self.get_bucket(targets[0], bucket)
        bucket_process_many(
            [source_bucket.blob(paths.remote_key(source)) for source in sources],
            [target_bucket.blob(paths.remote_key(target)) for target in targets],
            REMOTE_COPY,
        )


def gcs_read_dataframe(
    gcs_uri: str | Path,
    fs: gcsfs.GCSFileSystem | None = None,
    token: str | None = "anon",
) -> pd.DataFrame:
    """Read remote files directly into `pandas.DataFrame` with anonymous client credentials.

    If the gcsfs.GCSFileSystem object is not provided, one will be created with
    token set to `anon` (no authentication is performed), and you can only
    access data which is accessible to IAM principal allUsers.

    Parameters
    ----------
    gcs_uri : str | Path
        full remote paths to read.
        Must end with .csv, .csv.gz, or .parquet extension.
    fs : gcsfs.GCSFileSystem
        optionally pass an existing authenticated GCSFileSystem object to use.
    token : str | None
        optionally pass a token type to use for authenticating requests.
        Default is "anon" (only public objects). If an authentication error is raised,
        the method is retried with token=None to attempt to infer credentials in the following order:
        gcloud CLI default, gcsfs cached token, google compute metadata service, anonymous.

    """
    # Verify that the filepath is a GCS uri, if its not don't use gcsfs
    remote = str(gcs_uri).startswith("gs://")
    if fs is None:
        fs = gcsfs.GCSFileSystem(token=token)
    file_ext = Path(gcs_uri).suffix
    if file_ext in [".csv", ".gz"]:
        reader = pd.read_csv
    elif file_ext == ".parquet":
        reader = pd.read_parquet
    if remote:
        try:
            with fs.open(gcs_uri, "rb") as f:
                if file_ext == ".gz":
                    g = GzipFile(fileobj=f)  # Decompress data with gzip
                    data = reader(g)
                else:
                    data = reader(f)
        except gcsfs.retry.HttpError:
            if token is None:
                raise
            else:
                LOG.warning(
                    f"Failed to read {gcs_uri} with token=`anon`, retrying with token=None."
                )
                return gcs_read_dataframe(gcs_uri, token=None)
    else:
        # It's a local file
        data = reader(gcs_uri)
    return data
