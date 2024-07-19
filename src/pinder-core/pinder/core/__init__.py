"""Namespace package for pinder-core."""

from pinder.core._version import _get_version

__version__ = _get_version()

from pinder.core.utils import log
from pinder.core.index.utils import (
    get_index,
    get_metadata,
    get_pinder_bucket_root,
    get_pinder_location,
    get_supplementary_data,
    download_dataset,
    SupplementaryData,
)
from pinder.core.index.system import PinderSystem
from pinder.core.loader.loader import get_systems, PinderLoader


__all__ = [
    "download_dataset",
    "PinderLoader",
    "PinderSystem",
    "get_index",
    "get_metadata",
    "get_systems",
    "get_pinder_bucket_root",
    "get_pinder_location",
    "get_supplementary_data",
    "log",
    "SupplementaryData",
]
