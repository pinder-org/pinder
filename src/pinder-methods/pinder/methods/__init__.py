"""Namespace package root for pinder-methods."""

from enum import Enum

from pinder.methods._version import _get_version

__version__ = _get_version()


class SUPPORTED_PAIRS(Enum):
    ALL = "all"
    HOLO = "holo"
    APO = "apo"
    PREDICTED = "predicted"


class SUPPORTED_ACTIONS(Enum):
    PREPROCESS = "preprocess"
    TRAIN = "train"
    PREDICT = "predict"


SUPPORTED_METHODS = {
    "af2mm": [
        SUPPORTED_ACTIONS.PREPROCESS,
    ],
    "patchdock": [SUPPORTED_ACTIONS.PREDICT, SUPPORTED_ACTIONS.PREPROCESS],
    "hdock": [SUPPORTED_ACTIONS.PREDICT, SUPPORTED_ACTIONS.PREPROCESS],
    "frodock": [SUPPORTED_ACTIONS.PREDICT, SUPPORTED_ACTIONS.PREPROCESS],
}
