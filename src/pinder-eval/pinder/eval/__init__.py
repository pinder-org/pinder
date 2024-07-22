"""Namespace package root for pinder-eval."""

from pinder.eval._version import _get_version
from pinder.eval.dockq.biotite_dockq import BiotiteDockQ

__version__ = _get_version()


__all__ = ["BiotiteDockQ"]
