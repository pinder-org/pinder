import builtins
import importlib.metadata
import re
import sys

import pytest


@pytest.fixture
def hide_packages_from_import(monkeypatch):
    """
    pytest fixture factory to hide packages from `import`, making the test
    think that they are not installed.

    This allows us to get coverage over the branch of _version.py where
    importing setuptools_scm throws ImportError.

    https://stackoverflow.com/a/60229056

    This fixture factory returns a function with signature:

        hide_packages_from_import(*package_names: str) -> None

    To use it,

        def my_test(hide_packages_from_import):
            hide_packages_from_import("package_1", "package_2")
            import package_1  # will raise ImportError!

    """

    def _hide_packages_from_import(*package_names):
        import_orig = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name in package_names:
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)

    return _hide_packages_from_import


@pytest.fixture
def hide_packages_from_importlib(monkeypatch):
    """
    pytest fixture factory to hide packages from `importlib.metadata.version`,
    making importlib think that they are not installed.

    This allows us to get coverage over the branch of _version.py where
    checking the package version via importlib throws PackageNotFoundError.

    https://stackoverflow.com/a/60229056

    This fixture factory returns a function with signature:

        hide_packages_from_importlib(*package_names: str) -> None

    To use it,

        from importlib.metadata import version

        def my_test(hide_packages_from_importlib):
            hide_packages_from_importlib("package_1", "package_2")
            version("package_1") # will raise PackageNotFoundError!

    """

    def _hide_packages_from_importlib(*package_names):
        importlib_version_orig = importlib.metadata.version

        def mocked_importlib_version(name, *args, **kwargs):
            if name in package_names:
                raise importlib.metadata.PackageNotFoundError()
            return importlib_version_orig(name, *args, **kwargs)

        monkeypatch.setattr(importlib.metadata, "version", mocked_importlib_version)

    return _hide_packages_from_importlib


@pytest.fixture
def version_regex():
    """pytest fixture that provides a simple version regex."""
    return re.compile(r"(\d+\.)?\d+\.\d+.*")


@pytest.fixture
def scrub_pinder():
    def inner():
        sys.modules.pop("pinder", None)
        sys.modules.pop("pinder.core", None)

    return inner


def test_import():
    """The package can be imported."""
    import pinder.core as pinder

    assert pinder is not None


def test_get_version_with_setuptools_scm(version_regex):
    """
    The package reports version info as expected, when setuptools_scm is
    installed.
    """
    import setuptools_scm  # noqa: F401,I001 confirms that setuptools_scm is installed
    import pinder.core as pinder

    assert version_regex.match(pinder.__version__)


def test_get_version_without_setuptools_scm(version_regex, hide_packages_from_import):
    """
    The package reports version info as expected, when setuptools_scm is not
    installed.
    """
    hide_packages_from_import("setuptools_scm")
    import pinder.core as pinder

    assert version_regex.match(pinder.__version__)


def test_get_version_without_install(
    hide_packages_from_import, hide_packages_from_importlib
):
    """
    Neither setuptools_scm and nor our package are installed, the version is
    "unknown".
    """
    hide_packages_from_import("setuptools_scm")

    import pinder.core as pinder

    assert pinder.__version__
