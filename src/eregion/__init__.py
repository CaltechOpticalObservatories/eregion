"""Eregion: a detector characterization framework."""

__all__ = ["__version__"]

try:
    # Written at build time by setuptools-scm; the source of truth for a
    # built or editable install made from a git checkout.
    from ._version import version as __version__
except ModuleNotFoundError:
    # No generated _version.py (e.g. installed from a wheel someone else
    # built, or from a PyPI sdist), so fall back to installed metadata.
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("eregion")
    except PackageNotFoundError:
        __version__ = "unknown"
