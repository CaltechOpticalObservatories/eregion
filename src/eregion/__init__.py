from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("eregion")
except PackageNotFoundError:
    try:
        from ._version import version as vstr
        __version__ = vstr
    except ModuleNotFoundError as err:
        __version__ =  "unknown"
