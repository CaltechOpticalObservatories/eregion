import logging
import importlib
from types import ModuleType
from typing import Optional, Type


def configure_logger(name):
    """
    Configure a logger
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_class(path: str, default_module: Optional[ModuleType] = None) -> Type:
    """
    Dynamically load a class from a given path. Has to be in eregion package, or importable from the current environment.
    :param path: str
        The full path to the class, e.g. "module.submodule.ClassName". Paths to eregion's own subpackages
        (e.g. "tasks.imagegen.ImageCreator", "datamodels.CCDOutput") may be given without the "eregion." prefix, for backwards compatibility with pipeline/detector config files that predate eregion's package structure.        Alternatively, if default_module is also provided, the name of a class in that module

    :param default_module: Optional[ModuleType]
        If provided, the class will first be loaded from this module. If it is not found in that module, then the full search will be performed

    :return: class
        The loaded class call.
    """

    if hasattr(default_module, path):
        return getattr(default_module, path)
    else:
        modulename, clsname = path.rsplit(".", 1)
    try:
        # NOTE: add "eregion" here allows for relative imports if needed
        module = importlib.import_module(modulename, "eregion")
        return getattr(module, clsname)
    except ModuleNotFoundError:
        modulename = f"eregion.{modulename}"
        return getattr(importlib.import_module(modulename), clsname)


# A yaml constructor for slice objects
def slice_constructor(loader, node):
    values = loader.construct_sequence(node)
    # slice will be created from a list, e.g., [start, stop, step]
    start, stop, step = None, None, None
    match len(values):
        case 1:
            stop = values[0]
        case 2:
            start, stop = values
        case 3:
            start, stop, step = values
        case _:
            raise ValueError("Invalid number of arguments for slice.")

    if step is None:
        step = -1 if start > stop else 1
    if stop == -1:
        stop = None
    return slice(start, stop, step)
