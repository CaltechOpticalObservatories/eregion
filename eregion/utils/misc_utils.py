import logging
import importlib


def configure_logger(name):
    """
    Configure a logger
    """
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def load_class(path: str):
    """
    Dynamically load a class from a given path. Has to be in eregion package, or importable from the current environment.
    :param path: str
        The full path to the class, e.g. "module.submodule.ClassName".
    :return: class
        The loaded class call.
    """
    module, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), cls)