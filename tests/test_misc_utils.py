from pathlib import Path
import logging
import uuid

import pytest

from eregion.utils.misc_utils import configure_logger, load_class


def test_configure_logger_sets_expected_defaults():
    name = f"test-logger-{uuid.uuid4()}"
    logger = configure_logger(name)

    assert logger.name == name
    assert logger.level == logging.INFO
    assert logger.propagate is False
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_load_class_imports_known_class():
    cls = load_class("pathlib.Path")
    assert cls is Path


def test_load_class_raises_for_missing_attribute():
    with pytest.raises(AttributeError):
        load_class("pathlib.NotARealClass")


def test_load_class_imports_datamodels_class_like_detector_config_usage():
    cls = load_class("datamodels.CCDOutput")
    assert cls.__name__ == "CCDOutput"
    assert cls.__module__.startswith("datamodels")
