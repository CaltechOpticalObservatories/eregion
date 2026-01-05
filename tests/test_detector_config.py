import io
import os
import tempfile
import yaml
import pytest

from datamodels.detector_config import DetectorConfig


def valid_config_dict():
    return {
        "detector_type": "CCD",
        "detector_output_class": "CCDOutput",
        "objects": [
            {
                "name": "D1",
                "class": "Detector",
                "properties": {"x_size": 10, "y_size": 12, "pixel_size": 0.01},
                "outputs": [],
            }
        ],
    }


def test_yaml_slice_constructor_roundtrip():
    text = "s: !slice [1, 5, 2]"
    data = yaml.load(io.StringIO(text), Loader=yaml.FullLoader)
    assert isinstance(data["s"], slice)
    assert data["s"].start == 1 and data["s"].stop == 5 and data["s"].step == 2


def test_detectorconfig_init_with_dict_validates():
    cfg = DetectorConfig(config_input=valid_config_dict())
    assert isinstance(cfg.config, dict) and cfg.config["detector_type"] == "CCD"


def test_detectorconfig_init_with_yaml_file_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(valid_config_dict(), f)
        cfg = DetectorConfig(config_input=path)
        assert cfg.config["objects"][0]["name"] == "D1"


def test_detectorconfig_init_with_yaml_string():
    yaml_str = yaml.safe_dump(valid_config_dict())
    cfg = DetectorConfig(config_input=yaml_str)
    assert cfg.config["detector_output_class"] == "CCDOutput"


def test_detectorconfig_init_with_fits_path_raises_notimplemented():
    with pytest.raises(NotImplementedError):
        DetectorConfig(fits_path="example.fits")


def test_load_config_from_stream_success():
    cfg = DetectorConfig(config_input={})  # will be replaced by load_config
    stream = io.StringIO(yaml.safe_dump(valid_config_dict()))
    cfg.load_config(stream)
    assert cfg.config["objects"][0]["properties"]["x_size"] == 10


def test_load_config_invalid_yaml_raises_valueerror():
    cfg = DetectorConfig(config_input={})
    with pytest.raises(ValueError):
        cfg.load_config(io.StringIO(":: this is not valid yaml ::"))


def test_validate_config_missing_top_level_key_raises():
    bad = valid_config_dict()
    bad.pop("detector_type")
    cfg = DetectorConfig(config_input=bad)
    with pytest.raises(ValueError):
        cfg.validate_config()


def test_validate_config_missing_object_key_raises():
    bad = valid_config_dict()
    bad["objects"][0].pop("class")
    cfg = DetectorConfig(config_input=bad)
    with pytest.raises(ValueError):
        cfg.validate_config()


def test_validate_config_missing_property_key_raises():
    bad = valid_config_dict()
    bad["objects"][0]["properties"].pop("pixel_size")
    cfg = DetectorConfig(config_input=bad)
    with pytest.raises(ValueError):
        cfg.validate_config()