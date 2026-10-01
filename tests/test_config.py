import io
import os
import tempfile
import yaml
import pytest
import re

from eregion.configs import DetectorConfig


def valid_detconfig_dict():
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
    cfg = DetectorConfig(config_input=valid_detconfig_dict())
    assert isinstance(cfg.config, dict) and cfg.config["detector_type"] == "CCD"


def test_detectorconfig_init_with_yaml_file_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "config.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(valid_detconfig_dict(), f)
        cfg = DetectorConfig(config_input=path)
        assert cfg.config["objects"][0]["name"] == "D1"


def test_detectorconfig_init_with_yaml_string():
    yaml_str = yaml.safe_dump(valid_detconfig_dict())
    cfg = DetectorConfig(config_input=yaml_str)
    assert cfg.config["detector_output_class"] == "CCDOutput"


def test_validate_config_missing_top_level_key_raises():
    bad = valid_detconfig_dict()
    bad.pop("detector_type")
    with pytest.raises(KeyError):
        cfg = DetectorConfig(config_input=bad)


def test_validate_config_missing_object_key_raises():
    bad = valid_detconfig_dict()
    bad["objects"][0].pop("class")
    with pytest.raises(KeyError):
        cfg = DetectorConfig(config_input=bad)


def test_validate_config_missing_property_key_raises():
    bad = valid_detconfig_dict()
    bad["objects"][0]["properties"].pop("pixel_size")
    with pytest.raises(KeyError):
        cfg = DetectorConfig(config_input=bad)


def interpolation_detconfig_dict():
    return {
        "detector_type": "CCD",
        "detector_output_class": "CCDOutput",
        "objects": [
            {
                "name": "D1",
                "class": "Detector",
                "properties": {"x_size": 10, "y_size": 12, "pixel_size": 0.01},
                "outputs": [],
                "metadata": {
                    "flag": "${runtime.debug}",
                    "path": "/data/${detector.data_dir}/raw",
                    "count": "${runtime.count}",
                    "nested": [
                        "prefix-${runtime.suffix}",
                        {"inner": "${runtime.inner}"},
                    ],
                    "escaped": r"\${literal}",
                    "defaulted": "${runtime.missing:/tmp/default}",
                },
            }
        ],
    }


def interpolation_runtime_variables(**overrides):
    runtime = {"debug": True, "count": 7, "suffix": "end", "inner": "value"}
    runtime.update(overrides)
    return {"runtime": runtime, "detector": {"data_dir": "science"}}


def test_yaml_interpolation_simple_replace_defaults_and_typed_values():
    cfg = DetectorConfig(
        config_input=interpolation_detconfig_dict(),
        runtime_variables=interpolation_runtime_variables(),
    )

    metadata = cfg.config["objects"][0]["metadata"]
    assert metadata["flag"] is True
    assert metadata["path"] == "/data/science/raw"
    assert metadata["count"] == 7
    assert metadata["nested"] == ["prefix-end", {"inner": "value"}]
    assert metadata["defaulted"] == "/tmp/default"


def test_yaml_interpolation_escaping():
    cfg = DetectorConfig(
        config_input=interpolation_detconfig_dict(),
        runtime_variables=interpolation_runtime_variables(),
    )

    assert cfg.config["objects"][0]["metadata"]["escaped"] == "${literal}"


def test_yaml_interpolation_missing_var_raises():
    bad = interpolation_detconfig_dict()
    bad["objects"][0]["metadata"]["path"] = "${runtime.missing}"
    with pytest.raises(ValueError, match=re.escape("Unknown interpolation variable 'runtime.missing'")):
        DetectorConfig(config_input=bad, runtime_variables=interpolation_runtime_variables())


def test_yaml_interpolation_cycle_raises():
    bad = interpolation_detconfig_dict()
    with pytest.raises(ValueError, match="Interpolation cycle detected"):
        DetectorConfig(
            config_input=bad,
            runtime_variables=interpolation_runtime_variables(
                debug="${runtime.loop}", loop="${runtime.debug}"
            ),
        )


def test_yaml_interpolation_malformed_token_raises():
    bad = interpolation_detconfig_dict()
    bad["objects"][0]["metadata"]["path"] = "${runtime.data_dir"
    with pytest.raises(ValueError, match="Malformed interpolation token"):
        DetectorConfig(config_input=bad, runtime_variables=interpolation_runtime_variables())


def test_yaml_interpolation_env_vars_only_when_enabled(monkeypatch):
    bad = interpolation_detconfig_dict()
    bad["objects"][0]["metadata"]["path"] = "${EREGION_DATA_DIR}"
    monkeypatch.setenv("EREGION_DATA_DIR", "from-env")

    with pytest.raises(ValueError, match="Unknown interpolation variable 'EREGION_DATA_DIR'"):
        DetectorConfig(config_input=bad, runtime_variables=interpolation_runtime_variables())

    cfg = DetectorConfig(
        config_input=bad,
        runtime_variables=interpolation_runtime_variables(),
        enable_env_vars=True,
    )
    assert cfg.config["objects"][0]["metadata"]["path"] == "from-env"
