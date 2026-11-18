import io
import os
import tempfile
import yaml
import pytest
import re

from eregion.configs import DetectorConfig, FlowConfig


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


def valid_flowconfig_dict():
    return {
        "flows": [
            {
                "name": "calib_flow",
                "lazy": False,
                "nodes": [
                    {
                        "name": "image_creator",
                        "task": "tasks.imagegen.ImageCreator",
                    }
                ],
            }
        ]
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


def test_flowconfig_init_with_dict_validates():
    cfg = FlowConfig(config_input=valid_flowconfig_dict())
    assert isinstance(cfg.config, dict) and cfg.config["flows"][0]["name"] == "calib_flow"


def test_flowconfig_init_with_yaml_file_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "flow.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(valid_flowconfig_dict(), f)
        cfg = FlowConfig(config_input=path)
        assert cfg.config["flows"][0]["nodes"][0]["task"] == "tasks.imagegen.ImageCreator"


def test_flowconfig_init_with_yaml_string():
    yaml_str = yaml.safe_dump(valid_flowconfig_dict())
    cfg = FlowConfig(config_input=yaml_str)
    assert cfg.config["flows"][0]["lazy"] is False


def test_flowconfig_missing_top_level_key_raises():
    bad = valid_flowconfig_dict()
    bad.pop("flows")
    with pytest.raises(KeyError):
        cfg = FlowConfig(config_input=bad)


def test_flowconfig_missing_flow_key_raises():
    bad = valid_flowconfig_dict()
    bad["flows"][0].pop("nodes")
    with pytest.raises(KeyError):
        cfg = FlowConfig(config_input=bad)


def test_flowconfig_lazy_flow_missing_source_raises():
    bad = valid_flowconfig_dict()
    bad["flows"][0]["lazy"] = True
    with pytest.raises(KeyError, match="Missing required key 'source'"):
        cfg = FlowConfig(config_input=bad)


def test_flowconfig_accepts_the_deprecated_pipelines_key():
    old_style = valid_flowconfig_dict()
    old_style["pipelines"] = old_style.pop("flows")
    cfg = FlowConfig(config_input=old_style)
    assert "pipelines" not in cfg.config
    assert cfg.config["flows"] == valid_flowconfig_dict()["flows"]


def test_flowconfig_rejects_both_flows_and_pipelines():
    bad = valid_flowconfig_dict()
    bad["pipelines"] = bad["flows"]
    with pytest.raises(ValueError, match="both 'flows' and the deprecated 'pipelines'"):
        FlowConfig(config_input=bad)


def interpolation_flowconfig_dict():
    return {
        "debug": "${runtime.debug}",
        "flows": [
            {
                "name": "calib_flow",
                "lazy": False,
                "nodes": [
                    {
                        "name": "image_creator",
                        "task": "tasks.imagegen.ImageCreator",
                        "metadata": {
                            "path": "/data/${flow.data_dir}/raw",
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
        ],
    }


def test_yaml_interpolation_simple_replace_defaults_and_typed_values():
    cfg = FlowConfig(
        config_input=interpolation_flowconfig_dict(),
        runtime_variables={
            "runtime": {
                "debug": True,
                "count": 7,
                "suffix": "end",
                "inner": "value",
            },
            "flow": {"data_dir": "science"},
        },
    )

    node = cfg.config["flows"][0]["nodes"][0]
    assert cfg.config["debug"] is True
    assert cfg.config["flows"][0]["lazy"] is False
    assert node["metadata"]["path"] == "/data/science/raw"
    assert node["metadata"]["count"] == 7
    assert node["metadata"]["nested"] == ["prefix-end", {"inner": "value"}]
    assert node["metadata"]["defaulted"] == "/tmp/default"


def test_yaml_interpolation_escaping():
    cfg = FlowConfig(
        config_input=interpolation_flowconfig_dict(),
        runtime_variables={
            "runtime": {
                "debug": False,
                "count": 1,
                "suffix": "end",
                "inner": "value",
            },
            "flow": {"data_dir": "science"},
        },
    )

    assert cfg.config["flows"][0]["nodes"][0]["metadata"]["escaped"] == "${literal}"


def test_yaml_interpolation_missing_var_raises():
    bad = interpolation_flowconfig_dict()
    bad["flows"][0]["nodes"][0]["metadata"]["path"] = "${runtime.missing}"
    with pytest.raises(ValueError, match=re.escape("Unknown interpolation variable 'runtime.missing'")):
        FlowConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": False,
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                },
                "flow": {"data_dir": "science"},
            },
        )


def test_yaml_interpolation_cycle_raises():
    bad = interpolation_flowconfig_dict()
    with pytest.raises(ValueError, match="Interpolation cycle detected"):
        FlowConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": "${runtime.loop}",
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                    "loop": "${runtime.debug}",
                },
                "flow": {"data_dir": "science"},
            },
        )


def test_yaml_interpolation_malformed_token_raises():
    bad = interpolation_flowconfig_dict()
    bad["flows"][0]["nodes"][0]["metadata"]["path"] = "${runtime.data_dir"
    with pytest.raises(ValueError, match="Malformed interpolation token"):
        FlowConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": False,
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                },
                "flow": {"data_dir": "science"},
            },
        )


def test_yaml_interpolation_env_vars_only_when_enabled(monkeypatch):
    bad = interpolation_flowconfig_dict()
    bad["flows"][0]["nodes"][0]["metadata"]["path"] = "${FLOW_DATA_DIR}"
    monkeypatch.setenv("FLOW_DATA_DIR", "from-env")

    with pytest.raises(ValueError, match="Unknown interpolation variable 'FLOW_DATA_DIR'"):
        FlowConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": False,
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                },
                "flow": {"data_dir": "science"},
            },
        )

    cfg = FlowConfig(
        config_input=bad,
        runtime_variables={
            "runtime": {
                "debug": False,
                "lazy": False,
                "count": 1,
                "suffix": "end",
                "inner": "value",
            },
            "flow": {"data_dir": "science"},
        },
        enable_env_vars=True,
    )
    assert cfg.config["flows"][0]["nodes"][0]["metadata"]["path"] == "from-env"
