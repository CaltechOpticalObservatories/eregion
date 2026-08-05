import io
import os
import tempfile
import yaml
import pytest
import re

from eregion.configs import DetectorConfig, PipelineConfig


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


def valid_pipelineconfig_dict():
    return {
        "pipelines": [
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


def test_pipelineconfig_init_with_dict_validates():
    cfg = PipelineConfig(config_input=valid_pipelineconfig_dict())
    assert isinstance(cfg.config, dict) and cfg.config["pipelines"][0]["name"] == "calib_flow"


def test_pipelineconfig_init_with_yaml_file_path():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "pipeline.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(valid_pipelineconfig_dict(), f)
        cfg = PipelineConfig(config_input=path)
        assert cfg.config["pipelines"][0]["nodes"][0]["task"] == "tasks.imagegen.ImageCreator"


def test_pipelineconfig_init_with_yaml_string():
    yaml_str = yaml.safe_dump(valid_pipelineconfig_dict())
    cfg = PipelineConfig(config_input=yaml_str)
    assert cfg.config["pipelines"][0]["lazy"] is False


def test_pipelineconfig_missing_top_level_key_raises():
    bad = valid_pipelineconfig_dict()
    bad.pop("pipelines")
    with pytest.raises(KeyError):
        cfg = PipelineConfig(config_input=bad)


def test_pipelineconfig_missing_pipeline_key_raises():
    bad = valid_pipelineconfig_dict()
    bad["pipelines"][0].pop("nodes")
    with pytest.raises(KeyError):
        cfg = PipelineConfig(config_input=bad)


def test_pipelineconfig_lazy_pipeline_missing_source_raises():
    bad = valid_pipelineconfig_dict()
    bad["pipelines"][0]["lazy"] = True
    with pytest.raises(AssertionError, match="Missing required key 'source'"):
        cfg = PipelineConfig(config_input=bad)


def interpolation_pipelineconfig_dict():
    return {
        "debug": "${runtime.debug}",
        "pipelines": [
            {
                "name": "calib_flow",
                "lazy": False,
                "nodes": [
                    {
                        "name": "image_creator",
                        "task": "tasks.imagegen.ImageCreator",
                        "metadata": {
                            "path": "/data/${pipeline.data_dir}/raw",
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
    cfg = PipelineConfig(
        config_input=interpolation_pipelineconfig_dict(),
        runtime_variables={
            "runtime": {
                "debug": True,
                "count": 7,
                "suffix": "end",
                "inner": "value",
            },
            "pipeline": {"data_dir": "science"},
        },
    )

    node = cfg.config["pipelines"][0]["nodes"][0]
    assert cfg.config["debug"] is True
    assert cfg.config["pipelines"][0]["lazy"] is False
    assert node["metadata"]["path"] == "/data/science/raw"
    assert node["metadata"]["count"] == 7
    assert node["metadata"]["nested"] == ["prefix-end", {"inner": "value"}]
    assert node["metadata"]["defaulted"] == "/tmp/default"


def test_yaml_interpolation_escaping():
    cfg = PipelineConfig(
        config_input=interpolation_pipelineconfig_dict(),
        runtime_variables={
            "runtime": {
                "debug": False,
                "count": 1,
                "suffix": "end",
                "inner": "value",
            },
            "pipeline": {"data_dir": "science"},
        },
    )

    assert cfg.config["pipelines"][0]["nodes"][0]["metadata"]["escaped"] == "${literal}"


def test_yaml_interpolation_missing_var_raises():
    bad = interpolation_pipelineconfig_dict()
    bad["pipelines"][0]["nodes"][0]["metadata"]["path"] = "${runtime.missing}"
    with pytest.raises(ValueError, match=re.escape("Unknown interpolation variable 'runtime.missing'")):
        PipelineConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": False,
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                },
                "pipeline": {"data_dir": "science"},
            },
        )


def test_yaml_interpolation_cycle_raises():
    bad = interpolation_pipelineconfig_dict()
    with pytest.raises(ValueError, match="Interpolation cycle detected"):
        PipelineConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": "${runtime.loop}",
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                    "loop": "${runtime.debug}",
                },
                "pipeline": {"data_dir": "science"},
            },
        )


def test_yaml_interpolation_malformed_token_raises():
    bad = interpolation_pipelineconfig_dict()
    bad["pipelines"][0]["nodes"][0]["metadata"]["path"] = "${runtime.data_dir"
    with pytest.raises(ValueError, match="Malformed interpolation token"):
        PipelineConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": False,
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                },
                "pipeline": {"data_dir": "science"},
            },
        )


def test_yaml_interpolation_env_vars_only_when_enabled(monkeypatch):
    bad = interpolation_pipelineconfig_dict()
    bad["pipelines"][0]["nodes"][0]["metadata"]["path"] = "${PIPELINE_DATA_DIR}"
    monkeypatch.setenv("PIPELINE_DATA_DIR", "from-env")

    with pytest.raises(ValueError, match="Unknown interpolation variable 'PIPELINE_DATA_DIR'"):
        PipelineConfig(
            config_input=bad,
            runtime_variables={
                "runtime": {
                    "debug": False,
                    "count": 1,
                    "suffix": "end",
                    "inner": "value",
                },
                "pipeline": {"data_dir": "science"},
            },
        )

    cfg = PipelineConfig(
        config_input=bad,
        runtime_variables={
            "runtime": {
                "debug": False,
                "lazy": False,
                "count": 1,
                "suffix": "end",
                "inner": "value",
            },
            "pipeline": {"data_dir": "science"},
        },
        enable_env_vars=True,
    )
    assert cfg.config["pipelines"][0]["nodes"][0]["metadata"]["path"] == "from-env"
