"""
Tests for the `eregion` CLI.
"""
import os
import tempfile

import yaml
from typer.testing import CliRunner

from eregion.cli.main import app
from eregion.cli._common import parse_var_options

runner = CliRunner()


def detector_config_dict(x_size: int = 2188):
    return {
        "description": "test detector",
        "detector_type": "CCD",
        "detector_output_class": "CCDOutput",
        "objects": [
            {
                "name": "det_1",
                "class": "DetImage",
                "properties": {"x_size": x_size, "y_size": 4125, "pixel_size": 0.015},
                "outputs": [{"id": "chan_1", "ext_id": 1}, {"id": "chan_2", "ext_id": 2}],
            }
        ],
    }


def write_yaml(tmp_path, data):
    path = os.path.join(tmp_path, "detector.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def test_parse_var_options_empty():
    assert parse_var_options(None) == {}
    assert parse_var_options([]) == {}


def test_parse_var_options_parses_key_value_pairs():
    assert parse_var_options(["a=1", "b=two"]) == {"a": "1", "b": "two"}


def test_version_flag_reports_the_installed_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "eregion" in result.output


def test_validate_prints_detector_structure_and_reads_no_data():
    with tempfile.TemporaryDirectory() as td:
        path = write_yaml(td, detector_config_dict())
        result = runner.invoke(app, ["validate", path])

    assert result.exit_code == 0
    assert "Config OK: 1 detector object(s) defined." in result.output
    assert "detector_type: CCD" in result.output
    assert "det_1 (DetImage)" in result.output
    assert "2188 x 4125 px" in result.output
    assert "'chan_1', 'chan_2'" in result.output
    assert "No image data was read." in result.output


def test_validate_missing_config_file_is_a_usage_error():
    result = runner.invoke(app, ["validate", "/no/such/file.yaml"])
    assert result.exit_code != 0


def test_validate_invalid_config_exits_non_zero_with_an_error():
    with tempfile.TemporaryDirectory() as td:
        bad = detector_config_dict()
        bad.pop("detector_type")
        path = write_yaml(td, bad)
        result = runner.invoke(app, ["validate", path])

    assert result.exit_code == 1
    assert "Failed to load detector config" in result.output


def test_validate_accepts_runtime_variable_overrides():
    with tempfile.TemporaryDirectory() as td:
        data = detector_config_dict()
        data["objects"][0]["properties"]["x_size"] = "${width}"
        path = write_yaml(td, data)
        result = runner.invoke(app, ["validate", path, "--var", "width=1024"])

    assert result.exit_code == 0
    assert "1024 x 4125 px" in result.output


def test_validate_env_fallback_requires_the_env_flag(monkeypatch):
    monkeypatch.setenv("EREGION_TEST_WIDTH", "512")
    with tempfile.TemporaryDirectory() as td:
        data = detector_config_dict()
        data["objects"][0]["properties"]["x_size"] = "${EREGION_TEST_WIDTH}"
        path = write_yaml(td, data)

        without_flag = runner.invoke(app, ["validate", path])
        with_flag = runner.invoke(app, ["validate", path, "--env"])

    assert without_flag.exit_code == 1
    assert with_flag.exit_code == 0
    assert "512 x 4125 px" in with_flag.output
