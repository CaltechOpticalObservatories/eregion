"""
Tests for the `eregion` CLI.
"""
import os
import tempfile

import pytest
import typer
import yaml
from typer.testing import CliRunner

from eregion.cli.main import app
from eregion.cli._common import parse_var_options

runner = CliRunner()


def runnable_pipeline_dict(value: int = 42):
    return {
        "pipelines": [
            {
                "name": "PIPE",
                "lazy": False,
                "nodes": [
                    {
                        "name": "echo",
                        "task": "tests.cli_fixtures.EchoTask",
                        "run": {"params": {"value": value}},
                    }
                ],
            }
        ]
    }


def write_yaml(tmp_path, data):
    path = os.path.join(tmp_path, "pipeline.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
    return path


def test_parse_var_options_empty():
    assert parse_var_options(None) == {}
    assert parse_var_options([]) == {}


def test_parse_var_options_parses_key_value_pairs():
    assert parse_var_options(["a=1", "b=two"]) == {"a": "1", "b": "two"}


def test_validate_runnable_config_prints_dag_and_does_not_execute():
    with tempfile.TemporaryDirectory() as td:
        path = write_yaml(td, runnable_pipeline_dict())
        result = runner.invoke(app, ["validate", path])

    assert result.exit_code == 0
    assert "Config OK: 1 pipeline(s) defined." in result.output
    assert "PIPE" in result.output
    assert "No tasks were executed." in result.output


def test_validate_missing_config_file_is_a_usage_error():
    result = runner.invoke(app, ["validate", "/no/such/file.yaml"])
    assert result.exit_code != 0


def test_run_executes_pipeline_and_reports_results():
    with tempfile.TemporaryDirectory() as td:
        path = write_yaml(td, runnable_pipeline_dict(value=42))
        result = runner.invoke(app, ["run", path])

    assert result.exit_code == 0
    assert "Pipeline run complete." in result.output
    assert "PIPE.echo" in result.output
    assert "value" in result.output


def test_run_accepts_runtime_variable_overrides():
    with tempfile.TemporaryDirectory() as td:
        data = runnable_pipeline_dict()
        data["pipelines"][0]["nodes"][0]["run"]["params"]["value"] = "${my_value}"
        path = write_yaml(td, data)
        result = runner.invoke(app, ["run", path, "--var", "my_value=7"])

    assert result.exit_code == 0
    assert "Pipeline run complete." in result.output
