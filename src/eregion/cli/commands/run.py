"""
`eregion run`: execute a flow config end-to-end.
"""
from pathlib import Path
from typing import Optional

import typer

from eregion.flow import FlowRunner
from eregion.cli._common import parse_var_options, fail


def run(
    config: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False,
        help="Path to a flow YAML config.",
    ),
    var: Optional[list[str]] = typer.Option(
        None, "--var", "-v",
        help="Runtime variable used to resolve ${...} placeholders in the config, as KEY=VALUE.",
    ),
    env: bool = typer.Option(
        False, "--env",
        help="Allow ${VAR} placeholders in the config to fall back to environment variables.",
    ),
):
    """
    Build the flow DAG from a config and run every flow/task in dependency order.
    Equivalent to:
        from eregion.flow import FlowRunner
        runner = FlowRunner(config)
        runner.run()
    """
    runtime_variables = parse_var_options(var)

    try:
        runner = FlowRunner(str(config), runtime_variables=runtime_variables, enable_env_vars=env)
    except Exception as e:
        fail(f"Failed to build flow from '{config}': {e}")

    try:
        runner.run()
    except Exception as e:
        fail(f"Flow run failed: {e}")

    typer.secho("\nFlow run complete.", fg=typer.colors.GREEN)
    for node_name, result in runner.results.items():
        typer.echo(f"  - {node_name}: {list(result.keys())}")
