"""
`eregion validate`: parse a pipeline flow config and print its DAG execution order.
"""
from pathlib import Path
from typing import Optional

import typer

from eregion.pipeline import PipelineEngine
from eregion.cli._common import parse_var_options, fail


def validate(
    config: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False,
        help="Path to a pipeline flow YAML config.",
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
    Build the pipeline DAG from a config and print the resulting execution plan, 
    without executing any task. Useful for sanity-checking a config.
    """
    runtime_variables = parse_var_options(var)

    try:
        engine = PipelineEngine(str(config), runtime_variables=runtime_variables, enable_env_vars=env)
    except Exception as e:
        fail(f"Failed to build pipeline from '{config}': {e}")

    pipe_order, node_orders = engine.execution_orders
    typer.secho(f"Config OK: {len(engine.pipelines)} pipeline(s) defined.\n", fg=typer.colors.GREEN)

    for gen_idx, pipe_names in enumerate(pipe_order):
        typer.echo(f"pipeline generation {gen_idx}: {sorted(pipe_names)}")
        for pipe_name in sorted(pipe_names):
            for step_idx, node_names in enumerate(node_orders[pipe_name]):
                typer.echo(f"    [{pipe_name}] step {step_idx}: {sorted(node_names)}")

    typer.echo("\nNo tasks were executed.")
