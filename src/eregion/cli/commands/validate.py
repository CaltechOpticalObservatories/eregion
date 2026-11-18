"""
`eregion validate`: parse a flow config and print its DAG execution order.
"""
from pathlib import Path
from typing import Optional

import typer

from eregion.flow import FlowRunner
from eregion.cli._common import parse_var_options, fail


def validate(
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
    Build the flow DAG from a config and print the resulting execution plan,
    without executing any task. Building the runner dry runs the flow with dummy results,
    so wiring mistakes between tasks are reported here too.
    """
    runtime_variables = parse_var_options(var)

    try:
        runner = FlowRunner(str(config), runtime_variables=runtime_variables, enable_env_vars=env)
    except Exception as e:
        fail(f"Failed to build flow from '{config}': {e}")

    typer.secho(f"Config OK: {len(runner.flows)} flow(s) defined, "
                f"{len(runner.dummy_results)} node(s) checked by the dry run.\n", fg=typer.colors.GREEN)

    for gen_idx, flow_names in enumerate(runner.flow_order):
        typer.echo(f"flow generation {gen_idx}: {sorted(flow_names)}")
        for flow_name in sorted(flow_names):
            flow = runner.flows[flow_name]
            if flow.depends_on:
                typer.echo(f"    [{flow_name}] waits for: {sorted(flow.depends_on)}")
            for step_idx, node_names in enumerate(flow.node_order):
                typer.echo(f"    [{flow_name}] step {step_idx}: {sorted(node_names)}")

    typer.echo("\nNo tasks were executed.")
