"""
`eregion run`: execute a pipeline flow config end-to-end.
"""
from pathlib import Path
from typing import Optional

import typer

from eregion.pipeline import PipelineEngine
from eregion.cli._common import parse_var_options, fail


def run(
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
    max_workers: int = typer.Option(
        4, "--max-workers",
        help="Max number of independent pipelines to run concurrently.",
    ),
):
    """
    Build the pipeline DAG from a config and run every pipeline/task in dependency order.
    Equivalent to:
        from eregion.pipeline import PipelineEngine
        engine = PipelineEngine(config)
        engine.run()
    """
    runtime_variables = parse_var_options(var)

    try:
        engine = PipelineEngine(str(config), runtime_variables=runtime_variables, enable_env_vars=env)
    except Exception as e:
        fail(f"Failed to build pipeline from '{config}': {e}")

    try:
        engine.run(max_workers=max_workers)
    except Exception as e:
        fail(f"Pipeline run failed: {e}")

    typer.secho("\nPipeline run complete.", fg=typer.colors.GREEN)
    for node_name, result in engine.results.items():
        typer.echo(f"  - {node_name}: {list(result.keys())}")
