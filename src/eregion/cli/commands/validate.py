"""
`eregion validate`: parse a detector config and print the structure it describes.
"""
from pathlib import Path
from typing import Optional

import typer

from eregion.configs import DetectorConfig
from eregion.cli._common import parse_var_options, fail


def validate(
    config: Path = typer.Argument(
        ..., exists=True, readable=True, dir_okay=False,
        help="Path to a detector YAML config.",
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
    Load a detector config and print the detector structure it describes.

    No image data is read, so this is a cheap sanity check on a config.
    """
    runtime_variables = parse_var_options(var)

    try:
        detector_config = DetectorConfig(
            str(config), runtime_variables=runtime_variables, enable_env_vars=env,
        )
    except Exception as e:
        fail(f"Failed to load detector config from '{config}': {e}")

    cfg = detector_config.config
    objects = cfg["objects"]
    typer.secho(f"Config OK: {len(objects)} detector object(s) defined.\n", fg=typer.colors.GREEN)

    typer.echo(f"detector_type: {cfg['detector_type']}")
    typer.echo(f"detector_output_class: {cfg['detector_output_class']}")
    if cfg.get("description"):
        typer.echo(f"description: {cfg['description']}")

    for obj in objects:
        props = obj["properties"]
        typer.echo(f"\n{obj['name']} ({obj['class']})")
        typer.echo(f"    size: {props['x_size']} x {props['y_size']} px, pixel_size: {props['pixel_size']}")
        outputs = obj["outputs"]
        typer.echo(f"    outputs ({len(outputs)}): {[out.get('id') for out in outputs]}")

    typer.echo("\nNo image data was read.")
