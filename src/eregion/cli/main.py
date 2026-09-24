"""
eregion CLI entry point.

Lets pipeline flow configs be run/validated from the terminal.
"""
from typing import Optional

import typer

from eregion import __version__
from eregion.cli.commands.run import run
from eregion.cli.commands.validate import validate

app = typer.Typer(
    name="eregion",
    help="Run eregion pipeline flow configs as DAGs of tasks.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("run")(run)
app.command("validate")(validate)


def _show_version(value: bool):
    if value:
        typer.echo(f"eregion {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Optional[bool] = typer.Option(
        None, "--version",
        help="Show the installed eregion version and exit.",
        callback=_show_version, is_eager=True,
    ),
):
    """
    Run eregion pipeline flow configs as DAGs of tasks.
    """


def main():
    app()


if __name__ == "__main__":
    main()
