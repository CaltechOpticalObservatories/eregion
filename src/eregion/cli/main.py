"""
eregion CLI entry point.

Lets pipeline flow configs be run/validated from the terminal.
"""
import typer

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


def main():
    app()


if __name__ == "__main__":
    main()
