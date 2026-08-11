"""
Small helpers shared across CLI commands.
"""
import typer


def parse_var_options(pairs: list[str] | None) -> dict:
    """
    Parse repeated ``--var KEY=VALUE`` options into a runtime_variables dict.

    :param pairs: list of str, optional
        Raw ``KEY=VALUE`` strings collected from the command line. 
        May be None or empty.
    :return: dict
        Mapping of KEY -> VALUE (both as strings) for passing as
        variables to PipelineConfig/PipelineEngine.
    """
    variables = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if not sep:
            raise typer.BadParameter(f"Expected KEY=VALUE, got '{pair}'")
        variables[key] = value
    return variables


def fail(message: str) -> None:
    """
    Print an error message to stderr and exit the CLI with a non-zero status.

    :param message: str
        Error message to display to the user.
    """
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)
