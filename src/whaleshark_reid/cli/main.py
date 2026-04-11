"""Catalog CLI — Typer entry point.

Each subcommand lives in its own module under cli/commands/ and is registered
here. Commands themselves do no ML; they're thin orchestration over core.
"""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.ingest import ingest_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main() -> None:
    """Whale shark re-identification pipeline CLI."""
    pass


app.command(name="ingest")(ingest_command)


if __name__ == "__main__":
    app()
