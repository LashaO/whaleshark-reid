"""Catalog CLI — Typer entry point.

Each subcommand lives in its own module under cli/commands/ and is registered
here. Commands themselves do no ML; they're thin orchestration over core.
"""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command

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
app.command(name="embed")(embed_command)
app.command(name="cluster")(cluster_command)
app.command(name="matching")(matching_command)


if __name__ == "__main__":
    app()
