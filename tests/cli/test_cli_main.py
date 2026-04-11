"""Smoke tests for the catalog CLI app."""
from __future__ import annotations

from typer.testing import CliRunner


def test_app_importable():
    from whaleshark_reid.cli.main import app  # noqa: F401


def test_help_lists_no_commands_yet(cli_runner: CliRunner):
    """Before any commands are registered, --help should still work and exit 0."""
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "catalog" in result.stdout.lower() or "Usage" in result.stdout
