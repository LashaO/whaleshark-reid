"""Cross-cutting tests for CLI exit codes."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_missing_required_arg_exits_2(cli_runner: CliRunner):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["ingest"])  # missing --csv and --photos-dir
    assert result.exit_code == 2  # typer's "usage error" exit code


def test_unknown_command_exits_2(cli_runner: CliRunner):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code == 2


def test_cluster_with_no_embed_runs_exits_2(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path
):
    from whaleshark_reid.cli.main import app

    # Initialize the DB with no embed runs
    result = cli_runner.invoke(app, [
        "cluster",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 2
    assert "no successful embed run" in result.stdout.lower() or \
           "no successful embed run" in (result.stderr or "")


def test_matching_with_no_cluster_runs_exits_2(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    cli_runner.invoke(app, [
        "embed",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ])

    # Now matching without cluster
    result = cli_runner.invoke(app, [
        "matching",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 2


def test_pipeline_exception_exits_1_and_marks_run_failed(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, monkeypatch
):
    """Force ingest_inat_csv to raise → CLI catches it via run_context → exit 1."""
    from whaleshark_reid.cli.main import app
    from whaleshark_reid.cli.commands import ingest as ingest_module

    def boom(*args, **kwargs):
        raise RuntimeError("intentional explosion")

    monkeypatch.setattr(ingest_module, "ingest_inat_csv", boom)

    result = cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code != 0
    # Confirm the run row was marked failed
    from whaleshark_reid.storage.db import Storage
    storage = Storage(tmp_db_path)
    row = storage.conn.execute(
        "SELECT status, error FROM runs WHERE stage = 'ingest'"
    ).fetchone()
    assert row["status"] == "failed"
    assert "intentional explosion" in row["error"]
