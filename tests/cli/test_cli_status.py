"""CLI integration test for `catalog status`."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner


def test_catalog_status_on_empty_db(cli_runner: CliRunner, tmp_db_path: Path):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["status", "--db-path", str(tmp_db_path)])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "embed" in result.stdout
    assert "(no runs)" in result.stdout
    assert "Annotations: 0" in result.stdout


def test_catalog_status_after_run_all(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app

    FIXTURES = Path(__file__).parent.parent / "fixtures"
    cli_runner.invoke(app, [
        "run-all",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--distance-threshold", "2.0",
        "--num-workers", "0",
        "--device", "cpu",
    ])

    result = cli_runner.invoke(app, ["status", "--db-path", str(tmp_db_path)])
    assert result.exit_code == 0
    # All 5 stages should show as 'ok'
    assert result.stdout.count("ok") >= 5
    assert "Annotations: 10" in result.stdout
    # Should show at least some key metrics
    assert "ingested=" in result.stdout
    assert "clusters=" in result.stdout
    assert "pairs=" in result.stdout
