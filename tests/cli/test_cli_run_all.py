"""CLI integration test for `catalog run-all` — chains 5 stages."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_run_all_chains_five_stages(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, [
        "run-all",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--rich-csv", str(FIXTURES / "mini_inat_rich.csv"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--distance-threshold", "2.0",
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    rows = storage.conn.execute(
        "SELECT stage, status FROM runs ORDER BY started_at ASC"
    ).fetchall()
    stages = [r["stage"] for r in rows]
    assert stages == ["ingest", "embed", "cluster", "matching", "project"]
    assert all(r["status"] == "ok" for r in rows)

    assert storage.count("annotations") == 10
    assert storage.count("pair_queue") > 0
