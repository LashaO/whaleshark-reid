"""CLI integration test for `catalog ingest`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_ingest_creates_run_and_annotations(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path
):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(
        app,
        [
            "ingest",
            "--csv", str(FIXTURES / "mini_inat.csv"),
            "--photos-dir", str(FIXTURES / "photos"),
            "--rich-csv", str(FIXTURES / "mini_inat_rich.csv"),
            "--db-path", str(tmp_db_path),
            "--cache-dir", str(tmp_cache_dir),
        ],
    )

    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"
    assert "ingested" in result.stdout

    storage = Storage(tmp_db_path)
    assert storage.count("annotations") == 10

    runs = storage.conn.execute(
        "SELECT run_id, stage, status, metrics_json FROM runs ORDER BY started_at DESC"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["stage"] == "ingest"
    assert runs[0]["status"] == "ok"
    metrics = json.loads(runs[0]["metrics_json"])
    assert metrics["n_ingested"] == 10
    assert metrics["n_skipped_existing"] == 0
