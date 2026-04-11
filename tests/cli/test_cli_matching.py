"""CLI integration test for `catalog matching`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _seed_through_cluster(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
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
        "--batch-size", "4",
        "--num-workers", "0",
        "--device", "cpu",
    ])
    cli_runner.invoke(app, [
        "cluster",
        "--algo", "dbscan",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])


def test_catalog_matching_writes_pair_queue(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_through_cluster(cli_runner, tmp_db_path, tmp_cache_dir)

    result = cli_runner.invoke(app, [
        "matching",
        "--distance-threshold", "2.0",
        "--max-queue-size", "1000",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'matching'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    metrics = json.loads(runs[0]["metrics_json"])
    assert metrics["n_pairs"] > 0

    n_queue = storage.count("pair_queue", run_id=runs[0]["run_id"])
    assert n_queue == metrics["n_pairs"]
