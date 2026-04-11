"""CLI integration test for `catalog cluster`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.cluster_cache import read_clusters

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _seed_ingest_and_embed(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
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


def test_catalog_cluster_dbscan_uses_latest_embed_run(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_ingest_and_embed(cli_runner, tmp_db_path, tmp_cache_dir)

    result = cli_runner.invoke(app, [
        "cluster",
        "--algo", "dbscan",
        "--eps", "0.7",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    rows = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'cluster'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    metrics = json.loads(rows[0]["metrics_json"])
    assert metrics["algo"] == "dbscan"
    assert "n_clusters" in metrics

    df = read_clusters(tmp_cache_dir, rows[0]["run_id"])
    assert len(df) == 10


def test_catalog_cluster_explicit_embedding_run_id(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_ingest_and_embed(cli_runner, tmp_db_path, tmp_cache_dir)

    storage = Storage(tmp_db_path)
    embed_run_id = storage.get_latest_run_id("embed")
    assert embed_run_id is not None

    result = cli_runner.invoke(app, [
        "cluster",
        "--algo", "dbscan",
        "--embedding-run-id", embed_run_id,
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0
