"""CLI integration test for `catalog project`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.projection_cache import read_projections

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _seed_through_embed(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
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


def test_catalog_project_writes_projection_parquet(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_through_embed(cli_runner, tmp_db_path, tmp_cache_dir)

    result = cli_runner.invoke(app, [
        "project",
        "--n-neighbors", "5",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'project'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    metrics = json.loads(runs[0]["metrics_json"])
    assert metrics["n_points"] == 10
    assert metrics["algo"] == "umap"

    df = read_projections(tmp_cache_dir, runs[0]["run_id"])
    assert len(df) == 10
    assert "x" in df.columns and "y" in df.columns
