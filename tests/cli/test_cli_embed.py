"""CLI integration test for `catalog embed`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.embedding_cache import read_embeddings

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_embed_creates_parquet_and_run_row(
    cli_runner: CliRunner,
    tmp_db_path: Path,
    tmp_cache_dir: Path,
    stub_miewid,
):
    from whaleshark_reid.cli.main import app

    # Need ingested annotations first
    ingest_result = cli_runner.invoke(
        app,
        [
            "ingest",
            "--csv", str(FIXTURES / "mini_inat.csv"),
            "--photos-dir", str(FIXTURES / "photos"),
            "--db-path", str(tmp_db_path),
            "--cache-dir", str(tmp_cache_dir),
        ],
    )
    assert ingest_result.exit_code == 0

    # Now embed
    result = cli_runner.invoke(
        app,
        [
            "embed",
            "--db-path", str(tmp_db_path),
            "--cache-dir", str(tmp_cache_dir),
            "--batch-size", "4",
            "--num-workers", "0",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    embed_runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'embed'"
    ).fetchall()
    assert len(embed_runs) == 1
    assert embed_runs[0]["status"] == "ok"
    metrics = json.loads(embed_runs[0]["metrics_json"])
    assert metrics["n_embedded"] == 10
    assert metrics["embed_dim"] == 8

    df = read_embeddings(tmp_cache_dir, embed_runs[0]["run_id"])
    assert len(df) == 10
