"""Cross-cutting tests verifying that each stage respects idempotency rules."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _run_full_pipeline(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
    from whaleshark_reid.cli.main import app

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


def test_re_ingest_skips_existing(
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

    # Second ingest run on the same data
    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])

    storage = Storage(tmp_db_path)
    assert storage.count("annotations") == 10  # not doubled
    rows = storage.conn.execute(
        "SELECT metrics_json FROM runs WHERE stage = 'ingest' ORDER BY started_at"
    ).fetchall()
    assert len(rows) == 2
    second_metrics = json.loads(rows[1]["metrics_json"])
    assert second_metrics["n_skipped_existing"] == 10
    assert second_metrics["n_ingested"] == 0


def test_re_embed_with_only_missing_skips_already_cached(
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
        "embed", "--db-path", str(tmp_db_path), "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ])

    # Second embed run — gets a new run_id, writes its own parquet.
    # The only_missing flag is per-run_id parquet (not global), so the second run still embeds 10.
    cli_runner.invoke(app, [
        "embed", "--db-path", str(tmp_db_path), "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ])

    storage = Storage(tmp_db_path)
    embed_runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'embed' ORDER BY started_at"
    ).fetchall()
    assert len(embed_runs) == 2
    assert all(r["status"] == "ok" for r in embed_runs)
    # Both runs created their own parquet; the only_missing logic is per-run_id, so both
    # see 10 annotations as "missing" from their own (initially empty) parquet.
    for r in embed_runs:
        m = json.loads(r["metrics_json"])
        assert m["n_embedded"] == 10


def test_re_run_matching_replaces_pair_queue_for_same_run(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    """Each matching run gets a fresh run_id and writes its own pair_queue rows.
    The DELETE+INSERT in run_matching_stage prevents accidental dup within a single run."""
    from whaleshark_reid.cli.main import app
    _run_full_pipeline(cli_runner, tmp_db_path, tmp_cache_dir)

    storage = Storage(tmp_db_path)
    matching_runs_before = storage.conn.execute(
        "SELECT run_id FROM runs WHERE stage = 'matching'"
    ).fetchall()
    assert len(matching_runs_before) == 1
    pairs_before = storage.count("pair_queue", run_id=matching_runs_before[0]["run_id"])

    # Re-run matching → new run_id, separate pair_queue entries
    cli_runner.invoke(app, [
        "matching", "--db-path", str(tmp_db_path), "--cache-dir", str(tmp_cache_dir),
        "--distance-threshold", "2.0",
    ])

    matching_runs_after = storage.conn.execute(
        "SELECT run_id FROM runs WHERE stage = 'matching'"
    ).fetchall()
    assert len(matching_runs_after) == 2
    pairs_after_old = storage.count("pair_queue", run_id=matching_runs_before[0]["run_id"])
    assert pairs_after_old == pairs_before  # untouched
