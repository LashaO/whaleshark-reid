"""Tests for cli.run_context — RunContext lifecycle and the context manager wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from whaleshark_reid.cli.run_context import RunContext, detect_git_sha, run_context
from whaleshark_reid.storage.db import Storage


def test_run_context_new_creates_run_row(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    ctx = RunContext.new(stage="ingest", storage=storage, cache_dir=tmp_cache_dir,
                         config={"a": 1})

    assert ctx.run_id.startswith("run_")
    assert ctx.stage == "ingest"
    assert storage.get_run_status(ctx.run_id) == "running"
    row = storage.conn.execute(
        "SELECT config_json FROM runs WHERE run_id = ?", (ctx.run_id,)
    ).fetchone()
    assert json.loads(row["config_json"]) == {"a": 1}


def test_run_context_finish_ok(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ctx = RunContext.new(stage="embed", storage=storage, cache_dir=tmp_cache_dir, config={})

    ctx.finish(status="ok", metrics={"n": 5})

    assert storage.get_run_status(ctx.run_id) == "ok"
    row = storage.conn.execute(
        "SELECT metrics_json FROM runs WHERE run_id = ?", (ctx.run_id,)
    ).fetchone()
    assert json.loads(row["metrics_json"]) == {"n": 5}


def test_run_context_manager_success(tmp_db_path: Path, tmp_cache_dir: Path):
    """Block exits normally without explicit finish — safety net marks ok."""
    storage = Storage(tmp_db_path)
    storage.init_schema()

    with run_context(stage="cluster", storage=storage, cache_dir=tmp_cache_dir, config={"eps": 0.7}) as ctx:
        run_id = ctx.run_id
        # do nothing — exits without explicit finish

    assert storage.get_run_status(run_id) == "ok"


def test_run_context_manager_explicit_finish(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    with run_context(stage="ingest", storage=storage, cache_dir=tmp_cache_dir, config={}) as ctx:
        ctx.finish(status="ok", metrics={"n_ingested": 42})
        run_id = ctx.run_id

    assert storage.get_run_status(run_id) == "ok"
    row = storage.conn.execute(
        "SELECT metrics_json FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert json.loads(row["metrics_json"]) == {"n_ingested": 42}


def test_run_context_manager_exception_marks_failed(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    captured_run_id = None

    with pytest.raises(ValueError, match="boom"):
        with run_context(stage="matching", storage=storage, cache_dir=tmp_cache_dir, config={}) as ctx:
            captured_run_id = ctx.run_id
            raise ValueError("boom")

    assert captured_run_id is not None
    assert storage.get_run_status(captured_run_id) == "failed"
    row = storage.conn.execute(
        "SELECT error FROM runs WHERE run_id = ?", (captured_run_id,)
    ).fetchone()
    assert "boom" in row["error"]


def test_detect_git_sha_returns_string_or_none():
    sha = detect_git_sha()
    # In this repo it should be a real sha, but the function must tolerate non-repos too
    assert sha is None or (isinstance(sha, str) and len(sha) == 40)
