"""Tests for runs CRUD on Storage."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from whaleshark_reid.storage.db import Storage


def test_begin_run_creates_row_with_running_status(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    storage.begin_run(run_id="run_test_1", stage="ingest", config={"csv": "/a"})
    row = storage.conn.execute(
        "SELECT status, config_json, metrics_json, finished_at FROM runs WHERE run_id = ?",
        ("run_test_1",),
    ).fetchone()

    assert row["status"] == "running"
    assert json.loads(row["config_json"]) == {"csv": "/a"}
    assert row["metrics_json"] is None
    assert row["finished_at"] is None


def test_finish_run_ok_populates_metrics(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.begin_run(run_id="r1", stage="embed", config={"batch_size": 32})

    storage.finish_run(run_id="r1", status="ok", metrics={"n_embedded": 42}, notes="nominal")

    row = storage.conn.execute(
        "SELECT status, metrics_json, notes, finished_at, error FROM runs WHERE run_id = ?",
        ("r1",),
    ).fetchone()
    assert row["status"] == "ok"
    assert json.loads(row["metrics_json"]) == {"n_embedded": 42}
    assert row["notes"] == "nominal"
    assert row["finished_at"] is not None
    assert row["error"] is None


def test_finish_run_failed_populates_error(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.begin_run(run_id="r1", stage="cluster", config={"eps": 0.7})

    storage.finish_run(run_id="r1", status="failed", metrics={}, error="boom")

    row = storage.conn.execute(
        "SELECT status, error FROM runs WHERE run_id = ?",
        ("r1",),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "boom"


def test_get_run_status_transitions(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.begin_run(run_id="r1", stage="ingest", config={})
    assert storage.get_run_status("r1") == "running"

    storage.finish_run(run_id="r1", status="ok", metrics={})
    assert storage.get_run_status("r1") == "ok"


def test_get_run_status_missing_returns_none(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    assert storage.get_run_status("nope") is None
