"""Tests for storage.db — SQLite schema creation and PRAGMAs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from whaleshark_reid.storage.db import Storage


def _fetch_table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def test_storage_creates_file(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    assert tmp_db_path.exists()


def test_storage_has_expected_tables(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    conn = sqlite3.connect(tmp_db_path)
    try:
        tables = _fetch_table_names(conn)
    finally:
        conn.close()
    expected = {"annotations", "pair_decisions", "runs", "pair_queue"}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"
    # Explicitly verify the dropped tables are NOT present.
    assert "individuals_cache" not in tables
    assert "name_uuid_history" not in tables
    assert "experiments" not in tables


def test_storage_has_wal_mode(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    mode = storage.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_storage_has_foreign_keys_on(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    on = storage.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert on == 1


def test_runs_has_metrics_json_column(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    cols = {
        row[1] for row in storage.conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    assert "metrics_json" in cols
    assert "parent_run_id" not in cols


def test_annotations_has_uuid_primary_key(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    cols_with_pk = [
        (row[1], row[5])
        for row in storage.conn.execute("PRAGMA table_info(annotations)").fetchall()
    ]
    pk_cols = [name for name, pk in cols_with_pk if pk > 0]
    assert pk_cols == ["annotation_uuid"]


def test_init_schema_is_idempotent(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.init_schema()  # second call is a no-op thanks to IF NOT EXISTS
    conn = sqlite3.connect(tmp_db_path)
    try:
        tables = _fetch_table_names(conn)
    finally:
        conn.close()
    assert "annotations" in tables
