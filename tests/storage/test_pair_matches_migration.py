"""Verify pair_matches table is created by schema init + additive migration."""
from pathlib import Path

from whaleshark_reid.storage.db import Storage


def _table_columns(storage: Storage, table: str) -> set[str]:
    return {r["name"] for r in storage.conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_pair_matches_created_on_fresh_db(tmp_db_path: Path):
    s = Storage(tmp_db_path)
    s.init_schema()
    cols = _table_columns(s, "pair_matches")
    assert {"queue_id", "extractor", "n_matches", "mean_score",
            "median_score", "match_data", "img_a_size", "img_b_size",
            "computed_at"}.issubset(cols)


def test_pair_matches_added_by_migration(tmp_db_path: Path):
    """A DB that pre-dates pair_matches should get it via _apply_migrations."""
    s = Storage(tmp_db_path)
    # Bare schema without pair_matches
    s.conn.executescript("""
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, stage TEXT, config_json TEXT,
            metrics_json TEXT, notes TEXT, git_sha TEXT, started_at TEXT,
            finished_at TEXT, status TEXT, error TEXT);
        CREATE TABLE annotations (annotation_uuid TEXT PRIMARY KEY);
        CREATE TABLE pair_queue (queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, ann_a_uuid TEXT, ann_b_uuid TEXT,
            distance REAL, cluster_a INTEGER, cluster_b INTEGER,
            same_cluster INTEGER, position INTEGER);
    """)
    assert "pair_matches" not in {
        r["name"] for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    s._apply_migrations()
    assert "pair_matches" in {
        r["name"] for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def test_pair_matches_pk_prevents_duplicate_extractor(tmp_db_path: Path):
    import pytest, sqlite3
    s = Storage(tmp_db_path)
    s.init_schema()
    s.conn.execute("INSERT INTO runs(run_id, stage, config_json, started_at, status) VALUES('r','match','{}', '2026-04-13', 'ok')")
    s.conn.execute(
        "INSERT INTO annotations(annotation_uuid, image_uuid, source, source_annotation_id, "
        "source_image_id, file_path, file_name, bbox_x, bbox_y, bbox_w, bbox_h, ingested_run_id, created_at) "
        "VALUES('a', 'i1', 's', 'sa', 'si', 'fp', 'fn', 0, 0, 1, 1, 'r', '2026-04-13')"
    )
    s.conn.execute(
        "INSERT INTO annotations(annotation_uuid, image_uuid, source, source_annotation_id, "
        "source_image_id, file_path, file_name, bbox_x, bbox_y, bbox_w, bbox_h, ingested_run_id, created_at) "
        "VALUES('b', 'i2', 's', 'sb', 'si', 'fp', 'fn', 0, 0, 1, 1, 'r', '2026-04-13')"
    )
    s.conn.execute(
        "INSERT INTO pair_queue(run_id, ann_a_uuid, ann_b_uuid, distance, position) VALUES('r','a','b', 0.1, 0)"
    )
    qid = s.conn.execute("SELECT queue_id FROM pair_queue").fetchone()["queue_id"]
    s.conn.execute(
        "INSERT INTO pair_matches(queue_id, extractor, n_matches, match_data, img_a_size, img_b_size) "
        "VALUES(?, 'aliked', 12, '{}', '[440,440]', '[440,440]')", (qid,)
    )
    with pytest.raises(sqlite3.IntegrityError):
        s.conn.execute(
            "INSERT INTO pair_matches(queue_id, extractor, n_matches, match_data, img_a_size, img_b_size) "
            "VALUES(?, 'aliked', 99, '{}', '[440,440]', '[440,440]')", (qid,)
        )
