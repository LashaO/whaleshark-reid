"""Tests for feedback/unionfind.py — rebuild annotations.name_uuid from pair_decisions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from whaleshark_reid.core.feedback.unionfind import rebuild_individuals_cache
from whaleshark_reid.core.schema import (
    Annotation,
    inat_annotation_uuid,
    inat_image_uuid,
)
from whaleshark_reid.storage.db import Storage


def _seed_annotations(storage: Storage, n: int = 5) -> list[str]:
    uuids = []
    for i in range(n):
        ann = Annotation(
            annotation_uuid=inat_annotation_uuid(100 + i, 0),
            image_uuid=inat_image_uuid(100 + i, 0),
            source="inat",
            observation_id=100 + i,
            photo_index=0,
            file_path=f"/tmp/{100+i}.jpg",
            file_name=f"{100+i}.jpg",
            bbox=[0, 0, 10, 10],
        )
        storage.upsert_annotation(ann, run_id="r_seed")
        uuids.append(ann.annotation_uuid)
    return uuids


def _append_decision(storage: Storage, a: str, b: str, decision: str) -> None:
    storage.conn.execute(
        """
        INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, run_id, created_at)
        VALUES (?, ?, ?, 'r_test', ?)
        """,
        (a, b, decision, datetime.now(timezone.utc).isoformat()),
    )


def test_rebuild_creates_single_component_for_confirmed_pair(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    uuids = _seed_annotations(storage, 3)

    _append_decision(storage, uuids[0], uuids[1], "match")

    result = rebuild_individuals_cache(storage)

    assert result.n_components == 1
    assert result.n_singletons == 1  # uuids[2]
    assert result.n_annotations_updated == 3  # all three rows touched (2 set, 1 cleared)

    a0 = storage.get_annotation(uuids[0]).name_uuid
    a1 = storage.get_annotation(uuids[1]).name_uuid
    a2 = storage.get_annotation(uuids[2]).name_uuid
    assert a0 is not None
    assert a0 == a1
    assert a2 is None


def test_rebuild_creates_two_components(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    u = _seed_annotations(storage, 5)

    _append_decision(storage, u[0], u[1], "match")
    _append_decision(storage, u[1], u[2], "match")  # connects transitively
    _append_decision(storage, u[3], u[4], "match")

    result = rebuild_individuals_cache(storage)
    assert result.n_components == 2

    name0 = storage.get_annotation(u[0]).name_uuid
    name1 = storage.get_annotation(u[1]).name_uuid
    name2 = storage.get_annotation(u[2]).name_uuid
    name3 = storage.get_annotation(u[3]).name_uuid
    name4 = storage.get_annotation(u[4]).name_uuid

    assert name0 == name1 == name2
    assert name3 == name4
    assert name0 != name3


def test_rebuild_ignores_no_match_decisions(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    u = _seed_annotations(storage, 3)

    _append_decision(storage, u[0], u[1], "no_match")

    result = rebuild_individuals_cache(storage)
    assert result.n_components == 0
    assert result.n_singletons == 3


def test_rebuild_ignores_superseded_decisions(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    u = _seed_annotations(storage, 3)

    now = datetime.now(timezone.utc).isoformat()
    # Insert the original decision (will be superseded)
    storage.conn.execute(
        """
        INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, run_id, created_at)
        VALUES (?, ?, 'match', 'r_test', ?)
        """,
        (u[0], u[1], now),
    )
    old_id = storage.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Insert a superseding decision (the newer one that replaces it)
    storage.conn.execute(
        """
        INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, run_id, created_at)
        VALUES (?, ?, 'no_match', 'r_test', ?)
        """,
        (u[0], u[1], now),
    )
    new_id = storage.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    # Mark the original as superseded by the newer decision
    storage.conn.execute(
        "UPDATE pair_decisions SET superseded_by = ? WHERE decision_id = ?",
        (new_id, old_id),
    )

    result = rebuild_individuals_cache(storage)
    assert result.n_components == 0


def test_rebuild_clears_stale_name_uuids(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    u = _seed_annotations(storage, 3)

    # Manually set a name_uuid on u[0] to simulate a stale rebuild
    storage.set_annotation_name_uuid(u[0], "stale-00000000-0000-0000-0000-000000000000")

    # No confirmed pairs → rebuild should clear the stale uuid
    rebuild_individuals_cache(storage)

    assert storage.get_annotation(u[0]).name_uuid is None
