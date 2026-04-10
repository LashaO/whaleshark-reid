"""Tests for annotation CRUD on Storage."""
from __future__ import annotations

from pathlib import Path

import pytest

from whaleshark_reid.core.schema import Annotation, inat_annotation_uuid, inat_image_uuid
from whaleshark_reid.storage.db import Storage


def _make_ann(obs_id: int, idx: int = 0, name: str | None = None) -> Annotation:
    return Annotation(
        annotation_uuid=inat_annotation_uuid(obs_id, idx),
        image_uuid=inat_image_uuid(obs_id, idx),
        name=name,
        source="inat",
        observation_id=obs_id,
        photo_index=idx,
        file_path=f"/tmp/{obs_id}_{idx+1}.jpg",
        file_name=f"{obs_id}_{idx+1}.jpg",
        bbox=[10.0, 20.0, 100.0, 200.0],
        theta=0.0,
        viewpoint="unknown",
        species="whaleshark",
        gps_lat_captured=25.66,
        gps_lon_captured=-80.17,
        date_captured="2025-11-19",
    )


def test_upsert_and_get_annotation(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ann = _make_ann(327286790)

    storage.upsert_annotation(ann, run_id="run_test_1")
    got = storage.get_annotation(ann.annotation_uuid)

    assert got is not None
    assert got.annotation_uuid == ann.annotation_uuid
    assert got.observation_id == 327286790
    assert got.bbox == [10.0, 20.0, 100.0, 200.0]
    assert got.gps_lat_captured == 25.66


def test_upsert_annotation_is_idempotent_on_unique_key(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ann = _make_ann(327286790)

    storage.upsert_annotation(ann, run_id="run_test_1")
    storage.upsert_annotation(ann, run_id="run_test_2")  # second call

    assert storage.count("annotations") == 1


def test_list_annotations_returns_all(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.upsert_annotation(_make_ann(100, 0), run_id="r1")
    storage.upsert_annotation(_make_ann(101, 0), run_id="r1")
    storage.upsert_annotation(_make_ann(102, 0), run_id="r1")

    rows = storage.list_annotations()
    assert len(rows) == 3
    assert all(isinstance(r, Annotation) for r in rows)


def test_count_annotations(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.upsert_annotation(_make_ann(100, 0), run_id="r1")
    storage.upsert_annotation(_make_ann(101, 0), run_id="r1")
    assert storage.count("annotations") == 2


def test_set_annotation_name_uuid(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ann = _make_ann(100, 0)
    storage.upsert_annotation(ann, run_id="r1")

    storage.set_annotation_name_uuid(ann.annotation_uuid, "9ff00000-0000-0000-0000-000000000001")
    got = storage.get_annotation(ann.annotation_uuid)
    assert got.name_uuid == "9ff00000-0000-0000-0000-000000000001"

    storage.set_annotation_name_uuid(ann.annotation_uuid, None)
    got = storage.get_annotation(ann.annotation_uuid)
    assert got.name_uuid is None


def test_get_annotation_missing(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    assert storage.get_annotation("does-not-exist") is None
