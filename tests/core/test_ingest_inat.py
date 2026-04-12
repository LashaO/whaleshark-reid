"""Tests for iNat CSV ingest."""
from __future__ import annotations

from pathlib import Path

import pytest

from whaleshark_reid.core.ingest.inat import ingest_inat_csv
from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_ingest_minimal_csv_populates_annotations(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    result = ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
    )

    assert result.n_read == 10
    assert result.n_ingested == 10
    assert result.n_skipped_existing == 0
    assert result.n_missing_files == 0
    assert storage.count("annotations") == 10


def test_ingest_rebuilds_file_path(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
    )
    rows = storage.list_annotations()
    for ann in rows:
        assert ann.file_path.startswith(str(FIXTURES / "photos"))
        assert "stale" not in ann.file_path


def test_ingest_is_idempotent(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
    )
    second = ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t2",
    )

    assert second.n_skipped_existing == 10
    assert second.n_ingested == 0
    assert storage.count("annotations") == 10


def test_ingest_uses_deterministic_uuids(tmp_db_path: Path):
    from whaleshark_reid.core.schema import inat_annotation_uuid

    storage = Storage(tmp_db_path)
    storage.init_schema()
    ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
    )
    expected_uuid = inat_annotation_uuid(100, 0)
    row = storage.get_annotation(expected_uuid)
    assert row is not None
    assert row.observation_id == 100
    assert row.photo_index == 0


def test_ingest_joins_rich_csv_for_provenance(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
        rich_csv_path=FIXTURES / "mini_inat_rich.csv",
    )

    from whaleshark_reid.core.schema import inat_annotation_uuid
    ann = storage.get_annotation(inat_annotation_uuid(100, 0))
    assert ann.photographer == "alice"
    assert ann.gps_lat_captured == 25.66
    assert ann.date_captured == "2025-11-19"
    assert ann.license == "CC-BY"
    assert ann.quality_grade == "research"


def test_ingest_joins_raw_inat_csv_for_provenance(tmp_db_path: Path):
    """Raw iNat export (df_exploded_inat_v1.csv style) uses different column names
    than the dfx schema but should backfill the same fields. Detected by the
    'Encounter.decimalLatitude' column."""
    storage = Storage(tmp_db_path)
    storage.init_schema()

    ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
        rich_csv_path=FIXTURES / "mini_inat_raw.csv",
    )

    from whaleshark_reid.core.schema import inat_annotation_uuid
    ann = storage.get_annotation(inat_annotation_uuid(100, 0))
    assert ann.photographer == "alice"
    assert ann.gps_lat_captured == 25.66
    assert ann.gps_lon_captured == -80.17
    assert ann.date_captured == "2025-11-19"
    assert ann.quality_grade == "research"


def test_ingest_missing_file_is_counted_but_not_fatal(tmp_db_path: Path, tmp_path: Path):
    # Build a minimal CSV referencing a non-existent file
    csv_text = ",theta,viewpoint,name,file_name,species,file_path,x,y,w,h\n"
    csv_text += "0,0,unknown,unknown,missing_999_1.jpg,whaleshark,/stale/missing_999_1.jpg,1,1,10,10\n"
    csv_path = tmp_path / "missing.csv"
    csv_path.write_text(csv_text)

    empty_photos = tmp_path / "photos"
    empty_photos.mkdir()

    storage = Storage(tmp_db_path)
    storage.init_schema()
    result = ingest_inat_csv(
        csv_path=csv_path,
        photos_dir=empty_photos,
        storage=storage,
        run_id="run_t1",
    )

    assert result.n_missing_files == 1
    # Row is still inserted (so the web UI can surface it)
    assert result.n_ingested == 1


def test_ingest_normalizes_unknown_name_to_null(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="run_t1",
    )
    anns = storage.list_annotations()
    for ann in anns:
        assert ann.name is None  # 'unknown' → None
        assert ann.viewpoint == "unknown"  # viewpoint stays as 'unknown'
