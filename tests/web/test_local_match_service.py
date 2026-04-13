from pathlib import Path

from whaleshark_reid.core.match.lightglue import MatchResult
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.services import local_match as svc


def _seed_pair(s: Storage) -> int:
    s.conn.execute("INSERT INTO runs(run_id, stage, config_json, started_at, status) VALUES('r','match','{}', '2026-04-13', 'ok')")
    s.conn.execute(
        "INSERT INTO annotations(annotation_uuid, image_uuid, source, file_path, file_name, bbox_x, bbox_y, bbox_w, bbox_h, theta, ingested_run_id, created_at) "
        "VALUES('a', 'img_a', 'test', '/x/a.jpg', 'a.jpg', 0, 0, 10, 10, 0, 'r_ingest', '2026-04-13'), "
        "('b', 'img_b', 'test', '/x/b.jpg', 'b.jpg', 0, 0, 10, 10, 0, 'r_ingest', '2026-04-13')"
    )
    s.conn.execute("INSERT INTO pair_queue(run_id, ann_a_uuid, ann_b_uuid, distance, position) VALUES('r','a','b',0.1,0)")
    return s.conn.execute("SELECT queue_id FROM pair_queue").fetchone()["queue_id"]


def _make_result() -> MatchResult:
    return MatchResult(
        extractor="aliked", n_matches=2, mean_score=0.7, median_score=0.7,
        kpts_a=[[1, 2]], kpts_b=[[3, 4]], matches=[[0, 0, 0.9], [0, 0, 0.6]],
        img_a_size=[440, 440], img_b_size=[440, 440],
    )


def test_read_returns_none_if_uncached(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    assert svc.read_cached(s, qid, "aliked") is None


def test_write_then_read(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    svc.write_cached(s, qid, _make_result())
    got = svc.read_cached(s, qid, "aliked")
    assert got is not None
    assert got.n_matches == 2
    assert got.kpts_a == [[1, 2]]


def test_write_overwrite_replaces_row(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    svc.write_cached(s, qid, _make_result())
    updated = MatchResult(
        extractor="aliked", n_matches=99, mean_score=0.1, median_score=0.1,
        kpts_a=[], kpts_b=[], matches=[],
        img_a_size=[440, 440], img_b_size=[440, 440],
    )
    svc.write_cached(s, qid, updated)  # upsert
    got = svc.read_cached(s, qid, "aliked")
    assert got.n_matches == 99


def test_lookup_pair_paths(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    paths = svc.lookup_pair_image_paths(s, qid)
    assert paths == ("/x/a.jpg", "/x/b.jpg")


def test_lookup_pair_chip_specs(tmp_db_path: Path):
    """Chip specs carry file_path + bbox xywh + theta so the matcher can extract
    features on the same chip the /image endpoint renders."""
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    spec_a, spec_b = svc.lookup_pair_chip_specs(s, qid)
    assert spec_a == ("a", "/x/a.jpg", [0, 0, 10, 10], 0.0)
    assert spec_b == ("b", "/x/b.jpg", [0, 0, 10, 10], 0.0)
