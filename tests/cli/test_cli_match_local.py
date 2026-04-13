from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.cli.main import app
from whaleshark_reid.core.match import lightglue as lg_module
from whaleshark_reid.core.match.lightglue import MatchResult
from whaleshark_reid.storage.db import Storage


def _seed(s: Storage):
    s.conn.execute("INSERT INTO runs(run_id, stage, config_json, started_at, status) VALUES('r','match','{}', '2026-04-13', 'ok')")
    s.conn.execute("""
        INSERT INTO annotations(annotation_uuid, image_uuid, source, file_path, file_name, bbox_x, bbox_y, bbox_w, bbox_h, theta, ingested_run_id, created_at)
        VALUES
            ('a', 'img1', 'test', '/x/a.jpg', 'a.jpg', 0, 0, 10, 10, 0, 'test', '2026-04-13'),
            ('b', 'img2', 'test', '/x/b.jpg', 'b.jpg', 0, 0, 10, 10, 0, 'test', '2026-04-13'),
            ('c', 'img3', 'test', '/x/c.jpg', 'c.jpg', 0, 0, 10, 10, 0, 'test', '2026-04-13')
    """)
    s.conn.execute("INSERT INTO pair_queue(run_id, ann_a_uuid, ann_b_uuid, distance, position) VALUES('r','a','b',0.1,0),('r','a','c',0.2,1)")


def _fake_matcher(monkeypatch):
    class FM:
        extractor_name = "aliked"
        def _extract(self, path, bbox=None, theta=0.0):
            return {"keypoints": [[1, 2]]}, 440, 440
        def _match_prebuilt(self, fa, fb, sa, sb):
            return MatchResult(
                extractor="aliked", n_matches=5, mean_score=0.6, median_score=0.6,
                kpts_a=[[1, 2]], kpts_b=[[3, 4]], matches=[[0, 0, 0.9]],
                img_a_size=list(sa), img_b_size=list(sb),
            )
    lg_module._MATCHER_CACHE.clear()
    monkeypatch.setattr(lg_module, "_build_matcher", lambda e: FM())


def test_match_local_writes_cache_for_all_pairs(tmp_db_path: Path, monkeypatch):
    s = Storage(tmp_db_path); s.init_schema(); _seed(s); s.close()
    _fake_matcher(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(app, ["match-local", "--run-id", "r", "--db-path", str(tmp_db_path)])
    assert result.exit_code == 0, result.output

    s = Storage(tmp_db_path)
    rows = s.conn.execute("SELECT queue_id, n_matches FROM pair_matches ORDER BY queue_id").fetchall()
    assert len(rows) == 2
    assert all(r["n_matches"] == 5 for r in rows)


def test_match_local_skips_existing_without_overwrite(tmp_db_path: Path, monkeypatch):
    s = Storage(tmp_db_path); s.init_schema(); _seed(s)
    # Pre-insert a fake cache entry for queue_id 1
    from whaleshark_reid.web.services.local_match import write_cached
    qid1 = s.conn.execute("SELECT queue_id FROM pair_queue WHERE ann_b_uuid='b'").fetchone()["queue_id"]
    write_cached(s, qid1, MatchResult(
        extractor="aliked", n_matches=77, mean_score=0.5, median_score=0.5,
        kpts_a=[], kpts_b=[], matches=[], img_a_size=[440, 440], img_b_size=[440, 440],
    ))
    s.close()
    _fake_matcher(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(app, ["match-local", "--run-id", "r", "--db-path", str(tmp_db_path)])
    assert result.exit_code == 0

    s = Storage(tmp_db_path)
    preserved = s.conn.execute("SELECT n_matches FROM pair_matches WHERE queue_id=?", (qid1,)).fetchone()
    assert preserved["n_matches"] == 77  # not overwritten


def test_match_local_overwrite_replaces(tmp_db_path: Path, monkeypatch):
    s = Storage(tmp_db_path); s.init_schema(); _seed(s)
    from whaleshark_reid.web.services.local_match import write_cached
    qid1 = s.conn.execute("SELECT queue_id FROM pair_queue WHERE ann_b_uuid='b'").fetchone()["queue_id"]
    write_cached(s, qid1, MatchResult(
        extractor="aliked", n_matches=77, mean_score=0.5, median_score=0.5,
        kpts_a=[], kpts_b=[], matches=[], img_a_size=[440, 440], img_b_size=[440, 440],
    ))
    s.close()
    _fake_matcher(monkeypatch)
    runner = CliRunner()
    result = runner.invoke(app, ["match-local", "--run-id", "r", "--db-path", str(tmp_db_path), "--overwrite"])
    assert result.exit_code == 0

    s = Storage(tmp_db_path)
    row = s.conn.execute("SELECT n_matches FROM pair_matches WHERE queue_id=?", (qid1,)).fetchone()
    assert row["n_matches"] == 5
