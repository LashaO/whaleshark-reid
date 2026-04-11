"""CLI integration test for `catalog rebuild-individuals`."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.core.schema import inat_annotation_uuid
from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_rebuild_individuals_assigns_name_uuid(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path
):
    from whaleshark_reid.cli.main import app

    # Seed annotations via ingest
    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])

    # Manually insert a confirmed match between obs100 and obs101
    storage = Storage(tmp_db_path)
    u0 = inat_annotation_uuid(100, 0)
    u1 = inat_annotation_uuid(101, 0)
    storage.conn.execute(
        "INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, created_at) "
        "VALUES (?, ?, 'match', ?)",
        (u0, u1, datetime.now(timezone.utc).isoformat()),
    )
    storage.conn.commit()

    # Rebuild
    result = cli_runner.invoke(app, [
        "rebuild-individuals",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    a = storage.get_annotation(u0)
    b = storage.get_annotation(u1)
    assert a.name_uuid is not None
    assert a.name_uuid == b.name_uuid

    runs = storage.conn.execute(
        "SELECT status, metrics_json FROM runs WHERE stage = 'rebuild'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
