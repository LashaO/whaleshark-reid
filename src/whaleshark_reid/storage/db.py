"""SQLite storage layer. Single-file DB with WAL mode.

The Storage class wraps a sqlite3 connection and exposes typed operations for
annotations, pair decisions, runs, and the pair queue. Per-table operation
modules add to this class in subsequent tasks — here we establish the
connection management + schema initialization only.
"""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Returns NULL if any input is NULL."""
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return None
    r1 = math.radians(lat1)
    r2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


class Storage:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        # Register pure-Python UDFs for geometry calculations used by the pair
        # queue (haversine for km_delta). SQLite has no trig functions built in.
        self.conn.create_function("haversine_km", 4, _haversine_km, deterministic=True)

    def init_schema(self) -> None:
        with open(SCHEMA_SQL_PATH) as f:
            self.conn.executescript(f.read())
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        """Additive column migrations for tables that may pre-date newer columns.

        SQLite's CREATE TABLE IF NOT EXISTS is a no-op on existing tables, so
        any column added after initial release must be applied here.
        """
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(pair_queue)").fetchall()}
        if "km_delta" not in cols:
            self.conn.execute("ALTER TABLE pair_queue ADD COLUMN km_delta REAL")
        if "time_delta_days" not in cols:
            self.conn.execute("ALTER TABLE pair_queue ADD COLUMN time_delta_days REAL")

        # pair_matches table (post-V1 feature)
        has_pm = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='pair_matches'"
        ).fetchone()
        if not has_pm:
            self.conn.executescript("""
                CREATE TABLE pair_matches (
                    queue_id    INTEGER NOT NULL REFERENCES pair_queue(queue_id),
                    extractor   TEXT    NOT NULL,
                    n_matches   INTEGER NOT NULL,
                    mean_score  REAL,
                    median_score REAL,
                    match_data  TEXT    NOT NULL,
                    img_a_size  TEXT    NOT NULL,
                    img_b_size  TEXT    NOT NULL,
                    computed_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (queue_id, extractor)
                );
                CREATE INDEX idx_pm_queue ON pair_matches(queue_id);
            """)

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions in autocommit mode.

        Usage:
            with storage.transaction():
                storage.conn.execute("INSERT ...")
                storage.conn.execute("INSERT ...")

        Wraps the body in BEGIN ... COMMIT (or ROLLBACK on exception).
        """
        self.conn.execute("BEGIN")
        try:
            yield
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # ----- annotation CRUD -----

    def upsert_annotation(self, ann, run_id: str) -> None:
        """INSERT OR IGNORE an Annotation row. Does nothing if
        (source, observation_id, photo_index) already exists."""
        x, y, w, h = ann.bbox
        self.conn.execute(
            """
            INSERT OR IGNORE INTO annotations (
                annotation_uuid, image_uuid, name_uuid,
                source, source_annotation_id, source_image_id, source_individual_id,
                observation_id, photo_index,
                file_path, file_name, bbox_x, bbox_y, bbox_w, bbox_h, theta,
                viewpoint, species, name,
                photographer, license, date_captured, gps_lat_captured, gps_lon_captured,
                coco_url, flickr_url, height, width, conf, quality_grade,
                name_viewpoint, species_viewpoint,
                ingested_run_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ann.annotation_uuid, ann.image_uuid, ann.name_uuid,
                ann.source, ann.source_annotation_id, ann.source_image_id, ann.source_individual_id,
                ann.observation_id, ann.photo_index,
                ann.file_path, ann.file_name, x, y, w, h, ann.theta,
                ann.viewpoint, ann.species, ann.name,
                ann.photographer, ann.license, ann.date_captured, ann.gps_lat_captured, ann.gps_lon_captured,
                ann.coco_url, ann.flickr_url, ann.height, ann.width, ann.conf, ann.quality_grade,
                ann.name_viewpoint, ann.species_viewpoint,
                run_id, datetime.now(timezone.utc).isoformat(),
            ),
        )

    def get_annotation(self, annotation_uuid: str):
        from whaleshark_reid.core.schema import Annotation

        row = self.conn.execute(
            "SELECT * FROM annotations WHERE annotation_uuid = ?",
            (annotation_uuid,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_annotation(row)

    def list_annotations(self) -> list:
        rows = self.conn.execute("SELECT * FROM annotations").fetchall()
        return [self._row_to_annotation(r) for r in rows]

    def count(self, table: str, **where) -> int:
        clause = ""
        params: tuple = ()
        if where:
            clause = " WHERE " + " AND ".join(f"{k} = ?" for k in where)
            params = tuple(where.values())
        row = self.conn.execute(f"SELECT COUNT(*) FROM {table}{clause}", params).fetchone()
        return row[0]

    def set_annotation_name_uuid(self, annotation_uuid: str, name_uuid: str | None) -> None:
        self.conn.execute(
            "UPDATE annotations SET name_uuid = ? WHERE annotation_uuid = ?",
            (name_uuid, annotation_uuid),
        )

    @staticmethod
    def _row_to_annotation(row):
        from whaleshark_reid.core.schema import Annotation

        d = dict(row)
        d["bbox"] = [d.pop("bbox_x"), d.pop("bbox_y"), d.pop("bbox_w"), d.pop("bbox_h")]
        # Drop storage-only columns that Annotation does not have
        d.pop("ingested_run_id", None)
        d.pop("created_at", None)
        return Annotation(**d)

    # ----- runs CRUD -----

    def begin_run(
        self,
        run_id: str,
        stage: str,
        config: dict,
        git_sha: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO runs (run_id, stage, config_json, metrics_json, notes, git_sha,
                              started_at, finished_at, status, error)
            VALUES (?, ?, ?, NULL, NULL, ?, ?, NULL, 'running', NULL)
            """,
            (
                run_id,
                stage,
                json.dumps(config, default=str),
                git_sha,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    def finish_run(
        self,
        run_id: str,
        status: str,
        metrics: dict,
        error: str | None = None,
        notes: str = "",
    ) -> None:
        if status not in ("ok", "failed"):
            raise ValueError(f"Invalid terminal status: {status}")
        self.conn.execute(
            """
            UPDATE runs SET
                status = ?,
                metrics_json = ?,
                notes = ?,
                finished_at = ?,
                error = ?
            WHERE run_id = ?
            """,
            (
                status,
                json.dumps(metrics, default=str),
                notes,
                datetime.now(timezone.utc).isoformat(),
                error,
                run_id,
            ),
        )

    def get_run_status(self, run_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return row["status"] if row else None

    def get_latest_run_id(self, stage: str) -> str | None:
        """Return the most recent successful run_id for a given stage, or None if none exists."""
        row = self.conn.execute(
            """
            SELECT run_id FROM runs
            WHERE stage = ? AND status = 'ok'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (stage,),
        ).fetchone()
        return row["run_id"] if row else None

    # ----- pair_decisions / pair_queue / annotation list (consumed by matching + feedback) -----

    def list_annotation_uuids(self) -> list[str]:
        """All annotation_uuids in the database."""
        rows = self.conn.execute("SELECT annotation_uuid FROM annotations").fetchall()
        return [r["annotation_uuid"] for r in rows]

    def list_active_pair_decisions(self) -> list[tuple[str, str, str]]:
        """Return active (non-superseded) pair decisions as (ann_a_uuid, ann_b_uuid, decision)."""
        rows = self.conn.execute(
            """
            SELECT ann_a_uuid, ann_b_uuid, decision FROM pair_decisions
            WHERE superseded_by IS NULL
            """
        ).fetchall()
        return [(r["ann_a_uuid"], r["ann_b_uuid"], r["decision"]) for r in rows]

    def list_active_match_pairs(self) -> list[tuple[str, str]]:
        """Return active (non-superseded) match pairs as (ann_a_uuid, ann_b_uuid)."""
        rows = self.conn.execute(
            """
            SELECT ann_a_uuid, ann_b_uuid FROM pair_decisions
            WHERE decision = 'match' AND superseded_by IS NULL
            """
        ).fetchall()
        return [(r["ann_a_uuid"], r["ann_b_uuid"]) for r in rows]

    def replace_pair_queue(self, run_id: str, candidates) -> None:
        """Replace all pair_queue rows for a run_id with new candidates.

        candidates is an iterable of PairCandidate objects with fields:
        ann_a_uuid, ann_b_uuid, distance, cluster_a, cluster_b, same_cluster.
        Position is assigned by enumeration order.

        Also populates pair geometry columns (km_delta, time_delta_days) by
        joining annotations. Rows with missing date/GPS get NULLs.

        Wrapped in a transaction for atomicity.
        """
        with self.transaction():
            self.conn.execute("DELETE FROM pair_queue WHERE run_id = ?", (run_id,))
            for position, p in enumerate(candidates):
                self.conn.execute(
                    """
                    INSERT INTO pair_queue (
                        run_id, ann_a_uuid, ann_b_uuid, distance,
                        cluster_a, cluster_b, same_cluster, position
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        p.ann_a_uuid,
                        p.ann_b_uuid,
                        p.distance,
                        p.cluster_a,
                        p.cluster_b,
                        1 if p.same_cluster else 0,
                        position,
                    ),
                )
            self.recompute_pair_geometry(run_id)

    def recompute_pair_geometry(self, run_id: str) -> None:
        """Populate km_delta + time_delta_days from annotations for this run.

        Idempotent; call after inserts or to backfill existing rows that
        pre-date these columns. NULLs where either side lacks date/GPS.
        """
        self.conn.execute(
            """
            UPDATE pair_queue SET
                km_delta = (
                    SELECT haversine_km(a.gps_lat_captured, a.gps_lon_captured,
                                        b.gps_lat_captured, b.gps_lon_captured)
                    FROM annotations a, annotations b
                    WHERE a.annotation_uuid = pair_queue.ann_a_uuid
                      AND b.annotation_uuid = pair_queue.ann_b_uuid
                ),
                time_delta_days = (
                    SELECT ABS(julianday(a.date_captured) - julianday(b.date_captured))
                    FROM annotations a, annotations b
                    WHERE a.annotation_uuid = pair_queue.ann_a_uuid
                      AND b.annotation_uuid = pair_queue.ann_b_uuid
                      AND a.date_captured IS NOT NULL
                      AND b.date_captured IS NOT NULL
                )
            WHERE run_id = ?
            """,
            (run_id,),
        )
