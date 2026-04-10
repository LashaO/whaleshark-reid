"""SQLite storage layer. Single-file DB with WAL mode.

The Storage class wraps a sqlite3 connection and exposes typed operations for
annotations, pair decisions, runs, and the pair queue. Per-table operation
modules add to this class in subsequent tasks — here we establish the
connection management + schema initialization only.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_SQL_PATH = Path(__file__).parent / "schema.sql"


class Storage:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def init_schema(self) -> None:
        with open(SCHEMA_SQL_PATH) as f:
            self.conn.executescript(f.read())

    def close(self) -> None:
        self.conn.close()

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
