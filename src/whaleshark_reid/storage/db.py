"""SQLite storage layer. Single-file DB with WAL mode.

The Storage class wraps a sqlite3 connection and exposes typed operations for
annotations, pair decisions, runs, and the pair queue. Per-table operation
modules add to this class in subsequent tasks — here we establish the
connection management + schema initialization only.
"""
from __future__ import annotations

import sqlite3
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
