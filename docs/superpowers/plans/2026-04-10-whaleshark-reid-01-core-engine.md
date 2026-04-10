# Whaleshark Re-ID — Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pure-Python core of the whale shark re-identification system (schema, storage, ingest, embedding, clustering, matching, feedback, metrics) as described in [spec 01](../specs/2026-04-10-whaleshark-reid-01-core-engine.md).

**Architecture:** Package lives at `/workspace/catalog-match/whaleshark-reid/`, src layout, pydantic-first data contracts, SQLite + WAL for tabular state, parquet for embeddings/clusters/projections, reuse of `wbia_miew_id` for all MiewID + distance + metric primitives. No CLI, no web — this plan only delivers the importable core package that can be exercised from a notebook.

**Tech Stack:** Python 3.10+, pydantic 2, sqlite3 (stdlib), pandas, numpy, pyarrow, scikit-learn, hdbscan, umap-learn, torch, transformers (HuggingFace), `wbia_miew_id` installed in editable mode from `/workspace/wbia-plugin-miew-id`.

**Pre-flight assumption:** The spec directory `/workspace/catalog-match/whaleshark-reid/docs/superpowers/specs/` already contains the four spec files. This plan creates the project scaffolding around them.

**Execution notes:**
- Every task begins by `cd /workspace/catalog-match/whaleshark-reid` unless stated otherwise.
- `pytest -x` after each task. Stop on first failure.
- Commit after each task passes tests. Use conventional-commit prefixes (`feat:`, `test:`, `chore:`).
- The first task initializes a git repo — if one already exists, skip `git init`.

---

## Task 1: Project scaffolding + test infra

**Files:**
- Create: `/workspace/catalog-match/whaleshark-reid/pyproject.toml`
- Create: `/workspace/catalog-match/whaleshark-reid/.gitignore`
- Create: `/workspace/catalog-match/whaleshark-reid/README.md`
- Create: `/workspace/catalog-match/whaleshark-reid/src/whaleshark_reid/__init__.py`
- Create: `/workspace/catalog-match/whaleshark-reid/src/whaleshark_reid/core/__init__.py`
- Create: `/workspace/catalog-match/whaleshark-reid/src/whaleshark_reid/storage/__init__.py`
- Create: `/workspace/catalog-match/whaleshark-reid/tests/__init__.py`
- Create: `/workspace/catalog-match/whaleshark-reid/tests/conftest.py`
- Create: `/workspace/catalog-match/whaleshark-reid/tests/test_smoke.py`

- [ ] **Step 1: Initialize git repo (skip if already exists)**

```bash
cd /workspace/catalog-match/whaleshark-reid
git init -b main
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Editor
.ipynb_checkpoints/
.vscode/
.idea/

# Package state
cache/
*.db
*.db-journal
*.db-wal
*.db-shm
*.parquet

# OS
.DS_Store
```

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "whaleshark-reid"
version = "0.1.0"
description = "Whale shark re-identification pipeline and dev-mode review app"
requires-python = ">=3.10"
dependencies = [
  "pydantic>=2.0",
  "numpy",
  "pandas",
  "pyarrow",
  "scikit-learn",
  "hdbscan",
  "umap-learn",
  "transformers",
  "torch",
  "pillow",
  "wbia_miew_id @ file:///workspace/wbia-plugin-miew-id",
]

[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"
```

- [ ] **Step 4: Write the package `__init__.py` files**

`src/whaleshark_reid/__init__.py`:
```python
"""Whaleshark re-identification core package."""
__version__ = "0.1.0"
```

`src/whaleshark_reid/core/__init__.py`:
```python
"""Core engine: schema, ingest, embed, cluster, matching, feedback, metrics."""
```

`src/whaleshark_reid/storage/__init__.py`:
```python
"""Storage layer: SQLite + parquet caches."""
```

`tests/__init__.py`:
```python
```

- [ ] **Step 5: Write the shared conftest**

`tests/conftest.py`:
```python
"""Shared pytest fixtures for all tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory with embeddings/, clusters/, projections/, logs/ subdirs."""
    for sub in ("embeddings", "clusters", "projections", "logs"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite database path (file does not exist yet)."""
    return tmp_path / "state.db"
```

- [ ] **Step 6: Write the smoke test**

`tests/test_smoke.py`:
```python
def test_package_importable():
    import whaleshark_reid

    assert whaleshark_reid.__version__ == "0.1.0"


def test_core_submodule_importable():
    from whaleshark_reid import core  # noqa: F401


def test_storage_submodule_importable():
    from whaleshark_reid import storage  # noqa: F401
```

- [ ] **Step 7: Install the package in editable mode with dev extras**

```bash
cd /workspace/catalog-match/whaleshark-reid
pip install -e ".[dev]"
```

Expected: success. This pulls in `wbia_miew_id` from `/workspace/wbia-plugin-miew-id` as a file dependency. If the install errors on `wbia_miew_id`, verify `/workspace/wbia-plugin-miew-id/pyproject.toml` exists (it should — the package is already installed in the container).

- [ ] **Step 8: Run the smoke test to verify scaffolding**

Run: `pytest tests/test_smoke.py -v`
Expected: 3 passed.

- [ ] **Step 9: Write `README.md`**

```markdown
# whaleshark-reid

Whale shark re-identification pipeline and dev-mode review app.

Design specs live in `docs/superpowers/specs/`. Implementation plans live in `docs/superpowers/plans/`.

## Development

Install in editable mode:

    pip install -e ".[dev]"

Run tests:

    pytest -x

## Layout

- `src/whaleshark_reid/core/` — pure-Python core (schema, ingest, embed, cluster, matching, feedback, metrics)
- `src/whaleshark_reid/storage/` — SQLite + parquet state
- `tests/` — unit + integration tests
- `docs/superpowers/specs/` — design specs
- `docs/superpowers/plans/` — implementation plans
```

- [ ] **Step 10: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add pyproject.toml .gitignore README.md src/ tests/ docs/
git commit -m "chore: initial project scaffolding"
```

---

## Task 2: Schemas — Annotation, UUID helpers, result dataclasses

**Files:**
- Create: `src/whaleshark_reid/core/schema.py`
- Create: `tests/core/__init__.py`
- Create: `tests/core/test_schema.py`

- [ ] **Step 1: Write failing tests**

`tests/core/__init__.py`:
```python
```

`tests/core/test_schema.py`:
```python
"""Tests for core.schema — Annotation, UUIDs, result dataclasses."""
from __future__ import annotations

import uuid

import pytest

from whaleshark_reid.core.schema import (
    Annotation,
    ClusterResult,
    EmbedResult,
    INAT_NAMESPACE,
    IngestResult,
    MatchingResult,
    PairCandidate,
    ProjectionPoint,
    RebuildResult,
    inat_annotation_uuid,
    inat_image_uuid,
    new_name_uuid,
)


def test_inat_annotation_uuid_is_deterministic():
    u1 = inat_annotation_uuid(327286790, 0)
    u2 = inat_annotation_uuid(327286790, 0)
    assert u1 == u2
    assert uuid.UUID(u1)  # parseable


def test_inat_annotation_uuid_differs_by_photo_index():
    assert inat_annotation_uuid(327286790, 0) != inat_annotation_uuid(327286790, 1)


def test_inat_image_uuid_is_deterministic_and_distinct_from_annotation_uuid():
    ann = inat_annotation_uuid(327286790, 0)
    img = inat_image_uuid(327286790, 0)
    assert ann != img
    assert img == inat_image_uuid(327286790, 0)


def test_new_name_uuid_is_unique_uuid4():
    u1 = new_name_uuid()
    u2 = new_name_uuid()
    assert u1 != u2
    parsed = uuid.UUID(u1)
    assert parsed.version == 4


def test_inat_namespace_is_stable():
    assert str(INAT_NAMESPACE) == "6f4e6a5e-7b7a-4f3b-9c1d-1f0a2c3d4e5f"


def test_annotation_round_trip():
    ann = Annotation(
        annotation_uuid=inat_annotation_uuid(327286790, 0),
        image_uuid=inat_image_uuid(327286790, 0),
        source="inat",
        observation_id=327286790,
        photo_index=0,
        file_path="/tmp/327286790_1.jpg",
        file_name="327286790_1.jpg",
        bbox=[201.68, 4.49, 1349.56, 708.53],
        theta=0.0,
        viewpoint="unknown",
        species="whaleshark",
    )
    dumped = ann.model_dump()
    re = Annotation(**dumped)
    assert re == ann


def test_annotation_name_defaults_none():
    ann = Annotation(
        annotation_uuid=inat_annotation_uuid(1, 0),
        image_uuid=inat_image_uuid(1, 0),
        source="inat",
        file_path="/x.jpg",
        file_name="x.jpg",
        bbox=[0, 0, 10, 10],
    )
    assert ann.name is None
    assert ann.name_uuid is None


def test_pair_candidate_defaults():
    p = PairCandidate(
        ann_a_uuid="a",
        ann_b_uuid="b",
        distance=0.3,
    )
    assert p.same_cluster is False
    assert p.cluster_a is None


def test_cluster_result_holds_noise_label():
    r = ClusterResult(
        annotation_uuid="a",
        cluster_label=-1,
        cluster_algo="dbscan",
        cluster_params={"eps": 0.7},
    )
    assert r.cluster_label == -1


def test_projection_point_shape():
    p = ProjectionPoint(
        annotation_uuid="a",
        x=1.2,
        y=-3.4,
        algo="umap",
        params={"n_neighbors": 15},
    )
    assert p.x == 1.2


def test_ingest_result_fields():
    r = IngestResult(n_read=10, n_ingested=8, n_skipped_existing=2, n_missing_files=1)
    assert r.n_read == 10


def test_embed_result_fields():
    r = EmbedResult(
        n_embedded=10,
        n_skipped_existing=0,
        n_failed=0,
        embed_dim=512,
        model_id="conservationxlabs/miewid-msv3",
        duration_s=1.5,
    )
    assert r.embed_dim == 512


def test_matching_result_fields():
    r = MatchingResult(
        n_pairs=100,
        n_same_cluster=20,
        n_cross_cluster=80,
        n_filtered_out_by_decisions=5,
        median_distance=0.4,
        min_distance=0.1,
        max_distance=0.9,
        p10=0.15, p25=0.25, p50=0.4, p75=0.55, p90=0.8,
    )
    assert r.n_pairs == 100


def test_rebuild_result_fields():
    r = RebuildResult(n_components=3, n_singletons=5, n_annotations_updated=9)
    assert r.n_components == 3
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/core/test_schema.py -x`
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.core.schema'`

- [ ] **Step 3: Implement `core/schema.py`**

`src/whaleshark_reid/core/schema.py`:
```python
"""Core data schema — Annotation, UUID helpers, and stage result dataclasses.

All primary identifiers are UUIDs (strings). Integer IDs from source systems
(iNat observation_id, COCO id) are preserved as separate traceability columns
but never used as keys — this is what makes multi-catalog merging collision-safe.
"""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field

# Fixed namespace for deterministic uuid5() generation on iNat sources.
# Stable across ingest runs so re-running ingest is idempotent.
INAT_NAMESPACE = uuid.UUID("6f4e6a5e-7b7a-4f3b-9c1d-1f0a2c3d4e5f")


def inat_annotation_uuid(observation_id: int, photo_index: int) -> str:
    return str(uuid.uuid5(INAT_NAMESPACE, f"inat:annotation:{observation_id}:{photo_index}"))


def inat_image_uuid(observation_id: int, photo_index: int) -> str:
    return str(uuid.uuid5(INAT_NAMESPACE, f"inat:image:{observation_id}:{photo_index}"))


def new_name_uuid() -> str:
    """Fresh uuid4 for a derived individual. Called by feedback rebuild."""
    return str(uuid.uuid4())


class Annotation(BaseModel):
    # --- Canonical UUID identifiers (primary) ---
    annotation_uuid: str
    image_uuid: str
    name_uuid: Optional[str] = None

    # --- Source reference IDs (traceability only) ---
    source: str
    source_annotation_id: Optional[str] = None
    source_image_id: Optional[str] = None
    source_individual_id: Optional[str] = None
    observation_id: Optional[int] = None
    photo_index: Optional[int] = None

    # --- MiewID-required fields (names match MiewIdDataset.__getitem__) ---
    file_path: str
    file_name: str
    bbox: list[float] = Field(..., description="[x, y, w, h]")
    theta: float = 0.0
    viewpoint: str = "unknown"
    species: str = "whaleshark"
    name: Optional[str] = None

    # --- Provenance / dev-mode display ---
    photographer: Optional[str] = None
    license: Optional[str] = None
    date_captured: Optional[str] = None
    gps_lat_captured: Optional[float] = None
    gps_lon_captured: Optional[float] = None
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    conf: Optional[float] = None
    quality_grade: Optional[str] = None

    # --- Derived ---
    name_viewpoint: Optional[str] = None
    species_viewpoint: Optional[str] = None


class PairCandidate(BaseModel):
    ann_a_uuid: str
    ann_b_uuid: str
    distance: float
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    same_cluster: bool = False


class ClusterResult(BaseModel):
    annotation_uuid: str
    cluster_label: int
    cluster_algo: str
    cluster_params: dict


class ProjectionPoint(BaseModel):
    annotation_uuid: str
    x: float
    y: float
    algo: str
    params: dict


# --- Stage result dataclasses (returned by core.*.run_*_stage, attached to runs.metrics_json) ---

class IngestResult(BaseModel):
    n_read: int
    n_ingested: int
    n_skipped_existing: int
    n_missing_files: int


class EmbedResult(BaseModel):
    n_embedded: int
    n_skipped_existing: int
    n_failed: int
    embed_dim: int
    model_id: str
    duration_s: float


class ClusterStageResult(BaseModel):
    algo: str
    n_clusters: int
    n_noise: int
    largest_cluster_size: int
    singleton_fraction: float
    median_cluster_size: float


class ProjectStageResult(BaseModel):
    algo: str
    n_points: int
    params: dict


class MatchingResult(BaseModel):
    n_pairs: int
    n_same_cluster: int
    n_cross_cluster: int
    n_filtered_out_by_decisions: int
    median_distance: float
    min_distance: float
    max_distance: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class RebuildResult(BaseModel):
    n_components: int
    n_singletons: int
    n_annotations_updated: int
```

- [ ] **Step 4: Run tests — verify pass**

Run: `pytest tests/core/test_schema.py -x`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/schema.py tests/core/
git commit -m "feat: core schema — Annotation, UUID helpers, stage result dataclasses"
```

---

## Task 3: SQLite schema.sql + Storage base class

**Files:**
- Create: `src/whaleshark_reid/storage/schema.sql`
- Create: `src/whaleshark_reid/storage/db.py`
- Create: `tests/storage/__init__.py`
- Create: `tests/storage/test_db_schema.py`

- [ ] **Step 1: Write the failing schema tests**

`tests/storage/__init__.py`:
```python
```

`tests/storage/test_db_schema.py`:
```python
"""Tests for storage.db — SQLite schema creation and PRAGMAs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from whaleshark_reid.storage.db import Storage


def _fetch_table_names(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return {r[0] for r in rows}


def test_storage_creates_file(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    assert tmp_db_path.exists()


def test_storage_has_expected_tables(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    conn = sqlite3.connect(tmp_db_path)
    try:
        tables = _fetch_table_names(conn)
    finally:
        conn.close()
    expected = {"annotations", "pair_decisions", "runs", "pair_queue"}
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"
    # Explicitly verify the dropped tables are NOT present.
    assert "individuals_cache" not in tables
    assert "name_uuid_history" not in tables
    assert "experiments" not in tables


def test_storage_has_wal_mode(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    mode = storage.conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"


def test_storage_has_foreign_keys_on(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    on = storage.conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert on == 1


def test_runs_has_metrics_json_column(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    cols = {
        row[1] for row in storage.conn.execute("PRAGMA table_info(runs)").fetchall()
    }
    assert "metrics_json" in cols
    assert "parent_run_id" not in cols


def test_annotations_has_uuid_primary_key(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    cols_with_pk = [
        (row[1], row[5])
        for row in storage.conn.execute("PRAGMA table_info(annotations)").fetchall()
    ]
    pk_cols = [name for name, pk in cols_with_pk if pk > 0]
    assert pk_cols == ["annotation_uuid"]


def test_init_schema_is_idempotent(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.init_schema()  # second call is a no-op thanks to IF NOT EXISTS
    # still has the same tables, no error
    conn = sqlite3.connect(tmp_db_path)
    try:
        tables = _fetch_table_names(conn)
    finally:
        conn.close()
    assert "annotations" in tables
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/storage/test_db_schema.py -x`
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.storage.db'`

- [ ] **Step 3: Write `storage/schema.sql`**

`src/whaleshark_reid/storage/schema.sql`:
```sql
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS annotations (
    annotation_uuid TEXT PRIMARY KEY,
    image_uuid TEXT NOT NULL,
    name_uuid TEXT,

    source TEXT NOT NULL,
    source_annotation_id TEXT,
    source_image_id TEXT,
    source_individual_id TEXT,
    observation_id INTEGER,
    photo_index INTEGER,

    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    bbox_x REAL NOT NULL,
    bbox_y REAL NOT NULL,
    bbox_w REAL NOT NULL,
    bbox_h REAL NOT NULL,
    theta REAL NOT NULL DEFAULT 0.0,
    viewpoint TEXT NOT NULL DEFAULT 'unknown',
    species TEXT NOT NULL DEFAULT 'whaleshark',
    name TEXT,

    photographer TEXT,
    license TEXT,
    date_captured TEXT,
    gps_lat_captured REAL,
    gps_lon_captured REAL,
    coco_url TEXT,
    flickr_url TEXT,
    height INTEGER,
    width INTEGER,
    conf REAL,
    quality_grade TEXT,

    name_viewpoint TEXT,
    species_viewpoint TEXT,

    ingested_run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(source, observation_id, photo_index)
);
CREATE INDEX IF NOT EXISTS idx_ann_source ON annotations(source);
CREATE INDEX IF NOT EXISTS idx_ann_obs ON annotations(observation_id);
CREATE INDEX IF NOT EXISTS idx_ann_image_uuid ON annotations(image_uuid);
CREATE INDEX IF NOT EXISTS idx_ann_name_uuid ON annotations(name_uuid) WHERE name_uuid IS NOT NULL;

CREATE TABLE IF NOT EXISTS pair_decisions (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ann_a_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    ann_b_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    decision TEXT NOT NULL CHECK(decision IN ('match','no_match','skip','unsure')),
    distance REAL,
    run_id TEXT,
    user TEXT NOT NULL DEFAULT 'dev',
    notes TEXT,
    created_at TEXT NOT NULL,
    superseded_by INTEGER REFERENCES pair_decisions(decision_id)
);
CREATE INDEX IF NOT EXISTS idx_pd_ab ON pair_decisions(ann_a_uuid, ann_b_uuid);
CREATE INDEX IF NOT EXISTS idx_pd_run ON pair_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_pd_active ON pair_decisions(decision) WHERE superseded_by IS NULL;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT,
    notes TEXT,
    git_sha TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running','ok','failed')),
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_stage ON runs(stage);
CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at);

CREATE TABLE IF NOT EXISTS pair_queue (
    queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    ann_a_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    ann_b_uuid TEXT NOT NULL REFERENCES annotations(annotation_uuid),
    distance REAL NOT NULL,
    cluster_a INTEGER,
    cluster_b INTEGER,
    same_cluster INTEGER NOT NULL DEFAULT 0,
    position INTEGER NOT NULL,
    UNIQUE(run_id, ann_a_uuid, ann_b_uuid)
);
CREATE INDEX IF NOT EXISTS idx_pq_run_pos ON pair_queue(run_id, position);
```

- [ ] **Step 4: Implement `storage/db.py` — Storage class skeleton**

`src/whaleshark_reid/storage/db.py`:
```python
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
```

- [ ] **Step 5: Make the schema.sql file discoverable by setuptools**

Update `pyproject.toml` to include the SQL file as package data:

```toml
[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
whaleshark_reid = ["storage/schema.sql"]
```

Then reinstall:

```bash
cd /workspace/catalog-match/whaleshark-reid
pip install -e ".[dev]"
```

- [ ] **Step 6: Run tests — verify pass**

Run: `pytest tests/storage/test_db_schema.py -x`
Expected: all 7 tests pass.

- [ ] **Step 7: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/storage/ tests/storage/ pyproject.toml
git commit -m "feat: SQLite schema and Storage base class"
```

---

## Task 4: Storage — annotation CRUD

**Files:**
- Modify: `src/whaleshark_reid/storage/db.py` (add methods)
- Create: `tests/storage/test_db_annotations.py`

- [ ] **Step 1: Write the failing tests**

`tests/storage/test_db_annotations.py`:
```python
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
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/storage/test_db_annotations.py -x`
Expected: `AttributeError: 'Storage' object has no attribute 'upsert_annotation'`

- [ ] **Step 3: Add annotation CRUD methods to Storage**

Append to `src/whaleshark_reid/storage/db.py`:

```python
    # ----- annotation CRUD -----

    def upsert_annotation(self, ann, run_id: str) -> None:
        """INSERT OR IGNORE an Annotation row. Does nothing if
        (source, observation_id, photo_index) already exists."""
        from datetime import datetime, timezone

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
```

- [ ] **Step 4: Run tests — verify pass**

Run: `pytest tests/storage/test_db_annotations.py -x`
Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/storage/db.py tests/storage/test_db_annotations.py
git commit -m "feat: Storage annotation CRUD"
```

---

## Task 5: Storage — runs CRUD (begin/finish/get status)

**Files:**
- Modify: `src/whaleshark_reid/storage/db.py`
- Create: `tests/storage/test_db_runs.py`

- [ ] **Step 1: Write the failing tests**

`tests/storage/test_db_runs.py`:
```python
"""Tests for runs CRUD on Storage."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from whaleshark_reid.storage.db import Storage


def test_begin_run_creates_row_with_running_status(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    storage.begin_run(run_id="run_test_1", stage="ingest", config={"csv": "/a"})
    row = storage.conn.execute(
        "SELECT status, config_json, metrics_json, finished_at FROM runs WHERE run_id = ?",
        ("run_test_1",),
    ).fetchone()

    assert row["status"] == "running"
    assert json.loads(row["config_json"]) == {"csv": "/a"}
    assert row["metrics_json"] is None
    assert row["finished_at"] is None


def test_finish_run_ok_populates_metrics(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.begin_run(run_id="r1", stage="embed", config={"batch_size": 32})

    storage.finish_run(run_id="r1", status="ok", metrics={"n_embedded": 42}, notes="nominal")

    row = storage.conn.execute(
        "SELECT status, metrics_json, notes, finished_at, error FROM runs WHERE run_id = ?",
        ("r1",),
    ).fetchone()
    assert row["status"] == "ok"
    assert json.loads(row["metrics_json"]) == {"n_embedded": 42}
    assert row["notes"] == "nominal"
    assert row["finished_at"] is not None
    assert row["error"] is None


def test_finish_run_failed_populates_error(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.begin_run(run_id="r1", stage="cluster", config={"eps": 0.7})

    storage.finish_run(run_id="r1", status="failed", metrics={}, error="boom")

    row = storage.conn.execute(
        "SELECT status, error FROM runs WHERE run_id = ?",
        ("r1",),
    ).fetchone()
    assert row["status"] == "failed"
    assert row["error"] == "boom"


def test_get_run_status_transitions(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    storage.begin_run(run_id="r1", stage="ingest", config={})
    assert storage.get_run_status("r1") == "running"

    storage.finish_run(run_id="r1", status="ok", metrics={})
    assert storage.get_run_status("r1") == "ok"


def test_get_run_status_missing_returns_none(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    assert storage.get_run_status("nope") is None
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/storage/test_db_runs.py -x`
Expected: `AttributeError: 'Storage' object has no attribute 'begin_run'`

- [ ] **Step 3: Add runs CRUD methods to Storage**

Append to `src/whaleshark_reid/storage/db.py`:

```python
    # ----- runs CRUD -----

    def begin_run(self, run_id: str, stage: str, config: dict, git_sha: str | None = None) -> None:
        import json
        from datetime import datetime, timezone

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
        import json
        from datetime import datetime, timezone

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
```

- [ ] **Step 4: Run tests — verify pass**

Run: `pytest tests/storage/test_db_runs.py -x`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/storage/db.py tests/storage/test_db_runs.py
git commit -m "feat: Storage runs CRUD (begin/finish/get_status)"
```

---

## Task 6: Parquet cache helpers (embedding, cluster, projection)

**Files:**
- Create: `src/whaleshark_reid/storage/embedding_cache.py`
- Create: `src/whaleshark_reid/storage/cluster_cache.py`
- Create: `src/whaleshark_reid/storage/projection_cache.py`
- Create: `tests/storage/test_embedding_cache.py`
- Create: `tests/storage/test_cluster_cache.py`
- Create: `tests/storage/test_projection_cache.py`

- [ ] **Step 1: Write failing tests for embedding cache**

`tests/storage/test_embedding_cache.py`:
```python
"""Tests for parquet-backed embedding cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.storage.embedding_cache import (
    existing_annotation_uuids,
    read_embeddings,
    read_embeddings_as_array,
    write_embeddings,
)


def test_write_and_read_embeddings_roundtrip(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": f"uuid-{i}",
            "embedding": [0.1 * i, 0.2 * i, 0.3 * i],
            "model_id": "miewid-msv3",
            "model_version": "msv3",
            "created_at": "2026-04-10T00:00:00Z",
        }
        for i in range(5)
    ]

    out_path = write_embeddings(tmp_cache_dir, "run_abc", rows)
    assert out_path.exists()

    df = read_embeddings(tmp_cache_dir, "run_abc")
    assert len(df) == 5
    assert set(df.columns) == {"annotation_uuid", "embedding", "model_id", "model_version", "created_at"}


def test_read_embeddings_as_array(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": "a",
            "embedding": [1.0, 2.0, 3.0],
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        },
        {
            "annotation_uuid": "b",
            "embedding": [4.0, 5.0, 6.0],
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        },
    ]
    write_embeddings(tmp_cache_dir, "run_abc", rows)

    uuids, mat = read_embeddings_as_array(tmp_cache_dir, "run_abc")
    assert uuids == ["a", "b"]
    assert mat.shape == (2, 3)
    assert np.allclose(mat[0], [1.0, 2.0, 3.0])


def test_existing_annotation_uuids(tmp_cache_dir: Path):
    rows = [
        {"annotation_uuid": f"u{i}", "embedding": [0.0], "model_id": "m",
         "model_version": "v", "created_at": "t"}
        for i in range(3)
    ]
    write_embeddings(tmp_cache_dir, "run_abc", rows)

    assert existing_annotation_uuids(tmp_cache_dir, "run_abc") == {"u0", "u1", "u2"}


def test_existing_annotation_uuids_empty_when_no_file(tmp_cache_dir: Path):
    assert existing_annotation_uuids(tmp_cache_dir, "run_missing") == set()
```

- [ ] **Step 2: Write failing tests for cluster cache**

`tests/storage/test_cluster_cache.py`:
```python
"""Tests for parquet-backed cluster label cache."""
from __future__ import annotations

from pathlib import Path

from whaleshark_reid.storage.cluster_cache import read_clusters, write_clusters


def test_write_and_read_clusters(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": "a",
            "cluster_label": 0,
            "cluster_algo": "dbscan",
            "cluster_params_json": '{"eps": 0.7}',
        },
        {
            "annotation_uuid": "b",
            "cluster_label": -1,
            "cluster_algo": "dbscan",
            "cluster_params_json": '{"eps": 0.7}',
        },
    ]
    write_clusters(tmp_cache_dir, "run_abc", rows)

    df = read_clusters(tmp_cache_dir, "run_abc")
    assert len(df) == 2
    assert set(df["cluster_label"]) == {0, -1}
```

- [ ] **Step 3: Write failing tests for projection cache**

`tests/storage/test_projection_cache.py`:
```python
"""Tests for parquet-backed UMAP projection cache."""
from __future__ import annotations

from pathlib import Path

from whaleshark_reid.storage.projection_cache import read_projections, write_projections


def test_write_and_read_projections(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": "a",
            "x": 1.2,
            "y": -3.4,
            "algo": "umap",
            "params_json": '{"n_neighbors": 15}',
        },
    ]
    write_projections(tmp_cache_dir, "run_abc", rows)

    df = read_projections(tmp_cache_dir, "run_abc")
    assert len(df) == 1
    assert df.iloc[0]["x"] == 1.2
```

- [ ] **Step 4: Run tests — verify all three fail**

Run: `pytest tests/storage/test_embedding_cache.py tests/storage/test_cluster_cache.py tests/storage/test_projection_cache.py -x`
Expected: `ModuleNotFoundError` for each.

- [ ] **Step 5: Implement `embedding_cache.py`**

`src/whaleshark_reid/storage/embedding_cache.py`:
```python
"""Parquet-backed embedding cache.

One file per run: cache_dir/embeddings/<run_id>.parquet
Columns: annotation_uuid, embedding (list[float]), model_id, model_version, created_at
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _embeddings_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / "embeddings" / f"{run_id}.parquet"


def write_embeddings(cache_dir: Path, run_id: str, rows: Iterable[dict]) -> Path:
    out = _embeddings_path(cache_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


def read_embeddings(cache_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_embeddings_path(cache_dir, run_id), engine="pyarrow")


def read_embeddings_as_array(cache_dir: Path, run_id: str) -> tuple[list[str], np.ndarray]:
    df = read_embeddings(cache_dir, run_id)
    uuids = df["annotation_uuid"].tolist()
    mat = np.vstack([np.asarray(e, dtype=np.float32) for e in df["embedding"]])
    return uuids, mat


def existing_annotation_uuids(cache_dir: Path, run_id: str) -> set[str]:
    path = _embeddings_path(cache_dir, run_id)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, engine="pyarrow", columns=["annotation_uuid"])
    return set(df["annotation_uuid"].tolist())
```

- [ ] **Step 6: Implement `cluster_cache.py`**

`src/whaleshark_reid/storage/cluster_cache.py`:
```python
"""Parquet-backed cluster label cache.

One file per run: cache_dir/clusters/<run_id>.parquet
Columns: annotation_uuid, cluster_label, cluster_algo, cluster_params_json
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _clusters_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / "clusters" / f"{run_id}.parquet"


def write_clusters(cache_dir: Path, run_id: str, rows: Iterable[dict]) -> Path:
    out = _clusters_path(cache_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(out, engine="pyarrow", index=False)
    return out


def read_clusters(cache_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_clusters_path(cache_dir, run_id), engine="pyarrow")
```

- [ ] **Step 7: Implement `projection_cache.py`**

`src/whaleshark_reid/storage/projection_cache.py`:
```python
"""Parquet-backed 2D projection cache (UMAP/t-SNE).

One file per run: cache_dir/projections/<run_id>.parquet
Columns: annotation_uuid, x, y, algo, params_json
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _projections_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / "projections" / f"{run_id}.parquet"


def write_projections(cache_dir: Path, run_id: str, rows: Iterable[dict]) -> Path:
    out = _projections_path(cache_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(out, engine="pyarrow", index=False)
    return out


def read_projections(cache_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_projections_path(cache_dir, run_id), engine="pyarrow")
```

- [ ] **Step 8: Run tests — verify all three pass**

Run: `pytest tests/storage/ -x`
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/storage/embedding_cache.py src/whaleshark_reid/storage/cluster_cache.py src/whaleshark_reid/storage/projection_cache.py tests/storage/test_embedding_cache.py tests/storage/test_cluster_cache.py tests/storage/test_projection_cache.py
git commit -m "feat: parquet caches for embeddings, clusters, projections"
```

---

## Task 7: Ingest — iNat CSV parser

**Files:**
- Create: `src/whaleshark_reid/core/ingest/__init__.py`
- Create: `src/whaleshark_reid/core/ingest/inat.py`
- Create: `tests/fixtures/__init__.py`
- Create: `tests/fixtures/mini_inat.csv`
- Create: `tests/fixtures/mini_inat_rich.csv`
- Create: `tests/fixtures/photos/` (10 tiny JPEGs)
- Create: `tests/core/test_ingest_inat.py`

- [ ] **Step 1: Create the minimal fixture CSV (10 rows)**

`tests/fixtures/mini_inat.csv`:
```csv
,theta,viewpoint,name,file_name,species,file_path,x,y,w,h
0,0,unknown,unknown,obs100_1.jpg,whaleshark,/stale/path/obs100_1.jpg,10.0,20.0,100.0,200.0
1,0,unknown,unknown,obs101_1.jpg,whaleshark,/stale/path/obs101_1.jpg,15.0,25.0,120.0,180.0
2,0,unknown,unknown,obs102_1.jpg,whaleshark,/stale/path/obs102_1.jpg,5.0,10.0,80.0,160.0
3,0,unknown,unknown,obs103_1.jpg,whaleshark,/stale/path/obs103_1.jpg,20.0,30.0,150.0,220.0
4,0,unknown,unknown,obs104_1.jpg,whaleshark,/stale/path/obs104_1.jpg,25.0,35.0,110.0,190.0
5,0,unknown,unknown,obs105_1.jpg,whaleshark,/stale/path/obs105_1.jpg,30.0,40.0,130.0,210.0
6,0,unknown,unknown,obs106_1.jpg,whaleshark,/stale/path/obs106_1.jpg,12.0,22.0,105.0,170.0
7,0,unknown,unknown,obs107_1.jpg,whaleshark,/stale/path/obs107_1.jpg,18.0,28.0,140.0,200.0
8,0,unknown,unknown,obs108_1.jpg,whaleshark,/stale/path/obs108_1.jpg,22.0,32.0,115.0,185.0
9,0,unknown,unknown,obs109_1.jpg,whaleshark,/stale/path/obs109_1.jpg,8.0,18.0,95.0,165.0
```

- [ ] **Step 2: Create the rich fixture CSV (10 rows, subset of columns)**

`tests/fixtures/mini_inat_rich.csv`:
```csv
observation_id,photographer,license,date_captured,gps_lat_captured,gps_lon_captured,coco_url,height,width,conf,quality_grade
100,alice,CC-BY,2025-11-19,25.66,-80.17,https://inat/100,1080,1920,0.94,research
101,bob,CC-BY,2025-11-20,26.11,-80.12,https://inat/101,1080,1920,0.91,research
102,cara,CC0,2025-11-21,25.90,-80.05,https://inat/102,720,1280,0.88,research
103,dan,CC-BY,2025-11-22,24.90,-80.10,https://inat/103,1080,1920,0.77,casual
104,eva,CC-BY,2025-11-23,25.50,-80.20,https://inat/104,1080,1920,0.93,research
105,fran,CC0,2025-11-24,26.00,-80.00,https://inat/105,1080,1920,0.86,research
106,gus,CC-BY,2025-11-25,25.70,-80.15,https://inat/106,1080,1920,0.80,research
107,hal,CC-BY,2025-11-26,25.80,-80.18,https://inat/107,720,1280,0.79,casual
108,iva,CC-BY,2025-11-27,25.95,-80.08,https://inat/108,1080,1920,0.90,research
109,jon,CC-BY,2025-11-28,25.85,-80.12,https://inat/109,1080,1920,0.82,research
```

- [ ] **Step 3: Create 10 tiny fixture JPEGs**

```bash
cd /workspace/catalog-match/whaleshark-reid
mkdir -p tests/fixtures/photos
python -c "
from PIL import Image
import numpy as np
for i in range(10):
    obs_id = 100 + i
    # 64x64 random image per obs
    arr = (np.random.rand(64, 64, 3) * 255).astype('uint8')
    Image.fromarray(arr).save(f'tests/fixtures/photos/obs{obs_id}_1.jpg', quality=60)
print('done')
"
```

Expected output: `done`. Verify with `ls tests/fixtures/photos/` — 10 JPEGs.

- [ ] **Step 4: Write the failing ingest tests**

`src/whaleshark_reid/core/ingest/__init__.py`:
```python
```

`tests/core/test_ingest_inat.py`:
```python
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
```

- [ ] **Step 5: Run tests — verify failure**

Run: `pytest tests/core/test_ingest_inat.py -x`
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.core.ingest.inat'`

- [ ] **Step 6: Implement `core/ingest/inat.py`**

`src/whaleshark_reid/core/ingest/inat.py`:
```python
"""iNat CSV ingest.

Reads either the minimal 10-column schema (theta, viewpoint, name, file_name,
species, file_path, x, y, w, h) or the richer dfx schema (41 columns with
provenance). Auto-detects by presence of 'observation_id' column.

For Phase 1 cold-start, 'name' is always 'unknown' → stored as NULL. Viewpoint
stays as 'unknown' (it is categorical, not an identity).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from whaleshark_reid.core.schema import (
    Annotation,
    IngestResult,
    inat_annotation_uuid,
    inat_image_uuid,
)

# Filename patterns like "327286790_1.jpg" → observation_id=327286790, photo_index=0 (1-based → 0-based)
_FILENAME_RE = re.compile(r"^(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _parse_obs_and_index_from_filename(filename: str) -> tuple[int, int]:
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match <obs_id>_<n>.jpg pattern: {filename}")
    return int(m.group(1)), int(m.group(2)) - 1  # 1-based → 0-based


def _normalize_name(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    return None if s.lower() in ("", "unknown", "nan") else s


def ingest_inat_csv(
    csv_path: Path,
    photos_dir: Path,
    storage,
    run_id: str,
    rich_csv_path: Optional[Path] = None,
) -> IngestResult:
    df = pd.read_csv(csv_path)

    rich_df: Optional[pd.DataFrame] = None
    if rich_csv_path is not None:
        rich_df = pd.read_csv(rich_csv_path).set_index("observation_id")

    n_read = len(df)
    n_ingested = 0
    n_skipped_existing = 0
    n_missing_files = 0

    for _, row in df.iterrows():
        file_name = str(row["file_name"])
        obs_id, photo_index = _parse_obs_and_index_from_filename(file_name)

        resolved_path = photos_dir / file_name
        if not resolved_path.exists():
            n_missing_files += 1

        # Base fields from the minimal CSV
        ann_kwargs = dict(
            annotation_uuid=inat_annotation_uuid(obs_id, photo_index),
            image_uuid=inat_image_uuid(obs_id, photo_index),
            source="inat",
            observation_id=obs_id,
            photo_index=photo_index,
            file_path=str(resolved_path),
            file_name=file_name,
            bbox=[float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
            theta=float(row.get("theta", 0.0) or 0.0),
            viewpoint=str(row.get("viewpoint", "unknown")) or "unknown",
            species=str(row.get("species", "whaleshark")) or "whaleshark",
            name=_normalize_name(row.get("name")),
        )

        # Backfill provenance from rich CSV if available
        if rich_df is not None and obs_id in rich_df.index:
            r = rich_df.loc[obs_id]
            for col, field in [
                ("photographer", "photographer"),
                ("license", "license"),
                ("date_captured", "date_captured"),
                ("gps_lat_captured", "gps_lat_captured"),
                ("gps_lon_captured", "gps_lon_captured"),
                ("coco_url", "coco_url"),
                ("flickr_url", "flickr_url"),
                ("height", "height"),
                ("width", "width"),
                ("conf", "conf"),
                ("quality_grade", "quality_grade"),
            ]:
                if col in r and not (isinstance(r[col], float) and pd.isna(r[col])):
                    val = r[col]
                    if field in ("height", "width"):
                        val = int(val)
                    elif field in ("gps_lat_captured", "gps_lon_captured", "conf"):
                        val = float(val)
                    ann_kwargs[field] = val

        # Derived composite keys (used later by Wildbook stratification; cheap to fill now)
        ann_kwargs["name_viewpoint"] = f"{ann_kwargs['name']}_{ann_kwargs['viewpoint']}"
        ann_kwargs["species_viewpoint"] = f"{ann_kwargs['species']}_{ann_kwargs['viewpoint']}"

        ann = Annotation(**ann_kwargs)

        before = storage.count("annotations")
        storage.upsert_annotation(ann, run_id=run_id)
        after = storage.count("annotations")
        if after > before:
            n_ingested += 1
        else:
            n_skipped_existing += 1

    return IngestResult(
        n_read=n_read,
        n_ingested=n_ingested,
        n_skipped_existing=n_skipped_existing,
        n_missing_files=n_missing_files,
    )
```

- [ ] **Step 7: Run tests — verify pass**

Run: `pytest tests/core/test_ingest_inat.py -x`
Expected: all 7 tests pass.

- [ ] **Step 8: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/ingest/ tests/core/test_ingest_inat.py tests/fixtures/
git commit -m "feat: iNat CSV ingest with rich provenance backfill"
```

---

## Task 8: Embed — `embed_annotations` via wbia_miew_id

**Files:**
- Create: `src/whaleshark_reid/core/embed/__init__.py`
- Create: `src/whaleshark_reid/core/embed/miewid.py`
- Create: `tests/core/test_embed_miewid.py`

- [ ] **Step 1: Write failing tests with a stubbed MiewID**

`src/whaleshark_reid/core/embed/__init__.py`:
```python
```

`tests/core/test_embed_miewid.py`:
```python
"""Tests for core.embed.miewid with a stubbed HF model.

We monkeypatch transformers.AutoModel so the test suite does NOT hit HuggingFace
or require a GPU. The stub returns deterministic pseudo-embeddings so we can
verify pipeline plumbing (DataFrame construction, DataLoader, return shape).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from whaleshark_reid.core.schema import (
    Annotation,
    inat_annotation_uuid,
    inat_image_uuid,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


class _StubMiewId(torch.nn.Module):
    """Deterministic stand-in for MiewIdNet.

    extract_feat(x) returns a (B, D) tensor where each row is the mean
    of the corresponding image tensor, broadcast to D=8. Deterministic so
    tests can assert specific values.
    """

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        # A single parameter so .to(device) works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        per_sample = x.mean(dim=(1, 2, 3))  # (B,)
        return per_sample.unsqueeze(1).expand(-1, self.embed_dim)  # (B, 8)

    def forward(self, x, label=None):
        return self.extract_feat(x)


@pytest.fixture
def stub_miewid(monkeypatch):
    from transformers import AutoModel

    def _fake_from_pretrained(*args, **kwargs):
        return _StubMiewId()

    monkeypatch.setattr(AutoModel, "from_pretrained", _fake_from_pretrained)
    yield


def _make_ann(obs_id: int, image_path: Path) -> Annotation:
    return Annotation(
        annotation_uuid=inat_annotation_uuid(obs_id, 0),
        image_uuid=inat_image_uuid(obs_id, 0),
        source="inat",
        observation_id=obs_id,
        photo_index=0,
        file_path=str(image_path),
        file_name=image_path.name,
        bbox=[0.0, 0.0, 64.0, 64.0],
        theta=0.0,
        viewpoint="unknown",
        species="whaleshark",
    )


def test_embed_annotations_returns_correct_shape(stub_miewid):
    from whaleshark_reid.core.embed.miewid import embed_annotations

    anns = [
        _make_ann(100 + i, FIXTURES / "photos" / f"obs{100 + i}_1.jpg")
        for i in range(5)
    ]
    mat = embed_annotations(anns, batch_size=2, num_workers=0, device="cpu")
    assert mat.shape == (5, 8)
    assert not np.isnan(mat).any()


def test_embed_annotations_deterministic_for_same_input(stub_miewid):
    from whaleshark_reid.core.embed.miewid import embed_annotations

    anns = [_make_ann(100, FIXTURES / "photos" / "obs100_1.jpg")]
    a = embed_annotations(anns, batch_size=1, num_workers=0, device="cpu")
    b = embed_annotations(anns, batch_size=1, num_workers=0, device="cpu")
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/core/test_embed_miewid.py -x`
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.core.embed.miewid'`

- [ ] **Step 3: Implement `core/embed/miewid.py` with full reuse of wbia_miew_id**

`src/whaleshark_reid/core/embed/miewid.py`:
```python
"""MiewID embedding extraction via in-process wbia_miew_id reuse.

This module owns almost nothing: it just builds a DataFrame from a list of
Annotations, hands it to wbia_miew_id's MiewIdDataset, runs the canonical
extract_embeddings() loop from wbia_miew_id.engine.eval_fn, and returns a
numpy array of shape (N, embed_dim).

No custom inference loop. No custom transforms. No custom get_chip_from_img
call — MiewIdDataset handles all of that internally. This is critical: the
repo uses Albumentations transforms (not torchvision) and any drift here would
silently change embedding quality from the benchmark notebook's outputs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from whaleshark_reid.core.schema import Annotation, EmbedResult


def embed_annotations(
    annotations: list[Annotation],
    model_id: str = "conservationxlabs/miewid-msv3",
    image_size: tuple[int, int] = (440, 440),
    batch_size: int = 32,
    num_workers: int = 2,
    use_bbox: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract MiewID embeddings for a list of annotations.

    Returns np.ndarray of shape (N, embed_dim) float32. Preserves input order.
    """
    # Imported lazily so the test stub can monkeypatch AutoModel before import-time side effects.
    from transformers import AutoModel
    from wbia_miew_id.datasets import MiewIdDataset, get_test_transforms
    from wbia_miew_id.engine.eval_fn import extract_embeddings as _extract_embeddings

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.DataFrame(
        [
            {
                "file_path": ann.file_path,
                "bbox": ann.bbox,
                "theta": ann.theta,
                "name": i,            # dummy int label — MiewIdDataset casts this to a tensor
                "species": ann.species,
                "viewpoint": ann.viewpoint,
            }
            for i, ann in enumerate(annotations)
        ]
    )

    dataset = MiewIdDataset(
        csv=df,
        transforms=get_test_transforms(image_size),
        crop_bbox=use_bbox,
        fliplr=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    embeddings, _labels = _extract_embeddings(loader, model, device)
    return np.asarray(embeddings, dtype=np.float32)


def run_embed_stage(
    storage,
    cache_dir: Path,
    run_id: str,
    model_id: str = "conservationxlabs/miewid-msv3",
    batch_size: int = 32,
    num_workers: int = 2,
    use_bbox: bool = True,
    only_missing: bool = True,
    device: Optional[str] = None,
) -> EmbedResult:
    """End-to-end embed stage: load annotations → embed → write parquet → return metrics."""
    from datetime import datetime, timezone

    from whaleshark_reid.storage.embedding_cache import (
        existing_annotation_uuids,
        write_embeddings,
    )

    all_anns = storage.list_annotations()
    if only_missing:
        existing = existing_annotation_uuids(cache_dir, run_id)
        to_embed = [a for a in all_anns if a.annotation_uuid not in existing]
    else:
        to_embed = all_anns
    n_skipped = len(all_anns) - len(to_embed)

    if not to_embed:
        return EmbedResult(
            n_embedded=0,
            n_skipped_existing=n_skipped,
            n_failed=0,
            embed_dim=0,
            model_id=model_id,
            duration_s=0.0,
        )

    t0 = time.time()
    mat = embed_annotations(
        to_embed,
        model_id=model_id,
        batch_size=batch_size,
        num_workers=num_workers,
        use_bbox=use_bbox,
        device=device,
    )
    duration = time.time() - t0

    rows = [
        {
            "annotation_uuid": ann.annotation_uuid,
            "embedding": mat[i].tolist(),
            "model_id": model_id,
            "model_version": "msv3",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for i, ann in enumerate(to_embed)
    ]
    write_embeddings(cache_dir, run_id, rows)

    return EmbedResult(
        n_embedded=len(to_embed),
        n_skipped_existing=n_skipped,
        n_failed=0,
        embed_dim=int(mat.shape[1]),
        model_id=model_id,
        duration_s=duration,
    )
```

- [ ] **Step 4: Run tests — verify pass**

Run: `pytest tests/core/test_embed_miewid.py -x`
Expected: 2 tests pass. (If you hit import-time errors from wbia_miew_id pulling in wandb, add a pytest fixture that monkeypatches wandb — unlikely but possible. The repo's eval_fn imports wandb at module level; if that fails here, install `wandb` as a transitive test dep.)

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/embed/ tests/core/test_embed_miewid.py
git commit -m "feat: embed_annotations via wbia_miew_id reuse"
```

---

## Task 9: Cluster — DBSCAN + HDBSCAN + run_cluster_stage

**Files:**
- Create: `src/whaleshark_reid/core/cluster/__init__.py`
- Create: `src/whaleshark_reid/core/cluster/dbscan.py`
- Create: `src/whaleshark_reid/core/cluster/hdbscan.py`
- Create: `src/whaleshark_reid/core/cluster/common.py`
- Create: `tests/core/test_cluster_dbscan.py`
- Create: `tests/core/test_cluster_hdbscan.py`

- [ ] **Step 1: Write failing tests for DBSCAN**

`src/whaleshark_reid/core/cluster/__init__.py`:
```python
```

`tests/core/test_cluster_dbscan.py`:
```python
"""Tests for DBSCAN clustering wrapper."""
from __future__ import annotations

import numpy as np

from whaleshark_reid.core.cluster.dbscan import run_dbscan


def _two_clusters_plus_outlier() -> tuple[np.ndarray, list[str]]:
    # Two tight clusters at (10, 10, ...) and (-10, -10, ...), plus one far outlier
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=10.0, scale=0.1, size=(5, 8))
    cluster_b = rng.normal(loc=-10.0, scale=0.1, size=(5, 8))
    outlier = np.array([[100.0] * 8])
    mat = np.vstack([cluster_a, cluster_b, outlier])
    uuids = [f"u{i}" for i in range(11)]
    return mat, uuids


def test_run_dbscan_finds_two_clusters_and_one_noise_point():
    mat, uuids = _two_clusters_plus_outlier()
    results, metrics = run_dbscan(mat, uuids, eps=0.7, min_samples=2, metric="cosine", standardize=True)

    assert len(results) == 11
    labels = [r.cluster_label for r in results]
    assert -1 in labels
    n_noise = sum(1 for l in labels if l == -1)
    n_clusters = len(set(l for l in labels if l != -1))

    assert n_noise >= 1
    assert n_clusters == 2
    assert metrics["n_clusters"] == 2
    assert metrics["n_noise"] == n_noise


def test_run_dbscan_preserves_input_order_and_uuids():
    mat, uuids = _two_clusters_plus_outlier()
    results, _ = run_dbscan(mat, uuids, eps=0.7)
    assert [r.annotation_uuid for r in results] == uuids
```

- [ ] **Step 2: Write failing tests for HDBSCAN**

`tests/core/test_cluster_hdbscan.py`:
```python
"""Tests for HDBSCAN clustering wrapper."""
from __future__ import annotations

import numpy as np

from whaleshark_reid.core.cluster.hdbscan import run_hdbscan


def test_run_hdbscan_returns_labels():
    rng = np.random.default_rng(42)
    mat = np.vstack([
        rng.normal(loc=10.0, scale=0.1, size=(5, 8)),
        rng.normal(loc=-10.0, scale=0.1, size=(5, 8)),
    ])
    uuids = [f"u{i}" for i in range(10)]
    results, metrics = run_hdbscan(mat, uuids, min_cluster_size=3)

    assert len(results) == 10
    # HDBSCAN may label some points as noise; just verify we got ints back
    labels = [r.cluster_label for r in results]
    assert all(isinstance(l, int) for l in labels)
    assert "n_clusters" in metrics
```

- [ ] **Step 3: Run tests — verify failure**

Run: `pytest tests/core/test_cluster_dbscan.py tests/core/test_cluster_hdbscan.py -x`
Expected: `ModuleNotFoundError` for both.

- [ ] **Step 4: Implement `core/cluster/common.py` (shared metrics helper)**

`src/whaleshark_reid/core/cluster/common.py`:
```python
"""Shared cluster stage helpers: metric computation and stage entry point."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from whaleshark_reid.core.schema import ClusterResult, ClusterStageResult


def cluster_metrics(labels: np.ndarray) -> dict:
    """Compute descriptive metrics from a label array. -1 = noise."""
    counts = Counter(labels.tolist())
    n_noise = counts.get(-1, 0)
    cluster_sizes = [sz for lbl, sz in counts.items() if lbl != -1]
    n_clusters = len(cluster_sizes)
    largest = max(cluster_sizes) if cluster_sizes else 0
    median_size = float(np.median(cluster_sizes)) if cluster_sizes else 0.0
    n_singleton_or_noise = n_noise + sum(1 for s in cluster_sizes if s == 1)
    total = len(labels)
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "largest_cluster_size": largest,
        "singleton_fraction": n_singleton_or_noise / total if total else 0.0,
        "median_cluster_size": median_size,
    }


def labels_to_results(
    annotation_uuids: list[str],
    labels: np.ndarray,
    algo: str,
    params: dict,
) -> list[ClusterResult]:
    return [
        ClusterResult(
            annotation_uuid=uuid,
            cluster_label=int(lbl),
            cluster_algo=algo,
            cluster_params=params,
        )
        for uuid, lbl in zip(annotation_uuids, labels)
    ]
```

- [ ] **Step 5: Implement `core/cluster/dbscan.py`**

`src/whaleshark_reid/core/cluster/dbscan.py`:
```python
"""DBSCAN clustering wrapper.

Matches extract_and_evaluate_whalesharks.ipynb exactly:
  StandardScaler().fit_transform(embeddings) → DBSCAN(eps=0.7, min_samples=2, metric='cosine')
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from whaleshark_reid.core.cluster.common import cluster_metrics, labels_to_results
from whaleshark_reid.core.schema import ClusterResult


def run_dbscan(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    eps: float = 0.7,
    min_samples: int = 2,
    metric: str = "cosine",
    standardize: bool = True,
) -> tuple[list[ClusterResult], dict]:
    data = StandardScaler().fit_transform(embeddings) if standardize else embeddings
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
    labels = model.labels_

    params = {
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "standardize": standardize,
    }
    results = labels_to_results(annotation_uuids, labels, algo="dbscan", params=params)
    metrics = cluster_metrics(labels)
    metrics["algo"] = "dbscan"
    return results, metrics
```

- [ ] **Step 6: Implement `core/cluster/hdbscan.py`**

`src/whaleshark_reid/core/cluster/hdbscan.py`:
```python
"""HDBSCAN clustering wrapper (alternative to DBSCAN).

HDBSCAN does not require eps; instead it uses a minimum cluster size and picks
clusters at multiple density levels. Works well when cluster densities vary.
"""
from __future__ import annotations

import numpy as np

from whaleshark_reid.core.cluster.common import cluster_metrics, labels_to_results
from whaleshark_reid.core.schema import ClusterResult


def run_hdbscan(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> tuple[list[ClusterResult], dict]:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
    }
    results = labels_to_results(annotation_uuids, labels, algo="hdbscan", params=params)
    metrics = cluster_metrics(labels)
    metrics["algo"] = "hdbscan"
    return results, metrics
```

- [ ] **Step 7: Run tests — verify pass**

Run: `pytest tests/core/test_cluster_dbscan.py tests/core/test_cluster_hdbscan.py -x`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/cluster/ tests/core/test_cluster_dbscan.py tests/core/test_cluster_hdbscan.py
git commit -m "feat: DBSCAN + HDBSCAN clustering wrappers"
```

---

## Task 10: Cluster — run_cluster_stage + projection (UMAP)

**Files:**
- Create: `src/whaleshark_reid/core/cluster/project.py`
- Modify: `src/whaleshark_reid/core/cluster/common.py` (add run_cluster_stage)
- Create: `tests/core/test_cluster_project.py`
- Create: `tests/core/test_cluster_stage.py`

- [ ] **Step 1: Write failing tests for projection**

`tests/core/test_cluster_project.py`:
```python
"""Tests for UMAP 2D projection wrapper."""
from __future__ import annotations

import numpy as np

from whaleshark_reid.core.cluster.project import run_umap


def test_run_umap_returns_points():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(20, 8)).astype(np.float32)
    uuids = [f"u{i}" for i in range(20)]

    points = run_umap(mat, uuids, n_neighbors=5, min_dist=0.1, random_state=42)

    assert len(points) == 20
    for p in points:
        assert p.algo == "umap"
        assert isinstance(p.x, float)
        assert isinstance(p.y, float)
```

- [ ] **Step 2: Write failing tests for run_cluster_stage end-to-end**

`tests/core/test_cluster_stage.py`:
```python
"""Tests for the cluster stage entry point — reads embeddings, writes clusters parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.cluster.common import run_cluster_stage
from whaleshark_reid.storage.cluster_cache import read_clusters
from whaleshark_reid.storage.embedding_cache import write_embeddings


def test_run_cluster_stage_writes_parquet_and_returns_metrics(tmp_cache_dir: Path):
    rng = np.random.default_rng(42)
    mat = np.vstack([
        rng.normal(loc=10.0, scale=0.1, size=(5, 8)),
        rng.normal(loc=-10.0, scale=0.1, size=(5, 8)),
    ]).astype(np.float32)

    rows = [
        {
            "annotation_uuid": f"u{i}",
            "embedding": mat[i].tolist(),
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        }
        for i in range(10)
    ]
    write_embeddings(tmp_cache_dir, "run_abc", rows)

    result = run_cluster_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="run_abc",
        cluster_run_id="run_xyz",
        algo="dbscan",
        params={"eps": 0.7, "min_samples": 2, "metric": "cosine", "standardize": True},
    )

    assert result.algo == "dbscan"
    assert result.n_clusters == 2

    df = read_clusters(tmp_cache_dir, "run_xyz")
    assert len(df) == 10
    assert set(df["cluster_algo"]) == {"dbscan"}
```

- [ ] **Step 3: Run tests — verify failure**

Run: `pytest tests/core/test_cluster_project.py tests/core/test_cluster_stage.py -x`
Expected: both fail (module missing / function missing).

- [ ] **Step 4: Implement `core/cluster/project.py`**

`src/whaleshark_reid/core/cluster/project.py`:
```python
"""2D UMAP projection for the web cluster view."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.schema import ProjectionPoint, ProjectStageResult


def run_umap(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> list[ProjectionPoint]:
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }
    return [
        ProjectionPoint(
            annotation_uuid=uuid,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            algo="umap",
            params=params,
        )
        for i, uuid in enumerate(annotation_uuids)
    ]


def run_project_stage(
    cache_dir: Path,
    embedding_run_id: str,
    projection_run_id: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> ProjectStageResult:
    from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array
    from whaleshark_reid.storage.projection_cache import write_projections

    uuids, mat = read_embeddings_as_array(cache_dir, embedding_run_id)
    points = run_umap(
        mat, uuids,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    rows = [
        {
            "annotation_uuid": p.annotation_uuid,
            "x": p.x,
            "y": p.y,
            "algo": p.algo,
            "params_json": str(p.params),
        }
        for p in points
    ]
    write_projections(cache_dir, projection_run_id, rows)

    return ProjectStageResult(
        algo="umap",
        n_points=len(points),
        params={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state,
        },
    )
```

- [ ] **Step 5: Implement `run_cluster_stage` in `common.py`**

Append to `src/whaleshark_reid/core/cluster/common.py`:

```python
def run_cluster_stage(
    cache_dir: Path,
    embedding_run_id: str,
    cluster_run_id: str,
    algo: str = "dbscan",
    params: dict | None = None,
) -> ClusterStageResult:
    """Load embeddings parquet → run DBSCAN or HDBSCAN → write cluster parquet → return stage result."""
    import json

    from whaleshark_reid.storage.cluster_cache import write_clusters
    from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array

    params = params or {}
    uuids, mat = read_embeddings_as_array(cache_dir, embedding_run_id)

    if algo == "dbscan":
        from whaleshark_reid.core.cluster.dbscan import run_dbscan

        results, metrics = run_dbscan(
            mat, uuids,
            eps=params.get("eps", 0.7),
            min_samples=params.get("min_samples", 2),
            metric=params.get("metric", "cosine"),
            standardize=params.get("standardize", True),
        )
    elif algo == "hdbscan":
        from whaleshark_reid.core.cluster.hdbscan import run_hdbscan

        results, metrics = run_hdbscan(
            mat, uuids,
            min_cluster_size=params.get("min_cluster_size", 3),
            min_samples=params.get("min_samples"),
            metric=params.get("metric", "euclidean"),
        )
    else:
        raise ValueError(f"Unknown cluster algo: {algo}")

    rows = [
        {
            "annotation_uuid": r.annotation_uuid,
            "cluster_label": int(r.cluster_label),
            "cluster_algo": r.cluster_algo,
            "cluster_params_json": json.dumps(r.cluster_params, default=str),
        }
        for r in results
    ]
    write_clusters(cache_dir, cluster_run_id, rows)

    return ClusterStageResult(
        algo=metrics["algo"],
        n_clusters=metrics["n_clusters"],
        n_noise=metrics["n_noise"],
        largest_cluster_size=metrics["largest_cluster_size"],
        singleton_fraction=metrics["singleton_fraction"],
        median_cluster_size=metrics["median_cluster_size"],
    )
```

- [ ] **Step 6: Run tests — verify pass**

Run: `pytest tests/core/test_cluster_project.py tests/core/test_cluster_stage.py -x`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/cluster/project.py src/whaleshark_reid/core/cluster/common.py tests/core/test_cluster_project.py tests/core/test_cluster_stage.py
git commit -m "feat: run_cluster_stage + UMAP projection"
```

---

## Task 11: Matching — pairs_below_threshold + filter + annotate + run_matching_stage

**Files:**
- Create: `src/whaleshark_reid/core/matching/__init__.py`
- Create: `src/whaleshark_reid/core/matching/pairs.py`
- Create: `tests/core/test_matching_pairs.py`

- [ ] **Step 1: Write failing tests**

`src/whaleshark_reid/core/matching/__init__.py`:
```python
```

`tests/core/test_matching_pairs.py`:
```python
"""Tests for matching/pairs.py — pair candidate generation, filtering, annotation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.matching.pairs import (
    annotate_with_clusters,
    filter_by_decisions,
    pairs_below_threshold,
)
from whaleshark_reid.core.schema import PairCandidate


def _distmat_4x4() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1, 0.5, 0.9],
            [0.1, 0.0, 0.4, 0.8],
            [0.5, 0.4, 0.0, 0.3],
            [0.9, 0.8, 0.3, 0.0],
        ]
    )


def test_pairs_below_threshold_returns_upper_triangle_only():
    distmat = _distmat_4x4()
    uuids = ["a", "b", "c", "d"]
    pairs = pairs_below_threshold(distmat, uuids, threshold=0.6)

    # (a,b)=0.1, (a,c)=0.5, (b,c)=0.4, (c,d)=0.3
    # Upper triangle only, no (b,a) duplicate, no self-pair
    dists = sorted(p.distance for p in pairs)
    assert dists == [0.1, 0.3, 0.4, 0.5]
    assert all(p.ann_a_uuid != p.ann_b_uuid for p in pairs)


def test_pairs_below_threshold_sorted_ascending():
    distmat = _distmat_4x4()
    uuids = ["a", "b", "c", "d"]
    pairs = pairs_below_threshold(distmat, uuids, threshold=1.0)
    dists = [p.distance for p in pairs]
    assert dists == sorted(dists)


def test_filter_by_decisions_drops_match_and_no_match_but_keeps_skip():
    pairs = [
        PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1),
        PairCandidate(ann_a_uuid="a", ann_b_uuid="c", distance=0.5),
        PairCandidate(ann_a_uuid="c", ann_b_uuid="d", distance=0.3),
    ]
    decisions = [
        ("a", "b", "match"),
        ("a", "c", "no_match"),
        ("c", "d", "skip"),  # skip does NOT filter — still in queue
    ]
    filtered = filter_by_decisions(pairs, decisions)
    kept_pairs = {(p.ann_a_uuid, p.ann_b_uuid) for p in filtered}
    assert kept_pairs == {("c", "d")}


def test_filter_by_decisions_is_order_insensitive():
    pairs = [PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1)]
    decisions = [("b", "a", "match")]  # flipped order
    filtered = filter_by_decisions(pairs, decisions)
    assert filtered == []


def test_annotate_with_clusters():
    pairs = [
        PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1),
        PairCandidate(ann_a_uuid="a", ann_b_uuid="c", distance=0.5),
    ]
    cluster_by_uuid = {"a": 3, "b": 3, "c": 7}
    annotated = annotate_with_clusters(pairs, cluster_by_uuid)

    assert annotated[0].cluster_a == 3
    assert annotated[0].cluster_b == 3
    assert annotated[0].same_cluster is True

    assert annotated[1].cluster_a == 3
    assert annotated[1].cluster_b == 7
    assert annotated[1].same_cluster is False
```

- [ ] **Step 2: Write failing test for run_matching_stage**

Append to `tests/core/test_matching_pairs.py`:

```python
# --- run_matching_stage end-to-end ---

def test_run_matching_stage_writes_pair_queue(tmp_cache_dir: Path, tmp_db_path: Path):
    import numpy as np

    from whaleshark_reid.core.matching.pairs import run_matching_stage
    from whaleshark_reid.storage.cluster_cache import write_clusters
    from whaleshark_reid.storage.db import Storage
    from whaleshark_reid.storage.embedding_cache import write_embeddings

    # Prepare a fake state: 4 embeddings, 2 clusters
    storage = Storage(tmp_db_path)
    storage.init_schema()

    # Seed 4 annotations so foreign key constraints are satisfied
    from whaleshark_reid.core.schema import (
        Annotation,
        inat_annotation_uuid,
        inat_image_uuid,
    )
    for i in range(4):
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
        storage.upsert_annotation(ann, run_id="r_ingest")
    uuids = [inat_annotation_uuid(100 + i, 0) for i in range(4)]

    # Seed a run so the pair_queue foreign key is satisfied
    storage.begin_run(run_id="r_match", stage="matching", config={})

    # Fake embeddings: two tight pairs
    mat = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=np.float32,
    )
    emb_rows = [
        {"annotation_uuid": uuids[i], "embedding": mat[i].tolist(),
         "model_id": "m", "model_version": "v", "created_at": "t"}
        for i in range(4)
    ]
    write_embeddings(tmp_cache_dir, "r_embed", emb_rows)

    # Fake clusters
    cluster_rows = [
        {"annotation_uuid": uuids[i], "cluster_label": 0 if i < 2 else 1,
         "cluster_algo": "dbscan", "cluster_params_json": "{}"}
        for i in range(4)
    ]
    write_clusters(tmp_cache_dir, "r_cluster", cluster_rows)

    result = run_matching_stage(
        storage=storage,
        cache_dir=tmp_cache_dir,
        matching_run_id="r_match",
        embedding_run_id="r_embed",
        cluster_run_id="r_cluster",
        distance_threshold=2.0,   # loose, keep everything
        max_queue_size=1000,
    )

    # 6 upper-triangle pairs below threshold
    assert result.n_pairs == 6
    assert storage.count("pair_queue", run_id="r_match") == 6

    # Cross-cluster pairs: (0,2),(0,3),(1,2),(1,3) = 4; same-cluster: (0,1),(2,3) = 2
    assert result.n_same_cluster == 2
    assert result.n_cross_cluster == 4
```

- [ ] **Step 3: Run tests — verify failure**

Run: `pytest tests/core/test_matching_pairs.py -x`
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.core.matching.pairs'`

- [ ] **Step 4: Implement `core/matching/pairs.py`**

`src/whaleshark_reid/core/matching/pairs.py`:
```python
"""Pair candidate generation + filtering + cluster annotation + matching stage entry point."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.schema import MatchingResult, PairCandidate


def pairs_below_threshold(
    distmat: np.ndarray,
    annotation_uuids: list[str],
    threshold: float,
) -> list[PairCandidate]:
    """Return PairCandidates for every (i, j) with i < j and distmat[i,j] < threshold.
    Sorted ascending by distance.
    """
    n = len(annotation_uuids)
    assert distmat.shape == (n, n)
    iu, ju = np.triu_indices(n, k=1)
    mask = distmat[iu, ju] < threshold
    iu, ju = iu[mask], ju[mask]

    pairs = [
        PairCandidate(
            ann_a_uuid=annotation_uuids[i],
            ann_b_uuid=annotation_uuids[j],
            distance=float(distmat[i, j]),
        )
        for i, j in zip(iu, ju)
    ]
    pairs.sort(key=lambda p: p.distance)
    return pairs


def filter_by_decisions(
    candidates: list[PairCandidate],
    active_decisions: list[tuple[str, str, str]],
) -> list[PairCandidate]:
    """Drop candidates whose (a, b) or (b, a) already has a match/no_match decision.
    skip and unsure decisions do NOT filter — those pairs stay in the queue.
    """
    blocked: set[frozenset[str]] = set()
    for a, b, decision in active_decisions:
        if decision in ("match", "no_match"):
            blocked.add(frozenset((a, b)))
    return [p for p in candidates if frozenset((p.ann_a_uuid, p.ann_b_uuid)) not in blocked]


def annotate_with_clusters(
    candidates: list[PairCandidate],
    cluster_by_uuid: dict[str, int],
) -> list[PairCandidate]:
    """Fill in cluster_a, cluster_b, same_cluster for each candidate."""
    out = []
    for p in candidates:
        ca = cluster_by_uuid.get(p.ann_a_uuid)
        cb = cluster_by_uuid.get(p.ann_b_uuid)
        same = ca is not None and cb is not None and ca == cb and ca != -1
        out.append(p.model_copy(update={"cluster_a": ca, "cluster_b": cb, "same_cluster": same}))
    return out


def _distance_percentiles(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "min": float(distances.min()),
        "max": float(distances.max()),
        "median": float(np.median(distances)),
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "p50": float(np.percentile(distances, 50)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
    }


def run_matching_stage(
    storage,
    cache_dir: Path,
    matching_run_id: str,
    embedding_run_id: str,
    cluster_run_id: str,
    distance_threshold: float = 1.0,
    max_queue_size: int = 2000,
) -> MatchingResult:
    """Compute pairwise distances, filter, annotate with clusters, write pair_queue."""
    from wbia_miew_id.metrics.distance import compute_distance_matrix

    from whaleshark_reid.storage.cluster_cache import read_clusters
    from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array

    uuids, mat = read_embeddings_as_array(cache_dir, embedding_run_id)
    cluster_df = read_clusters(cache_dir, cluster_run_id)
    cluster_by_uuid = dict(zip(cluster_df["annotation_uuid"], cluster_df["cluster_label"].astype(int)))

    distmat_tensor = compute_distance_matrix(mat, mat, metric="cosine")
    distmat = distmat_tensor.numpy() if hasattr(distmat_tensor, "numpy") else np.asarray(distmat_tensor)

    candidates = pairs_below_threshold(distmat, uuids, threshold=distance_threshold)

    # Pull active pair decisions
    rows = storage.conn.execute(
        """
        SELECT ann_a_uuid, ann_b_uuid, decision FROM pair_decisions
        WHERE superseded_by IS NULL
        """
    ).fetchall()
    active = [(r["ann_a_uuid"], r["ann_b_uuid"], r["decision"]) for r in rows]
    n_before_filter = len(candidates)
    candidates = filter_by_decisions(candidates, active)
    n_filtered = n_before_filter - len(candidates)

    candidates = annotate_with_clusters(candidates, cluster_by_uuid)
    candidates = candidates[:max_queue_size]

    # Replace pair_queue rows for this matching run
    storage.conn.execute("DELETE FROM pair_queue WHERE run_id = ?", (matching_run_id,))
    for position, p in enumerate(candidates):
        storage.conn.execute(
            """
            INSERT INTO pair_queue (
                run_id, ann_a_uuid, ann_b_uuid, distance,
                cluster_a, cluster_b, same_cluster, position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                matching_run_id,
                p.ann_a_uuid,
                p.ann_b_uuid,
                p.distance,
                p.cluster_a,
                p.cluster_b,
                1 if p.same_cluster else 0,
                position,
            ),
        )

    dists = np.array([p.distance for p in candidates], dtype=np.float64)
    pctiles = _distance_percentiles(dists)
    n_same = sum(1 for p in candidates if p.same_cluster)
    n_cross = len(candidates) - n_same

    return MatchingResult(
        n_pairs=len(candidates),
        n_same_cluster=n_same,
        n_cross_cluster=n_cross,
        n_filtered_out_by_decisions=n_filtered,
        median_distance=pctiles["median"],
        min_distance=pctiles["min"],
        max_distance=pctiles["max"],
        p10=pctiles["p10"],
        p25=pctiles["p25"],
        p50=pctiles["p50"],
        p75=pctiles["p75"],
        p90=pctiles["p90"],
    )
```

- [ ] **Step 5: Run tests — verify pass**

Run: `pytest tests/core/test_matching_pairs.py -x`
Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/matching/ tests/core/test_matching_pairs.py
git commit -m "feat: pair candidate generation + matching stage"
```

---

## Task 12: Feedback — union-find rebuild of individuals

**Files:**
- Create: `src/whaleshark_reid/core/feedback/__init__.py`
- Create: `src/whaleshark_reid/core/feedback/unionfind.py`
- Create: `tests/core/test_feedback_unionfind.py`

- [ ] **Step 1: Write failing tests**

`src/whaleshark_reid/core/feedback/__init__.py`:
```python
```

`tests/core/test_feedback_unionfind.py`:
```python
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

    # Decision exists but is superseded
    storage.conn.execute(
        """
        INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, run_id, created_at, superseded_by)
        VALUES (?, ?, 'match', 'r_test', ?, 999)
        """,
        (u[0], u[1], datetime.now(timezone.utc).isoformat()),
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
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/core/test_feedback_unionfind.py -x`
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.core.feedback.unionfind'`

- [ ] **Step 3: Implement `core/feedback/unionfind.py`**

`src/whaleshark_reid/core/feedback/unionfind.py`:
```python
"""Rebuild annotations.name_uuid from pair_decisions via union-find.

Source of truth is pair_decisions (append-only). annotations.name_uuid is a
materialized view: for each connected component of confirmed-match pairs, we
mint a fresh uuid4 and set it on every annotation in the component. Annotations
with no confirmed matches get name_uuid = NULL.
"""
from __future__ import annotations

from whaleshark_reid.core.schema import RebuildResult, new_name_uuid


class _UnionFind:
    def __init__(self, items: list[str]):
        self.parent: dict[str, str] = {x: x for x in items}
        self.rank: dict[str, int] = {x: 0 for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # path compression
            x = self.parent[x]
        return x

    def union(self, x: str, y: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if y not in self.parent:
            self.parent[y] = y
            self.rank[y] = 0
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def components(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for x in self.parent:
            root = self.find(x)
            out.setdefault(root, []).append(x)
        return out


def rebuild_individuals_cache(storage) -> RebuildResult:
    # Fetch all annotations (need full list so we can null out stale name_uuids)
    all_uuids = [
        row["annotation_uuid"]
        for row in storage.conn.execute("SELECT annotation_uuid FROM annotations").fetchall()
    ]

    # Active match decisions
    rows = storage.conn.execute(
        """
        SELECT ann_a_uuid, ann_b_uuid FROM pair_decisions
        WHERE decision = 'match' AND superseded_by IS NULL
        """
    ).fetchall()

    uf = _UnionFind(all_uuids)
    for r in rows:
        uf.union(r["ann_a_uuid"], r["ann_b_uuid"])

    components = uf.components()
    # Only components with >= 2 members are "individuals". Singletons get NULL.
    new_name_uuid_by_ann: dict[str, str | None] = {}
    n_components = 0
    n_singletons = 0
    for root, members in components.items():
        if len(members) >= 2:
            n_components += 1
            uuid = new_name_uuid()
            for m in members:
                new_name_uuid_by_ann[m] = uuid
        else:
            n_singletons += 1
            new_name_uuid_by_ann[members[0]] = None

    n_updated = 0
    for ann_uuid in all_uuids:
        desired = new_name_uuid_by_ann.get(ann_uuid)
        storage.set_annotation_name_uuid(ann_uuid, desired)
        n_updated += 1

    return RebuildResult(
        n_components=n_components,
        n_singletons=n_singletons,
        n_annotations_updated=n_updated,
    )
```

- [ ] **Step 4: Run tests — verify pass**

Run: `pytest tests/core/test_feedback_unionfind.py -x`
Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/feedback/ tests/core/test_feedback_unionfind.py
git commit -m "feat: feedback rebuild via union-find"
```

---

## Task 13: Metrics — distribution + cluster quality + queue priority

**Files:**
- Create: `src/whaleshark_reid/core/metrics/__init__.py`
- Create: `src/whaleshark_reid/core/metrics/distributions.py`
- Create: `tests/core/test_metrics_distributions.py`

- [ ] **Step 1: Write failing tests**

`src/whaleshark_reid/core/metrics/__init__.py`:
```python
```

`tests/core/test_metrics_distributions.py`:
```python
"""Tests for core.metrics.distributions."""
from __future__ import annotations

import numpy as np
import pytest

from whaleshark_reid.core.metrics.distributions import (
    cluster_quality_stats,
    distance_distribution_stats,
    queue_priority_stats,
)
from whaleshark_reid.core.schema import PairCandidate


def test_distance_distribution_stats_basic():
    distmat = np.array([
        [0.0, 0.2, 0.5],
        [0.2, 0.0, 0.3],
        [0.5, 0.3, 0.0],
    ])
    stats = distance_distribution_stats(distmat)
    # Upper triangle: [0.2, 0.5, 0.3]
    assert stats["n"] == 3
    assert np.isclose(stats["median"], 0.3)
    assert "histogram" in stats
    assert len(stats["histogram"]) == 20


def test_cluster_quality_stats_with_two_clusters():
    rng = np.random.default_rng(0)
    mat = np.vstack([
        rng.normal(loc=10.0, scale=0.1, size=(5, 4)),
        rng.normal(loc=-10.0, scale=0.1, size=(5, 4)),
    ])
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    stats = cluster_quality_stats(mat, labels)
    assert stats["noise_fraction"] == 0.0
    assert stats["silhouette_score"] is not None


def test_queue_priority_stats_with_mixed_clusters():
    pairs = [
        PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1, same_cluster=True),
        PairCandidate(ann_a_uuid="a", ann_b_uuid="c", distance=0.2, same_cluster=False),
        PairCandidate(ann_a_uuid="c", ann_b_uuid="d", distance=0.3, same_cluster=True),
    ]
    stats = queue_priority_stats(pairs)
    assert stats["n_pairs"] == 3
    assert stats["fraction_same_cluster"] == pytest.approx(2 / 3)
    assert np.isclose(stats["median_distance"], 0.2)
```

- [ ] **Step 2: Run tests — verify failure**

Run: `pytest tests/core/test_metrics_distributions.py -x`
Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement `core/metrics/distributions.py`**

`src/whaleshark_reid/core/metrics/distributions.py`:
```python
"""Distance distribution + cluster quality + queue priority metrics."""
from __future__ import annotations

import numpy as np


def distance_distribution_stats(distmat: np.ndarray) -> dict:
    """Compute descriptive stats on the upper triangle of a square distance matrix.

    Returns dict with n, mean, std, median, percentiles, histogram (20 bins, [0, 2]).
    """
    n = distmat.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    d = distmat[iu, ju]
    hist, edges = np.histogram(d, bins=20, range=(0.0, 2.0))
    return {
        "n": int(len(d)),
        "mean": float(d.mean()) if len(d) else 0.0,
        "std": float(d.std()) if len(d) else 0.0,
        "median": float(np.median(d)) if len(d) else 0.0,
        "p10": float(np.percentile(d, 10)) if len(d) else 0.0,
        "p25": float(np.percentile(d, 25)) if len(d) else 0.0,
        "p50": float(np.percentile(d, 50)) if len(d) else 0.0,
        "p75": float(np.percentile(d, 75)) if len(d) else 0.0,
        "p90": float(np.percentile(d, 90)) if len(d) else 0.0,
        "histogram": hist.tolist(),
        "histogram_edges": edges.tolist(),
    }


def cluster_quality_stats(embeddings: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """Compute silhouette score (if possible), noise fraction, cluster size histogram."""
    from collections import Counter

    labels = np.asarray(cluster_labels)
    n = len(labels)
    noise = int((labels == -1).sum())
    non_noise_mask = labels != -1
    n_unique_non_noise = len(set(labels[non_noise_mask].tolist()))

    silhouette: float | None = None
    if n_unique_non_noise > 1 and non_noise_mask.sum() > 1:
        from sklearn.metrics import silhouette_score
        try:
            silhouette = float(silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask]))
        except Exception:
            silhouette = None

    sizes = list(Counter(labels[non_noise_mask].tolist()).values())

    return {
        "silhouette_score": silhouette,
        "noise_fraction": noise / n if n else 0.0,
        "n_clusters": n_unique_non_noise,
        "cluster_size_histogram": sizes,
        "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
    }


def queue_priority_stats(pair_queue) -> dict:
    """Summary stats over a list of PairCandidate."""
    if not pair_queue:
        return {
            "n_pairs": 0,
            "fraction_same_cluster": 0.0,
            "median_distance": 0.0,
        }
    distances = np.array([p.distance for p in pair_queue], dtype=np.float64)
    same = sum(1 for p in pair_queue if p.same_cluster)
    return {
        "n_pairs": len(pair_queue),
        "fraction_same_cluster": same / len(pair_queue),
        "median_distance": float(np.median(distances)),
    }
```

- [ ] **Step 4: Run tests — verify pass**

Run: `pytest tests/core/test_metrics_distributions.py -x`
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/core/metrics/ tests/core/test_metrics_distributions.py
git commit -m "feat: distribution + cluster quality + queue priority metrics"
```

---

## Task 14: Integration test — full pipeline on mini fixture

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_full_pipeline.py`

- [ ] **Step 1: Write the end-to-end integration test**

`tests/integration/__init__.py`:
```python
```

`tests/integration/test_full_pipeline.py`:
```python
"""End-to-end pipeline test: ingest → embed (stubbed) → cluster → match → project → rebuild.

Uses a stubbed MiewID so the test runs on CPU in <5 seconds without hitting HuggingFace.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from whaleshark_reid.core.cluster.common import run_cluster_stage
from whaleshark_reid.core.cluster.project import run_project_stage
from whaleshark_reid.core.embed.miewid import run_embed_stage
from whaleshark_reid.core.feedback.unionfind import rebuild_individuals_cache
from whaleshark_reid.core.ingest.inat import ingest_inat_csv
from whaleshark_reid.core.matching.pairs import run_matching_stage
from whaleshark_reid.core.schema import inat_annotation_uuid
from whaleshark_reid.storage.cluster_cache import read_clusters
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.embedding_cache import read_embeddings
from whaleshark_reid.storage.projection_cache import read_projections

FIXTURES = Path(__file__).parent.parent / "fixtures"


class _StubMiewId(torch.nn.Module):
    """Same stub as test_embed_miewid — deterministic embed_dim=8."""
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        per_sample = x.mean(dim=(1, 2, 3))
        return per_sample.unsqueeze(1).expand(-1, self.embed_dim)

    def forward(self, x, label=None):
        return self.extract_feat(x)


@pytest.fixture
def stub_miewid(monkeypatch):
    from transformers import AutoModel

    monkeypatch.setattr(
        AutoModel, "from_pretrained",
        lambda *a, **k: _StubMiewId(),
    )
    yield


def test_full_pipeline_on_mini_fixture(
    stub_miewid,
    tmp_db_path: Path,
    tmp_cache_dir: Path,
):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    # --- Stage 1: ingest ---
    ingest_result = ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="r_ingest",
        rich_csv_path=FIXTURES / "mini_inat_rich.csv",
    )
    assert ingest_result.n_ingested == 10

    # --- Stage 2: embed ---
    storage.begin_run(run_id="r_embed", stage="embed", config={})
    embed_result = run_embed_stage(
        storage=storage,
        cache_dir=tmp_cache_dir,
        run_id="r_embed",
        batch_size=4,
        num_workers=0,
        device="cpu",
    )
    assert embed_result.n_embedded == 10
    assert embed_result.embed_dim == 8
    storage.finish_run("r_embed", "ok", metrics=embed_result.model_dump())

    df_emb = read_embeddings(tmp_cache_dir, "r_embed")
    assert len(df_emb) == 10

    # --- Stage 3: cluster ---
    storage.begin_run(run_id="r_cluster", stage="cluster", config={})
    cluster_result = run_cluster_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="r_embed",
        cluster_run_id="r_cluster",
        algo="dbscan",
        params={"eps": 0.7, "min_samples": 2, "metric": "cosine", "standardize": True},
    )
    storage.finish_run("r_cluster", "ok", metrics=cluster_result.model_dump())

    df_clu = read_clusters(tmp_cache_dir, "r_cluster")
    assert len(df_clu) == 10

    # --- Stage 4: matching ---
    storage.begin_run(run_id="r_match", stage="matching", config={})
    match_result = run_matching_stage(
        storage=storage,
        cache_dir=tmp_cache_dir,
        matching_run_id="r_match",
        embedding_run_id="r_embed",
        cluster_run_id="r_cluster",
        distance_threshold=2.0,  # loose, include everything
        max_queue_size=1000,
    )
    storage.finish_run("r_match", "ok", metrics=match_result.model_dump())
    assert storage.count("pair_queue", run_id="r_match") > 0

    # --- Stage 5: project (UMAP) ---
    storage.begin_run(run_id="r_project", stage="project", config={})
    project_result = run_project_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="r_embed",
        projection_run_id="r_project",
        n_neighbors=5,
    )
    storage.finish_run("r_project", "ok", metrics=project_result.model_dump())

    df_proj = read_projections(tmp_cache_dir, "r_project")
    assert len(df_proj) == 10

    # --- Stage 6: seed one pair decision, rebuild individuals ---
    u0 = inat_annotation_uuid(100, 0)
    u1 = inat_annotation_uuid(101, 0)
    storage.conn.execute(
        "INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, created_at) "
        "VALUES (?, ?, 'match', ?)",
        (u0, u1, datetime.now(timezone.utc).isoformat()),
    )
    rebuild_result = rebuild_individuals_cache(storage)
    assert rebuild_result.n_components == 1

    a = storage.get_annotation(u0)
    b = storage.get_annotation(u1)
    assert a.name_uuid is not None
    assert a.name_uuid == b.name_uuid
```

- [ ] **Step 2: Run the integration test**

Run: `pytest tests/integration/test_full_pipeline.py -x -v`
Expected: 1 test passes in under 10 seconds.

If it fails with import-time errors from `wbia_miew_id` (e.g. wandb import), add `wandb` to the dev dependencies in `pyproject.toml` and reinstall.

- [ ] **Step 3: Run the entire test suite**

Run: `pytest -x`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add tests/integration/
git commit -m "test: full pipeline integration on mini fixture"
```

---

## Task 15: Validation notebook against benchmark

**Files:**
- Create: `/workspace/catalog-match/whaleshark/verify_core.ipynb` (lives next to `extract_and_evaluate_whalesharks.ipynb` so it can reuse the same data paths)

- [ ] **Step 1: Create the verification notebook via nbformat**

Run in a Python shell or as a script:

```bash
cd /workspace/catalog-match/whaleshark-reid
python <<'PY'
import json
from pathlib import Path

nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# verify_core — reproduce `extract_and_evaluate_whalesharks.ipynb` outputs from `whaleshark_reid.core`\n",
                "\n",
                "This notebook validates that the Phase 1 core engine produces the same DBSCAN cluster count and distance distribution stats as the benchmark notebook on the same iNat data.\n",
                "\n",
                "Run after Task 14 is complete."
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import tempfile\n",
                "from pathlib import Path\n",
                "\n",
                "from whaleshark_reid.core.cluster.common import run_cluster_stage\n",
                "from whaleshark_reid.core.embed.miewid import run_embed_stage\n",
                "from whaleshark_reid.core.ingest.inat import ingest_inat_csv\n",
                "from whaleshark_reid.core.matching.pairs import run_matching_stage\n",
                "from whaleshark_reid.storage.cluster_cache import read_clusters\n",
                "from whaleshark_reid.storage.db import Storage\n",
                "from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array\n",
                "\n",
                "tmpdir = Path(tempfile.mkdtemp(prefix='verify_core_'))\n",
                "print('Working dir:', tmpdir)\n",
                "db_path = tmpdir / 'state.db'\n",
                "cache_dir = tmpdir / 'cache'\n",
                "cache_dir.mkdir()"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Ingest iNat CSV\n",
                "storage = Storage(db_path)\n",
                "storage.init_schema()\n",
                "\n",
                "ingest_result = ingest_inat_csv(\n",
                "    csv_path=Path('/workspace/catalog-match/whaleshark/whaleshark_inat_v1.csv'),\n",
                "    photos_dir=Path('/workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/photos'),\n",
                "    storage=storage,\n",
                "    run_id='verify_ingest',\n",
                "    rich_csv_path=Path('/workspace/catalog-match/whaleshark/dfx_whaleshark_inat_v1.csv'),\n",
                ")\n",
                "print('Ingest result:', ingest_result)"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Embed (real MiewID msv3 via HuggingFace)\n",
                "storage.begin_run(run_id='verify_embed', stage='embed', config={'model': 'miewid-msv3'})\n",
                "embed_result = run_embed_stage(\n",
                "    storage=storage,\n",
                "    cache_dir=cache_dir,\n",
                "    run_id='verify_embed',\n",
                "    batch_size=16,\n",
                "    num_workers=2,\n",
                "    use_bbox=True,\n",
                ")\n",
                "storage.finish_run('verify_embed', 'ok', metrics=embed_result.model_dump())\n",
                "print('Embed result:', embed_result)"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# DBSCAN with the exact benchmark notebook config\n",
                "storage.begin_run(run_id='verify_cluster', stage='cluster', config={})\n",
                "cluster_result = run_cluster_stage(\n",
                "    cache_dir=cache_dir,\n",
                "    embedding_run_id='verify_embed',\n",
                "    cluster_run_id='verify_cluster',\n",
                "    algo='dbscan',\n",
                "    params={'eps': 0.7, 'min_samples': 2, 'metric': 'cosine', 'standardize': True},\n",
                ")\n",
                "storage.finish_run('verify_cluster', 'ok', metrics=cluster_result.model_dump())\n",
                "print('Cluster result:', cluster_result)"
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Compare against the benchmark notebook output.\n",
                "#\n",
                "# Open /workspace/catalog-match/whaleshark/extract_and_evaluate_whalesharks.ipynb\n",
                "# in parallel and compare:\n",
                "#   - n_clusters (from its DBSCAN call)\n",
                "#   - n_noise\n",
                "#   - distance distribution stats (from its compute_distance_matrix + np.triu)\n",
                "#\n",
                "# Expected: EXACT match on n_clusters / n_noise (same sklearn + same eps),\n",
                "# median distance within ~1e-3.\n",
                "#\n",
                "# If numbers differ materially, debug: check that use_bbox=True is matching\n",
                "# the notebook's crop config, check that embeddings are being extracted with\n",
                "# the same transforms.\n",
                "print('Compare with the benchmark notebook manually and record deltas below.')\n",
                "print('n_clusters:', cluster_result.n_clusters)\n",
                "print('n_noise:', cluster_result.n_noise)\n",
                "print('singleton_fraction:', cluster_result.singleton_fraction)"
            ],
        },
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path('/workspace/catalog-match/whaleshark/verify_core.ipynb')
out.write_text(json.dumps(nb, indent=1))
print('Wrote', out)
PY
```

Expected: `Wrote /workspace/catalog-match/whaleshark/verify_core.ipynb`.

- [ ] **Step 2: Commit the notebook**

```bash
cd /workspace/catalog-match/whaleshark-reid
# The notebook lives outside whaleshark-reid/, so commit separately
git add /workspace/catalog-match/whaleshark/verify_core.ipynb 2>/dev/null || true
# If the above fails because /workspace/catalog-match/ is not a repo (which it isn't per earlier check),
# just note the file exists and move on. The notebook is a manual validation tool, not a repo artifact.
echo "Validation notebook created at /workspace/catalog-match/whaleshark/verify_core.ipynb"
```

- [ ] **Step 3: Run the full test suite one last time**

```bash
cd /workspace/catalog-match/whaleshark-reid
pytest -x
```

Expected: all tests pass.

- [ ] **Step 4: Final commit for the plan completion**

```bash
cd /workspace/catalog-match/whaleshark-reid
git status
# Should be clean. If there's anything untracked, add it here.
git log --oneline | head -20
```

Expected: 14 tidy commits, one per task 1-14.

---

## Success criteria (from spec 01)

After all 15 tasks are complete, all of the following should be true:

1. ✅ All modules in `src/whaleshark_reid/core/` and `src/whaleshark_reid/storage/` exist and have passing unit tests.
2. ✅ `verify_core.ipynb` reproduces the benchmark notebook's DBSCAN cluster count and distance distribution (manual verification — run the notebook and compare to the existing `extract_and_evaluate_whalesharks.ipynb` output).
3. ✅ `pytest tests/core tests/storage` passes in under 3 seconds.
4. ✅ `Annotation` pydantic round-trips (dict → Annotation → dict).
5. ✅ SQLite schema is fresh-creatable from `schema.sql` via `Storage(db_path).init_schema()`.
6. ✅ Re-running `ingest_inat_csv` on the same CSV is a no-op (`n_skipped_existing == n_read`).
7. ✅ Re-running `run_embed_stage` with `only_missing=True` skips already-cached annotation_uuids.
8. ✅ `rebuild_individuals_cache` correctly materializes `annotations.name_uuid` from confirmed pair decisions.

## What this plan explicitly does NOT cover

- Typer CLI (plan 02, written later).
- FastAPI web app (plan 03, written later).
- Phase 2: Wildbook ingest, GT metrics, calibration reports, reconcile mode.
- Phase 3+: advanced matching experiments.
