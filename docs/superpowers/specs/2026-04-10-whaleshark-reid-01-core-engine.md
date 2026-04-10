# Whale Shark Re-ID — Spec 01: Core Engine

**Date:** 2026-04-10
**Depends on:** nothing
**Related specs:** [overview](2026-04-10-whaleshark-reid-overview.md), [02-cli](2026-04-10-whaleshark-reid-02-cli-and-orchestration.md), [03-web-ui](2026-04-10-whaleshark-reid-03-web-ui.md)

## Purpose

Build the pure-Python core of the whale shark re-ID system: data schema, storage layer, ingest, embedding extraction, clustering, matching, feedback, and metrics. No CLI, no web. Notebook-first — every function must be importable and callable from a Jupyter cell sitting next to `extract_and_evaluate_whalesharks.ipynb` with the same ergonomics the existing notebook uses.

This sub-spec is implemented first. Its validation checkpoint is: **reproduce the DBSCAN cluster count and distance distribution from `extract_and_evaluate_whalesharks.ipynb` on the same iNat data, from a notebook that imports `whaleshark_reid.core`.**

## Module layout

```
src/whaleshark_reid/
├── core/
│   ├── __init__.py
│   ├── schema.py           # Annotation pydantic + UUID helpers
│   ├── ingest/
│   │   ├── __init__.py
│   │   └── inat.py         # iNat CSV → annotations table
│   ├── embed/
│   │   ├── __init__.py
│   │   ├── miewid.py       # In-process MiewID wrapper
│   │   └── cache.py        # Parquet embedding cache
│   ├── cluster/
│   │   ├── __init__.py
│   │   ├── dbscan.py       # DBSCAN baseline matching notebook config
│   │   ├── hdbscan.py      # HDBSCAN alternate
│   │   └── project.py      # 2D UMAP projection for web cluster view
│   ├── matching/
│   │   ├── __init__.py
│   │   └── pairs.py        # pairs_below_threshold, pair queue materialization
│   ├── feedback/
│   │   ├── __init__.py
│   │   └── unionfind.py    # Union-find over pair_decisions → annotations.name_uuid
│   └── metrics/
│       ├── __init__.py
│       └── distributions.py # Distance distribution stats, cluster metrics
└── storage/
    ├── __init__.py
    ├── db.py               # SQLite session + schema + query helpers
    ├── schema.sql          # DDL for all tables
    ├── embedding_cache.py  # Parquet read/write for embeddings
    ├── cluster_cache.py    # Parquet read/write for cluster labels
    └── projection_cache.py # Parquet read/write for 2D projections
```

## 1. `core/schema.py`

### Annotation pydantic model

```python
import uuid
from pydantic import BaseModel, Field
from typing import Optional

INAT_NAMESPACE = uuid.UUID("6f4e6a5e-7b7a-4f3b-9c1d-1f0a2c3d4e5f")

class Annotation(BaseModel):
    # Canonical UUID identifiers
    annotation_uuid: str
    image_uuid: str
    name_uuid: Optional[str] = None

    # Source reference
    source: str
    source_annotation_id: Optional[str] = None
    source_image_id: Optional[str] = None
    source_individual_id: Optional[str] = None
    observation_id: Optional[int] = None
    photo_index: Optional[int] = None

    # MiewID-required fields (names match MiewIdDataset.__getitem__)
    file_path: str
    file_name: str
    bbox: list[float] = Field(..., description="[x, y, w, h]")
    theta: float = 0.0
    viewpoint: str = "unknown"
    species: str = "whaleshark"
    name: Optional[str] = None

    # Provenance / dev-mode display
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

    # Derived
    name_viewpoint: Optional[str] = None
    species_viewpoint: Optional[str] = None
```

### UUID helpers

```python
def inat_annotation_uuid(observation_id: int, photo_index: int) -> str:
    return str(uuid.uuid5(INAT_NAMESPACE, f"inat:annotation:{observation_id}:{photo_index}"))

def inat_image_uuid(observation_id: int, photo_index: int) -> str:
    return str(uuid.uuid5(INAT_NAMESPACE, f"inat:image:{observation_id}:{photo_index}"))

def new_name_uuid() -> str:
    return str(uuid.uuid4())
```

### Other dataclasses

```python
class PairCandidate(BaseModel):
    ann_a_uuid: str
    ann_b_uuid: str
    distance: float
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    same_cluster: bool = False

class ClusterResult(BaseModel):
    annotation_uuid: str
    cluster_label: int     # -1 = noise
    cluster_algo: str
    cluster_params: dict

class ProjectionPoint(BaseModel):
    annotation_uuid: str
    x: float
    y: float
    algo: str
    params: dict
```

## 2. `storage/db.py` + `schema.sql`

### SQLite connection

```python
class Storage:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), isolation_level=None)  # autocommit off via explicit txns
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.execute("PRAGMA busy_timeout = 5000;")
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        with open(SCHEMA_SQL_PATH) as f:
            self.conn.executescript(f.read())

    def upsert_annotation(self, ann: Annotation, run_id: str) -> None: ...
    def get_annotation(self, annotation_uuid: str) -> Optional[Annotation]: ...
    def list_annotations(self, filters: dict) -> list[Annotation]: ...
    def count(self, table: str, **where) -> int: ...
    def begin_run(self, run_id: str, stage: str, config: dict) -> None: ...
    def finish_run(self, run_id: str, status: str, metrics: dict, error: Optional[str] = None, notes: str = "") -> None: ...
    def set_annotation_name_uuid(self, annotation_uuid: str, name_uuid: Optional[str]) -> None: ...
```

### Full SQLite DDL (in `storage/schema.sql`)

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

-- Unified runs table: every CLI command writes one row here. metrics_json is filled in at finish_run time.
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    config_json TEXT NOT NULL,
    metrics_json TEXT,                -- filled by finish_run(); NULL if still running
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

**Derived individual identity lives directly on `annotations.name_uuid`.** The feedback rebuild (`core/feedback/unionfind.py::rebuild_individuals_cache`) computes union-find over `pair_decisions WHERE decision='match'` and writes the resulting `name_uuid` for each component back into `annotations.name_uuid` — no separate `individuals_cache` table, no `name_uuid_history`. Fresh `uuid4()` on every rebuild is fine; users don't compare UUIDs across rebuilds, they compare current state.

### `storage/embedding_cache.py`

```python
def write_embeddings(cache_dir: Path, run_id: str, rows: Iterator[dict]) -> Path:
    """Write to cache_dir/embeddings/<run_id>.parquet.
    rows yields dicts: {annotation_uuid, embedding (list[float]), model_id, model_version, created_at}
    """

def read_embeddings(cache_dir: Path, run_id: str) -> pd.DataFrame:
    """Return DataFrame with columns: annotation_uuid, embedding, model_id, model_version, created_at."""

def read_embeddings_as_array(cache_dir: Path, run_id: str) -> tuple[list[str], np.ndarray]:
    """Return (annotation_uuids, embeddings_matrix) — annotation_uuids[i] corresponds to embeddings_matrix[i]."""

def existing_annotation_uuids(cache_dir: Path, run_id: str) -> set[str]:
    """For incremental embed: which annotation_uuids are already cached for this run?"""
```

### `storage/cluster_cache.py` and `storage/projection_cache.py`

Same shape: parquet files per run_id, loaded as DataFrames. Keep them simple.

## 3. `core/ingest/inat.py`

### Function

```python
def ingest_inat_csv(
    csv_path: Path,
    photos_dir: Path,
    storage: Storage,
    run_id: str,
    rich_csv_path: Optional[Path] = None,
) -> IngestResult:
    """
    Read whaleshark_inat_v1.csv (minimal, 10 cols) or dfx_whaleshark_inat_v1.csv (rich, 41 cols)
    — auto-detect by column set.

    For each row:
      1. Build Annotation with deterministic UUIDs (uuid5).
      2. Strip stale hard-coded file_path prefix, rebuild against photos_dir.
      3. If rich_csv_path is provided, LEFT JOIN on observation_id to backfill provenance.
      4. Verify the image file exists; log warning for missing (don't skip the row — insert with a missing_file flag in extra).
      5. Compute name_viewpoint, species_viewpoint.
      6. INSERT OR IGNORE into annotations (unique on (source, observation_id, photo_index)).

    Returns IngestResult(n_read, n_ingested, n_skipped_existing, n_missing_files).
    """

class IngestResult(BaseModel):
    n_read: int
    n_ingested: int
    n_skipped_existing: int
    n_missing_files: int
```

### Known quirks to handle

- The CSV has a `file_path` column with a stale macOS path (`/Users/lashaotarashvili/Desktop/...`). Must be stripped and rebuilt against `photos_dir` using just the `file_name`.
- iNat CSV only has `observation_id`, not a separate `photo_index` column for the minimal format — derive `photo_index` from the filename suffix (`327286790_1.jpg` → photo_index=0, `327286790_2.jpg` → photo_index=1) or from the order of rows if filename doesn't encode it.
- The minimal CSV has `name='unknown'`, `viewpoint='unknown'`. Normalize these to `None` for `name` and keep `'unknown'` for `viewpoint` (viewpoint is a categorical, not an identity).
- The rich CSV has provenance fields missing from the minimal CSV — `ingest` opportunistically joins the two if both are provided via `--rich-csv`.

## 4. `core/embed/miewid.py` + `core/embed/cache.py`

**Design note — code reuse:** `wbia_miew_id` already provides a canonical embedding extraction pipeline. We reuse it verbatim to guarantee our embeddings match the repo's preprocessing exactly. This is critical: the repo uses Albumentations transforms with specific resize + normalize + ToTensor ordering, not the torchvision equivalents. Rewriting the preprocessing would silently drift our embeddings from the benchmark notebook's outputs. The reused entry points are:

| What | Where | Role |
|------|-------|------|
| `MiewIdDataset(csv, transforms, crop_bbox, fliplr, fliplr_view)` | `wbia_miew_id/datasets/default_dataset.py` | DataFrame → batched tensors; handles `load_image` + `get_chip_from_img` internally |
| `get_test_transforms(image_size)` | `wbia_miew_id/datasets/transforms.py` | Canonical inference preprocessing: `albumentations.Resize` + `albumentations.Normalize()` + `ToTensorV2` |
| `extract_embeddings(data_loader, model, device)` | `wbia_miew_id/engine/eval_fn.py` | Canonical batched inference loop with tqdm, autocast, no_grad, NaN assertion. Returns `(embeddings: np.ndarray, labels: list)` |
| `AutoModel.from_pretrained("conservationxlabs/miewid-msv3", trust_remote_code=True)` | HuggingFace | Loads a `MiewIdNet` instance whose `extract_feat(x)` method returns features |

### `embed_annotations` — the whole thing

```python
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from wbia_miew_id.datasets import MiewIdDataset, get_test_transforms
from wbia_miew_id.engine.eval_fn import extract_embeddings as _extract_embeddings

def embed_annotations(
    annotations: list[Annotation],
    model_id: str = "conservationxlabs/miewid-msv3",
    image_size: tuple[int, int] = (440, 440),
    batch_size: int = 32,
    num_workers: int = 2,
    use_bbox: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Extract MiewID embeddings for a list of annotations by wrapping wbia_miew_id's
    canonical pipeline (MiewIdDataset + get_test_transforms + extract_embeddings).
    Returns np.ndarray of shape (N, embed_dim) float32.

    Field mapping: Annotation.file_path/bbox/theta/viewpoint/species/name pass through
    to the DataFrame columns MiewIdDataset expects. The 'name' column is set to a dummy
    integer per row (the range index) since MiewIdDataset casts it to a label tensor and
    we don't use labels for embedding extraction.

    Quirks:
      - If use_bbox=False, MiewIdDataset.crop_bbox=False means it embeds the whole image.
      - For Phase 1 cold-start, use_bbox=True (we have detected bboxes in the CSV).
      - theta in radians (0.0 is fine for axis-aligned boxes).
      - Device defaults to cuda if available, else cpu. Single-GPU only (see persistent note
        on DDP being fragile).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.DataFrame([{
        "file_path": ann.file_path,
        "bbox": ann.bbox,
        "theta": ann.theta,
        "name": i,            # dummy int label; MiewIdDataset casts to tensor
        "species": ann.species,
        "viewpoint": ann.viewpoint,
    } for i, ann in enumerate(annotations)])

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
        pin_memory=True,
        drop_last=False,
    )

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    embeddings, _labels = _extract_embeddings(loader, model, device)
    return embeddings  # np.ndarray, shape (N, embed_dim)
```

No custom inference loop. No custom transforms. No custom `get_chip_from_img` call — it's inside `MiewIdDataset`. The only thing we own is the DataFrame construction and the HuggingFace model load.

### `run_embed_stage` entry point

```python
def run_embed_stage(
    storage: Storage,
    cache_dir: Path,
    run_id: str,
    model_id: str = "conservationxlabs/miewid-msv3",
    batch_size: int = 32,
    use_bbox: bool = True,
    only_missing: bool = True,
) -> EmbedResult:
    """
    1. Load annotations from SQLite.
    2. If only_missing, filter to annotations whose annotation_uuid is NOT in the existing embedding cache parquet (idempotency).
    3. Call embed_annotations() to get the (N, D) matrix.
    4. Write rows (annotation_uuid, embedding, model_id, model_version, created_at) to cache_dir/embeddings/<run_id>.parquet.
    5. Return EmbedResult. Caller (the CLI command) attaches metrics to the run row via finish_run(run_id, status='ok', metrics=result.model_dump()).
    """
```

## 5. `core/cluster/dbscan.py`

```python
def run_dbscan(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    eps: float = 0.7,
    min_samples: int = 2,
    metric: str = "cosine",
    standardize: bool = True,
) -> tuple[list[ClusterResult], dict]:
    """
    Matches extract_and_evaluate_whalesharks.ipynb exactly:
      1. StandardScaler().fit_transform(embeddings) if standardize
      2. DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(scaled)
      3. Return (list[ClusterResult], metrics_dict)

    metrics_dict includes:
      - n_clusters (excluding noise)
      - n_noise
      - largest_cluster_size
      - singleton_fraction (fraction of points in clusters of size 1 OR noise)
      - median_cluster_size
    """
```

`core/cluster/hdbscan.py` is structurally identical, uses `hdbscan.HDBSCAN(...)`. Same return shape.

`run_cluster_stage` entry point reads embeddings parquet, calls `run_dbscan` (or hdbscan), writes `clusters/<run_id>.parquet`, returns a `ClusterStageResult` with metrics for the caller to attach via `finish_run`.

## 6. `core/matching/pairs.py`

```python
def pairs_below_threshold(
    distmat: np.ndarray,
    annotation_uuids: list[str],
    threshold: float,
) -> list[PairCandidate]:
    """
    Return all (i, j) with i < j and distmat[i,j] < threshold, as PairCandidate objects.
    Sorted by distance ascending.
    Does NOT apply pair_decisions filter — that's run_matching_stage's job.
    """

def filter_by_decisions(
    candidates: list[PairCandidate],
    active_decisions: list[tuple[str, str, str]],  # (ann_a, ann_b, decision)
) -> list[PairCandidate]:
    """Drop candidates where a match/no_match decision already exists."""

def annotate_with_clusters(
    candidates: list[PairCandidate],
    cluster_by_uuid: dict[str, int],
) -> list[PairCandidate]:
    """Fill in cluster_a, cluster_b, same_cluster fields."""

def run_matching_stage(
    storage: Storage,
    cache_dir: Path,
    run_id: str,
    distance_threshold: float = 1.0,
    max_queue_size: int = 2000,
) -> MatchingResult:
    """
    1. Load embeddings + cluster labels from parquet.
    2. Compute distance matrix via wbia_miew_id.metrics.distance.compute_distance_matrix(emb, emb, metric='cosine').
       For large N, use compute_batched_distance_matrix.
    3. pairs_below_threshold → filter_by_decisions → annotate_with_clusters.
    4. Sort ascending, cap at max_queue_size.
    5. Write to pair_queue table (delete existing rows for this run_id first).
    6. Return MatchingResult with distance distribution stats (for the caller to attach
       to the run row via finish_run(run_id, status='ok', metrics=result.model_dump())):
       - n_pairs, median_distance, min_distance, max_distance
       - p10, p25, p50, p75, p90 of distance
       - n_same_cluster, n_cross_cluster
       - n_filtered_out_by_decisions
    """
```

## 7. `core/feedback/unionfind.py`

```python
def rebuild_individuals_cache(storage: Storage) -> RebuildResult:
    """
    Rebuild annotations.name_uuid from the current pair_decisions state.

    1. Read all confirmed pairs:
         SELECT ann_a_uuid, ann_b_uuid FROM pair_decisions
         WHERE decision = 'match' AND superseded_by IS NULL.
    2. Run union-find over those pairs. Each connected component = one individual.
    3. For each component: mint a fresh uuid4() and set annotations.name_uuid
       to that value for every annotation in the component (UPDATE annotations SET name_uuid = ...).
    4. For annotations NOT in any component (no confirmed matches): set name_uuid = NULL
       if it is currently a derived UUID. Phase 1 has no other source of name_uuid so
       this amounts to "clear and rewrite".

    Returns RebuildResult(n_components, n_singletons, n_annotations_updated).

    Notes:
      - Phase 2 will add handling for annotations with source_of_label='source'
        (Wildbook-labeled). Those come in with a pre-assigned name_uuid and are NOT
        overwritten by rebuild. For Phase 1 there is no such case.
      - 'Fresh uuid4 per rebuild' is intentional: callers should not rely on UUID
        stability across rebuilds. The source of truth is pair_decisions; name_uuid
        is a materialized view.
    """
```

**Design note:** Singletons (annotations with no confirmed matches) have `name_uuid = NULL`. The UI's Individuals tab queries `annotations` grouped by `name_uuid`, with NULL showing up as a "singletons" section at the end of the list.

## 8. `core/metrics/distributions.py`

```python
def distance_distribution_stats(distmat: np.ndarray) -> dict:
    """Upper triangle stats. Returns dict with mean, median, std, percentiles, histogram (20 bins)."""

def cluster_quality_stats(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict:
    """Returns:
      - silhouette_score (if > 1 cluster)
      - median intra-cluster distance per cluster
      - median inter-cluster distance
      - noise_fraction
      - size histogram
    """

def queue_priority_stats(pair_queue: list[PairCandidate]) -> dict:
    """Distance distribution of the queue itself; fraction same-cluster; median distance."""
```

Phase 1 does NOT call `eval_onevsall` or `precision_at_k` because there is no GT. Phase 2 will add `core/metrics/gt.py` that wraps those functions.

## 9. `core/cluster/project.py`

```python
def run_umap(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> list[ProjectionPoint]:
    """2D UMAP projection for the web cluster view. Wraps umap-learn."""
```

`run_project_stage` reads embeddings parquet, calls `run_umap`, writes `projections/<run_id>.parquet`. Lives alongside DBSCAN/HDBSCAN in the cluster module since projections are conceptually a form of structural visualization.

## 10. Dependencies (pyproject.toml)

```toml
[project]
name = "whaleshark-reid"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "pydantic>=2.0",
  "numpy",
  "pandas",
  "pyarrow",           # parquet
  "scikit-learn",      # DBSCAN, StandardScaler
  "hdbscan",           # HDBSCAN alt
  "umap-learn",        # 2D projection
  "transformers",      # MiewID HuggingFace model
  "torch",             # inference
  "pillow",            # image loading
  "wbia_miew_id @ file:///workspace/wbia-plugin-miew-id",  # MiewIdDataset, get_test_transforms, extract_embeddings, metrics.* — pulls in torch, torchvision, albumentations transitively
]

[project.optional-dependencies]
dev = [
  "pytest",
  "pytest-cov",
  "ruff",
]
```

## Validation checkpoint (before moving to spec 02)

Create a notebook at `/workspace/catalog-match/whaleshark/verify_core.ipynb` that:

1. Imports `whaleshark_reid.core` modules.
2. Runs `ingest_inat_csv` on `whaleshark_inat_v1.csv` against a fresh temporary SQLite.
3. Runs `run_embed_stage` on the ingested annotations with `use_bbox=True`.
4. Runs `run_dbscan(embeddings, eps=0.7, min_samples=2, metric='cosine', standardize=True)`.
5. Runs `run_matching_stage` with `distance_threshold=1.0`.
6. Asserts the DBSCAN cluster count matches what `extract_and_evaluate_whalesharks.ipynb` produces on the same data (within sklearn determinism tolerance — probably exact match since both use the same sklearn version).
7. Asserts the distance distribution stats are within tolerance of the notebook.

If the notebook passes, core is solid. Move to spec 02.

## Success criteria

1. All modules in `src/whaleshark_reid/core/` and `src/whaleshark_reid/storage/` exist and have passing unit tests.
2. `verify_core.ipynb` reproduces the benchmark notebook's DBSCAN cluster count and distance distribution.
3. `pytest tests/core tests/storage` passes in under 3 seconds.
4. `Annotation` pydantic round-trips (dict → Annotation → dict is a no-op).
5. SQLite schema is fresh-creatable from `schema.sql` via `Storage(db_path).init_schema()`.
6. Re-running `ingest_inat_csv` on the same CSV is a no-op (`n_skipped_existing == n_read`).
7. Re-running `run_embed_stage` skips already-cached annotation_uuids.
8. `rebuild_individuals_cache` correctly materializes `annotations.name_uuid` from the current pair_decisions state: confirmed components all share one `name_uuid`, singletons are NULL.

## What this spec explicitly does NOT cover

- Typer CLI (spec 02).
- FastAPI app, routes, templates, service layer (spec 03).
- Subprocess orchestration for run-all (spec 02).
- Experiment comparison UI (spec 03).
- Phase 2: Wildbook ingest, GT metrics, calibration reports.
- Phase 3+: advanced matching experiments.
