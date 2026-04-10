# Whale Shark Re-ID — Spec 03: Web UI

**Date:** 2026-04-10
**Depends on:** [01-core-engine.md](2026-04-10-whaleshark-reid-01-core-engine.md)
**Related specs:** [overview](2026-04-10-whaleshark-reid-overview.md), [02-cli](2026-04-10-whaleshark-reid-02-cli-and-orchestration.md)

## Purpose

Build a single-user dev-mode web app for reviewing proposed whale shark matches, inspecting clusters, editing individuals, and comparing pipeline experiments. FastAPI + HTMX + vanilla JS, server-rendered Jinja templates, dark theme, raw-data metadata panels on every page.

The backend is designed so that a future React SPA swap is a pure replacement of `templates/` + `static/` — the `services/` layer is the stable seam.

## Module layout

```
src/whaleshark_reid/
└── web/
    ├── __init__.py
    ├── app.py                  # FastAPI factory, includes routers, mounts static
    ├── settings.py             # Host, port, db_path, cache_dir, title (pydantic-settings)
    ├── dependencies.py         # DI: get_storage(), get_settings()
    ├── services/               # Stack-agnostic service layer (pure Python, returns dataclasses)
    │   ├── __init__.py
    │   ├── pair_queue.py       # get_pair, submit_decision, get_next_undecided, filters
    │   ├── cluster_view.py     # get_projection, get_cluster_summary
    │   ├── annotations.py      # list_annotations, get_annotation
    │   ├── decisions.py        # list_decisions (read-only in Phase 1)
    │   ├── individuals.py      # list_individuals, get_individual, rename, split
    │   ├── experiments.py      # list_runs, get_run_detail, diff_runs
    │   ├── pipeline.py         # spawn CLI subprocess, stream stdout
    │   └── images.py           # Crop + serve image bytes
    ├── routes/
    │   ├── __init__.py
    │   ├── home.py             # /  → redirect to /review/pairs/<latest_run>
    │   ├── pairs.py            # /review/pairs/<run_id>, /api/pairs/<queue_id>/decide
    │   ├── list.py             # /list/annotations, /list/decisions, /list/individuals, /annotation/<uuid>, /individual/<name_uuid>
    │   ├── clusters.py         # /clusters/<run_id>
    │   ├── experiments.py      # /experiments, /run/<run_id>, /run/<a>/diff/<b>, /run/new (poll-based status)
    │   ├── map.py              # /map → Phase 2 stub
    │   └── image.py            # /image/<annotation_uuid>[?crop=true]
    ├── templates/
    │   ├── base.html           # Layout, dark theme, nav
    │   ├── partials/           # HTMX fragments
    │   │   ├── pair_card.html
    │   │   ├── annotation_row.html
    │   │   ├── decision_row.html
    │   │   ├── individual_row.html
    │   │   ├── run_row.html
    │   │   └── empty_queue.html
    │   ├── pairs/
    │   │   └── carousel.html
    │   ├── list/
    │   │   ├── annotations.html
    │   │   ├── decisions.html
    │   │   ├── individuals.html
    │   │   ├── annotation_detail.html
    │   │   └── individual_detail.html
    │   ├── clusters/
    │   │   └── scatter.html
    │   ├── experiments/
    │   │   ├── index.html
    │   │   ├── detail.html
    │   │   ├── diff.html
    │   │   └── run_new.html    # subprocess launcher form + poll-based status toast
    │   └── map/
    │       └── stub.html
    └── static/
        ├── css/
        │   └── app.css         # Dark theme, minimal, ~200 lines
        └── js/
            ├── carousel.js     # Keyboard shortcuts, HTMX swap coordination
            ├── leaflet_inset.js # Map inset init for pair carousel
            ├── plotly_scatter.js # Cluster view scatter init
            └── run_poll.js     # Poll runs.status every 2s to detect state transitions, show toast
```

## 1. App factory and settings

```python
# app.py
def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()
    app = FastAPI(title=settings.title, debug=False)
    app.state.settings = settings
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.include_router(home.router)
    app.include_router(pairs.router)
    app.include_router(list_router.router)
    app.include_router(clusters.router)
    app.include_router(experiments.router)
    app.include_router(map_router.router)
    app.include_router(image.router)
    return app
```

```python
# settings.py
class Settings(BaseSettings):
    title: str = "whaleshark-reid (dev)"
    host: str = "0.0.0.0"
    port: int = 8090
    db_path: Path = Path("cache/state.db")
    cache_dir: Path = Path("cache/")
    user: str = "dev"   # for pair_decisions.user column
    max_queue_page_size: int = 100
```

## 2. Service layer contracts

All services return pydantic dataclasses (JSON-able). Routes call services then render templates. A future React variant would add `/api/*` JSON routes that call the same services and return `.dict()`.

### `services/pair_queue.py`

```python
class PairView(BaseModel):
    queue_id: int
    run_id: str
    position: int
    total: int
    ann_a: AnnotationDetail
    ann_b: AnnotationDetail
    distance: float
    cluster_a: Optional[int]
    cluster_b: Optional[int]
    same_cluster: bool
    existing_decision: Optional[str]   # if user is revisiting
    cluster_context: ClusterContext    # same_cluster members + confirmed count
    gps_delta_km: Optional[float]
    time_delta_days: Optional[int]

class PairFilter(BaseModel):
    same_cluster: Optional[bool] = None
    min_distance: Optional[float] = None
    max_distance: Optional[float] = None
    unreviewed_only: bool = True

def get_pair(storage: Storage, run_id: str, position: int) -> Optional[PairView]: ...
def get_pair_by_id(storage: Storage, queue_id: int) -> Optional[PairView]: ...
def submit_decision(storage: Storage, queue_id: int, decision: str, user: str, notes: str = "") -> PairView: ...
def get_next_undecided(storage: Storage, run_id: str, from_position: int, filter: PairFilter) -> Optional[PairView]: ...
```

### `services/cluster_view.py`

```python
class ProjectionView(BaseModel):
    run_id: str
    algo: str
    params: dict
    points: list[ProjectionPoint]
    cluster_labels: dict[str, int]   # annotation_uuid -> cluster_label

class ClusterRow(BaseModel):
    cluster_label: int
    count: int
    median_intra_distance: Optional[float]
    n_confirmed_pairs: int
    sample_thumbnails: list[str]    # annotation_uuids, first 5

def get_projection(storage: Storage, cache_dir: Path, run_id: str) -> ProjectionView: ...
def get_cluster_summary(storage: Storage, cache_dir: Path, run_id: str) -> list[ClusterRow]: ...
def build_adhoc_queue(storage: Storage, cache_dir: Path, annotation_uuids: list[str]) -> str:
    """Build a temporary pair_queue from all pairs within the selected annotations.
    Creates a new synthetic run with stage='adhoc_queue', writes pair_queue rows under it,
    and returns the new run_id so the caller can redirect to /review/pairs/<run_id>.
    Ad-hoc runs are filtered out of the default experiments view unless explicitly shown."""
```

### `services/annotations.py`, `services/decisions.py`, `services/individuals.py`

```python
class AnnotationDetail(BaseModel):
    annotation_uuid: str
    image_uuid: str
    name_uuid: Optional[str]
    name: Optional[str]
    file_path: str
    file_name: str
    bbox: list[float]
    theta: float
    viewpoint: str
    species: str
    photographer: Optional[str]
    license: Optional[str]
    date_captured: Optional[str]
    gps_lat_captured: Optional[float]
    gps_lon_captured: Optional[float]
    observation_id: Optional[int]
    photo_index: Optional[int]
    conf: Optional[float]
    quality_grade: Optional[str]
    n_decisions_touching: int
    # ... all dev-mode fields surfaced

class AnnotationFilters(BaseModel):
    source: Optional[str] = None
    name_uuid: Optional[str] = None
    has_decisions: Optional[bool] = None
    search: Optional[str] = None

def list_annotations(storage: Storage, filters: AnnotationFilters, page: int, page_size: int) -> AnnotationPage: ...
def get_annotation(storage: Storage, annotation_uuid: str) -> Optional[AnnotationDetail]: ...

def list_decisions(storage: Storage, filters: DecisionFilters, page: int, page_size: int) -> DecisionPage: ...

def list_individuals(storage: Storage, page: int, page_size: int) -> IndividualPage: ...
def get_individual(storage: Storage, name_uuid: str) -> IndividualDetail: ...
def rename_individual(storage: Storage, name_uuid: str, new_name: str) -> None: ...
```

**Phase 2 deferred:** `supersede_decision` (editing past decisions) and `split_individual` (breaking off members from a derived individual) are deferred to Phase 2. Phase 1 only supports the forward flow: propose pair → yes/no/skip/unsure → append to pair_decisions. Mistakes are rare in early use and can be fixed by re-running `rebuild-individuals` after deleting the bad row directly via sqlite3 CLI.

### `services/experiments.py`

```python
class RunSummary(BaseModel):
    run_id: str
    stage: str
    algo: Optional[str]
    params_preview: str       # human-readable one-liner
    metrics_preview: dict     # flat key metrics for the table (extracted from metrics_json)
    status: str
    started_at: str
    finished_at: Optional[str]
    duration_s: Optional[float]

class RunDetail(BaseModel):
    run: RunSummary
    config: dict              # full runs.config_json
    metrics: dict             # full runs.metrics_json
    reproduce_cmd: str        # e.g. "catalog cluster --algo dbscan --eps 0.7 ..."
    log_tail: list[str]       # last N lines of cache_dir/logs/<run_id>.log

class RunDiff(BaseModel):
    run_a: RunDetail
    run_b: RunDetail
    config_diff: list[DiffLine]    # red/green per-key diff of config_json
    metrics_diff: list[DiffLine]   # red/green per-key diff of metrics_json

def list_runs(storage: Storage, filters: RunFilters) -> list[RunSummary]: ...
def get_run_detail(storage: Storage, cache_dir: Path, run_id: str) -> RunDetail: ...
def diff_runs(storage: Storage, cache_dir: Path, run_a: str, run_b: str) -> RunDiff: ...
```

### `services/pipeline.py`

Simple subprocess launcher with poll-based status. No SSE streaming — the UI polls `runs.status` every 2 seconds after triggering a pipeline command, and surfaces state transitions (`running → ok/failed`) as a toast. Log tail visible via the run detail page.

```python
def start_pipeline_subprocess(command: list[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    """Spawn `catalog <command>` as a detached subprocess, redirect stdout/stderr to log_path.
    Returns immediately with the Popen handle (caller doesn't wait)."""

def get_recent_run_status(storage: Storage, since_ts: str) -> list[RunSummary]:
    """Return all runs started since since_ts, used by the UI poller to detect state changes."""
```

### `services/images.py`

```python
def serve_annotation_image(
    storage: Storage,
    annotation_uuid: str,
    crop: bool = False,
) -> tuple[bytes, str]:
    """
    Returns (jpeg_bytes, content_type). If crop=True, uses bbox+theta via get_chip_from_img
    and resizes to max 800px. If the image is missing or cropping fails, falls back to the
    full image and sets X-Crop-Fallback=true via an out-of-band mechanism.
    """
```

## 3. Routes (thin adapters)

Every route follows the pattern:

```python
@router.get("/review/pairs/{run_id}", response_class=HTMLResponse)
def carousel(run_id: str, position: int = 1, request: Request = ..., storage: Storage = Depends(get_storage)):
    pair = pair_queue_service.get_pair(storage, run_id, position)
    if pair is None:
        return templates.TemplateResponse("partials/empty_queue.html", {"request": request, "run_id": run_id})
    return templates.TemplateResponse("pairs/carousel.html", {"request": request, "pair": pair})

@router.post("/api/pairs/{queue_id}/decide", response_class=HTMLResponse)
def decide(queue_id: int, decision: str = Form(...), notes: str = Form(""),
           request: Request = ..., storage: Storage = Depends(get_storage), settings: Settings = Depends(get_settings)):
    current = pair_queue_service.get_pair_by_id(storage, queue_id)
    if current is None:
        raise HTTPException(404)
    pair_queue_service.submit_decision(storage, queue_id, decision, settings.user, notes)
    next_pair = pair_queue_service.get_next_undecided(
        storage,
        run_id=current.run_id,
        from_position=current.position + 1,
        filter=PairFilter(),
    )
    if next_pair is None:
        return templates.TemplateResponse("partials/empty_queue.html", {"request": request, "run_id": current.run_id})
    return templates.TemplateResponse("partials/pair_card.html", {"request": request, "pair": next_pair})
```

HTMX requests are detected via the `HX-Request` header. GET routes return full pages; POST routes return fragments suitable for `hx-swap="outerHTML"`.

## 4. Templates

### `base.html`

Dark theme, persistent nav bar with links:
`whaleshark-reid (dev) · [Review] [List] [Clusters] [Experiments] · run: <latest_run_id>`

### `pairs/carousel.html`

Layout (reused as partial `pair_card.html` for HTMX swaps):

```
┌─ run abc123 · pair 37/847 · d=0.234 · same_cluster=yes ─────────┐
│                                                                 │
│  ┌────ann_a img────┐  ┌────ann_b img────┐  ┌──leaflet inset──┐  │
│  │                 │  │                 │  │ ●────●          │  │
│  │   (cropped)     │  │   (cropped)     │  │ 23km · 14d      │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                 │
│  ┌─ann_a metadata──┐  ┌─ann_b metadata──┐                      │
│  │ uuid ...        │  │ uuid ...        │                      │
│  │ obs_id 327286790│  │ obs_id 325113002│                      │
│  │ date 2025-11-19 │  │ date 2025-11-05 │                      │
│  │ gps 25.66 -80.17│  │ gps 25.71 -80.09│                      │
│  │ photographer ...│  │ photographer ...│                      │
│  │ bbox [...]      │  │ bbox [...]      │                      │
│  │ conf 0.94       │  │ conf 0.87       │                      │
│  │ license CC-BY   │  │ license CC-BY   │                      │
│  └─────────────────┘  └─────────────────┘                      │
│                                                                 │
│  [✓ Match (Y)] [✗ No (N)] [? Unsure (U)] [⏭ Skip (Space)]      │
│                                                                 │
│  ┌─ cluster context ─────────────────────────────────────┐     │
│  │ ann_a: cluster #42 (8 members, 3 confirmed)           │     │
│  │ ann_b: cluster #42 (8 members)                        │     │
│  │ [Show all cluster members →]                          │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  ← Prev (J)  filter: [all|same-cluster|...]   Next (K) →       │
└─────────────────────────────────────────────────────────────────┘
```

Data attributes on the root element: `data-pair-position="37"`, `data-pair-total="847"`, `data-run-id="abc123"`, `data-queue-id="...", data-annotation-uuid-a="..."`, `data-annotation-uuid-b="..."`, `data-distance="0.234"`. JS reads these to wire keyboard handlers and HTMX calls.

Decision buttons are `<button hx-post="/api/pairs/{queue_id}/decide" hx-vals='{"decision": "match"}' hx-target="#pair-card" hx-swap="outerHTML">`.

### `clusters/scatter.html`

Loads `/api/projections/<run_id>.json` (a JSON-returning endpoint backed by `services.cluster_view.get_projection`) and renders it with Plotly.js. Box-select + "Open in carousel" button calls `build_adhoc_queue` and redirects to `/review/pairs/<adhoc_queue_id>`.

### `experiments/index.html`, `detail.html`, `diff.html`, `run_new.html`

- **index.html** — table of runs with sortable columns. Row click → `/run/<run_id>`.
- **detail.html** — shows `config_json`, `metrics_json`, the reproduce CLI command (copy button), `log_tail`, linked children runs.
- **diff.html** — two columns for run_a and run_b; red/green diff of config keys and metrics.
- **run_new.html** — form to launch `catalog run-all` (or individual stages) with arg inputs. Form POSTs to `/run/new/start` which spawns the subprocess (log goes to `cache_dir/logs/<run_id>.log`) and returns the new `run_id`. The page then shows a toast that polls `runs.status` every 2 seconds; on transition to `ok`/`failed`, links to the run detail page where the log tail is visible.

## 5. Static JS (~50 LOC total)

### `carousel.js`

```js
// Keyboard shortcut wiring — reads data-shortcut attrs from the rendered buttons
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toLowerCase();
  const btn = document.querySelector(`[data-shortcut="${key}"]`);
  if (btn) { btn.click(); e.preventDefault(); }
});

// After HTMX swap: re-init map inset and cluster context
document.body.addEventListener('htmx:afterSwap', (e) => {
  if (e.detail.target.id === 'pair-card') {
    initMapInset();
  }
});
```

### `leaflet_inset.js`

Reads two GPS coordinates from data attrs on the inset div, initializes a Leaflet map bounded to fit both points, adds markers + a polyline. ~30 LOC.

### `plotly_scatter.js`

Fetches `/api/projections/<run_id>.json`, builds a Plotly scatter with hover + selection callbacks. ~40 LOC.

### `run_poll.js`

After the user kicks off a pipeline command from the run-new form, polls `GET /api/runs/<run_id>/status` every 2s. On state transition (`running` → `ok`/`failed`), shows a toast and links to the run detail page. Stops polling after terminal state or after 10 minutes, whichever comes first. ~15 LOC.

## 6. Image serving

```python
@router.get("/image/{annotation_uuid}")
def serve_image(
    annotation_uuid: str,
    crop: bool = False,
    storage: Storage = Depends(get_storage),
):
    try:
        body, content_type = images_service.serve_annotation_image(storage, annotation_uuid, crop=crop)
        return Response(content=body, media_type=content_type)
    except FileNotFoundError:
        raise HTTPException(404)
```

Crops are cached in memory via `lru_cache` keyed on `(annotation_uuid, crop)`. Cache is bounded to 256 entries.

## 7. Testing (four layers)

The testing strategy from Section 5 of the brainstorm lives here. Summary:

### Layer 1 — Core unit tests (`tests/core/`)
Covered in spec 01.

### Layer 2 — Storage tests (`tests/storage/`)
Covered in spec 01.

### Layer 3 — Full-pipeline integration test (`tests/integration/`)
`test_run_all_on_fixture.py`: runs all five pipeline stages via core entry points on a 10-annotation fixture, asserts parquet files exist, pair_queue populated, experiments rows written.

### Layer 4 — Web tests (`tests/web/`)

Using Starlette `TestClient` backed by a pre-built fixture DB (`tests/fixtures/fixture_state.db`):

- **`test_pair_carousel.py`** — route returns 200, HTML contains required `data-*` attrs, metadata panels include `observation_id`, `annotation_uuid`, `gps_lat_captured`; all four decision buttons present with correct `data-shortcut` values; POST to `/api/pairs/<id>/decide` with HX-Request header returns a fragment (no `<html>`), advances the queue, writes a `pair_decisions` row; empty queue route renders gracefully.
- **`test_list_views.py`** — annotations tab renders with pagination; decisions tab filters by decision type; individuals tab renames an individual and updates all member annotations; annotation detail and individual detail pages render.
- **`test_cluster_view.py`** — route returns 200; the `/api/projections/<run_id>.json` endpoint returns the expected JSON schema with every point having `x`, `y`, `cluster_label`, `annotation_uuid`; cluster summary table row count equals `n_clusters + 1` (noise row).
- **`test_experiments_view.py`** — runs list renders; run detail shows reproduce CLI command; diff endpoint returns a diff where at least one metric differs between two fixture runs.
- **`test_image_serving.py`** — full image and crop variants return JPEG; invalid UUID → 404; missing file → 404 with X-Crop-Fallback hint.
- **`test_htmx_fragment_shape.py`** — POST endpoints with HX-Request return fragments (no `<html>` tag), GET endpoints return full pages.

Target: all web tests pass in under 3 seconds.

### Error handling

(From Section 5 of the brainstorm.)

- Empty pair queue → `partials/empty_queue.html` with "re-run matching" link.
- Fully reviewed queue → same template variant with decision count.
- Cluster view with < 2 points → warning banner, scatter still renders.
- Image cropping failure → fallback to full image, `X-Crop-Fallback: true` header, template shows warning indicator.
- SQLite concurrency → WAL mode + busy_timeout.
- Name edit conflict → last-write-wins with `updated_at` check, banner if row changed since page load.
- Subprocess crash → run row marked `status='failed'` with exit code + error in `runs.error`; UI poller picks up the state change and shows a failure toast.

## 8. Deployment

Not a production concern. Run with:

```bash
cd /workspace/catalog-match/whaleshark-reid
uvicorn whaleshark_reid.web.app:create_app --factory --host 0.0.0.0 --port 8090
```

Container exposes port 8090 via docker-compose or `-p 8090:8090`. User opens `http://localhost:8090` in a host browser.

`whaleshark-reid/cache/` holds `state.db`, `embeddings/`, `clusters/`, `projections/`, `logs/`. This is the only mutable state. Back it up if it matters.

## 9. Dependencies (added to pyproject.toml)

```toml
dependencies = [
  # ... core deps from spec 01 ...
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "jinja2",
  "python-multipart",    # for form parsing
  "pydantic-settings",
  "rich",                # CLI logging (shared with spec 02)
]
```

Frontend assets (Plotly, Leaflet, HTMX) are **vendored locally** in `static/vendor/` — no CDN dependency so the app works offline.

## Success criteria

1. All five pages render on fixture data with no 500s.
2. Pair carousel decision round-trip works: click → decision persisted → next pair rendered via HTMX.
3. Keyboard shortcuts Y/N/U/Space/J/K work.
4. Map inset renders for pairs with GPS data.
5. Cluster view loads UMAP projection and renders Plotly scatter.
6. Experiments view diffs two runs correctly.
7. `/image/<uuid>?crop=true` returns a cropped JPEG.
8. Web test suite (Layer 4) passes in under 3 seconds.
9. `pytest` entire suite (all four layers) passes in under 10 seconds.
10. Swapping to a React SPA later requires zero changes to `services/` — only `templates/` and `static/` are replaced.

## What this spec explicitly does NOT cover

- Authentication / multi-user support.
- React implementation (just designed to be swappable).
- Phase 2: full map page, GT metrics views, calibration report page.
- Production deployment beyond uvicorn in a container.
- CI / CD pipelines.
