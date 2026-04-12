# Whaleshark Re-ID — Web UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-user dev-mode web app (FastAPI + HTMX + Jinja2) for reviewing whale shark pairwise matches, inspecting embedding clusters, browsing annotation/decision/individual tables, comparing pipeline experiment runs, and triggering new pipeline runs — all deployed inside the existing Docker container on port 8090.

**Architecture:** Server-rendered HTML via Jinja2 with HTMX for partial updates (no full SPA). A `web/services/` layer returns pydantic dataclasses consumed by thin route handlers that render templates. The services layer is the React-swap seam — if the UI is later rewritten as a SPA, only templates + static JS change; services stay. Static JS is ~100 LOC across 4 files (keyboard shortcuts, Leaflet map inset, Plotly scatter, pipeline poll). Dark theme, all metadata surfaces visible in dev-mode.

**Tech Stack:** FastAPI, Jinja2, HTMX (vendored), Leaflet (vendored), Plotly.js (vendored), Pydantic-Settings, uvicorn, TestClient (starlette) for tests. All existing Phase 1 + 2 dependencies remain.

**Pre-flight:** Phase 2 CLI is complete (29 commits on main, 102 tests passing). Spec at `docs/superpowers/specs/2026-04-10-whaleshark-reid-03-web-ui.md`. The CLI commands (`catalog ingest`, `embed`, `cluster`, `matching`, `project`, `rebuild-individuals`, `run-all`, `status`) are all callable via subprocess from the web app. `Storage` has public methods for annotation CRUD, run CRUD, pair_queue write, pair_decisions read, and a `transaction()` context manager.

**Execution notes:**
- Every task begins with `cd /workspace/catalog-match/whaleshark-reid` unless stated otherwise.
- `pytest -x` after each task. Commit after green.
- For vendored JS libraries (HTMX, Leaflet, Plotly), download the minified single-file build via `curl` and save to `static/vendor/`. No npm, no build step.
- Templates use Jinja2 with `{% extends "base.html" %}` and `{% block content %}` pattern.

---

## Task 1: Web scaffolding — deps, app factory, settings, base template, CSS, vendor libs

**Files:**
- Modify: `pyproject.toml` (add fastapi, uvicorn, jinja2, python-multipart, pydantic-settings)
- Create: `src/whaleshark_reid/web/__init__.py`
- Create: `src/whaleshark_reid/web/app.py`
- Create: `src/whaleshark_reid/web/settings.py`
- Create: `src/whaleshark_reid/web/dependencies.py`
- Create: `src/whaleshark_reid/web/templates/base.html`
- Create: `src/whaleshark_reid/web/static/css/app.css`
- Create: `src/whaleshark_reid/web/static/vendor/` (htmx.min.js)
- Create: `tests/web/__init__.py`
- Create: `tests/web/conftest.py`
- Create: `tests/web/test_web_smoke.py`

- [ ] **Step 1: Add web deps to pyproject.toml**

Append to the `dependencies` list in `pyproject.toml`:
```
  "fastapi>=0.110",
  "uvicorn[standard]>=0.27",
  "jinja2",
  "python-multipart",
  "pydantic-settings",
  "httpx",
```

Also add to `[tool.setuptools.package-data]`:
```toml
[tool.setuptools.package-data]
whaleshark_reid = ["storage/schema.sql", "web/templates/**/*.html", "web/static/**/*"]
```

Then reinstall: `pip install -e ".[dev]"`

- [ ] **Step 2: Create the Settings class**

`src/whaleshark_reid/web/__init__.py`:
```python
"""Web UI package — FastAPI + HTMX + Jinja2."""
```

`src/whaleshark_reid/web/settings.py`:
```python
"""App-wide settings loaded from environment or defaults."""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    title: str = "whaleshark-reid (dev)"
    host: str = "0.0.0.0"
    port: int = 8090
    db_path: Path = Path("cache/state.db")
    cache_dir: Path = Path("cache/")
    user: str = "dev"
    max_queue_page_size: int = 100

    model_config = {"env_prefix": "WHALESHARK_"}
```

- [ ] **Step 3: Create the dependency injection helpers**

`src/whaleshark_reid/web/dependencies.py`:
```python
"""FastAPI dependency injection — Storage and Settings singletons."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.settings import Settings


@lru_cache(maxsize=1)
def _get_settings() -> Settings:
    return Settings()


_storage_instance: Storage | None = None


def get_settings() -> Settings:
    return _get_settings()


def get_storage() -> Storage:
    global _storage_instance
    if _storage_instance is None:
        settings = get_settings()
        _storage_instance = Storage(settings.db_path)
        _storage_instance.init_schema()
    return _storage_instance


def override_storage(storage: Storage) -> None:
    """For tests: inject a pre-configured Storage instance."""
    global _storage_instance
    _storage_instance = storage


def override_settings(settings: Settings) -> None:
    """For tests: inject custom settings."""
    _get_settings.cache_clear()
    # Monkey-patch the cached getter
    import whaleshark_reid.web.dependencies as mod
    mod._get_settings = lru_cache(maxsize=1)(lambda: settings)
```

- [ ] **Step 4: Create the FastAPI app factory**

`src/whaleshark_reid/web/app.py`:
```python
"""FastAPI application factory."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

TEMPLATES_DIR = Path(__file__).parent / "templates"
STATIC_DIR = Path(__file__).parent / "static"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def create_app() -> FastAPI:
    from whaleshark_reid.web.settings import Settings

    settings = Settings()

    app = FastAPI(title=settings.title, docs_url=None, redoc_url=None)
    app.state.settings = settings

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Routes will be added in subsequent tasks via app.include_router(...)
    # For now, a minimal health check:
    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
```

- [ ] **Step 5: Create the base template + dark theme CSS**

`src/whaleshark_reid/web/templates/base.html`:
```html
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}whaleshark-reid{% endblock %}</title>
    <link rel="stylesheet" href="/static/css/app.css">
    <script src="/static/vendor/htmx.min.js" defer></script>
    {% block head_extra %}{% endblock %}
</head>
<body>
    <nav class="top-nav">
        <span class="brand">whaleshark-reid <span class="tag">dev</span></span>
        <a href="/review/pairs/">Review</a>
        <a href="/list/annotations">List</a>
        <a href="/clusters/">Clusters</a>
        <a href="/experiments">Experiments</a>
    </nav>
    <main>
        {% block content %}{% endblock %}
    </main>
</body>
</html>
```

`src/whaleshark_reid/web/static/css/app.css`:
```css
/* Dark theme for whaleshark-reid dev UI */
:root {
    --bg: #1a1a2e;
    --bg-card: #16213e;
    --bg-input: #0f3460;
    --text: #e0e0e0;
    --text-muted: #888;
    --accent: #00adb5;
    --success: #4caf50;
    --danger: #f44336;
    --warning: #ff9800;
    --border: #333;
    --radius: 6px;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    background: var(--bg);
    color: var(--text);
    font-family: 'SF Mono', 'Menlo', 'Consolas', monospace;
    font-size: 13px;
    line-height: 1.5;
}

.top-nav {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.5rem 1rem;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
}

.top-nav .brand { font-weight: bold; color: var(--accent); }
.top-nav .tag { font-size: 10px; color: var(--text-muted); }
.top-nav a { color: var(--text); text-decoration: none; }
.top-nav a:hover { color: var(--accent); }

main { padding: 1rem; max-width: 1400px; margin: 0 auto; }

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Tables */
table { width: 100%; border-collapse: collapse; }
th, td { padding: 0.4rem 0.6rem; text-align: left; border-bottom: 1px solid var(--border); }
th { color: var(--text-muted); font-weight: normal; text-transform: uppercase; font-size: 11px; }
tr:hover { background: rgba(255,255,255,0.03); }

/* Buttons */
.btn {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg-input);
    color: var(--text);
    cursor: pointer;
    font-family: inherit;
    font-size: 12px;
}
.btn:hover { border-color: var(--accent); }
.btn-match { border-color: var(--success); color: var(--success); }
.btn-nomatch { border-color: var(--danger); color: var(--danger); }
.btn-unsure { border-color: var(--warning); color: var(--warning); }

/* Metadata panels */
.meta-panel {
    font-size: 11px;
    color: var(--text-muted);
}
.meta-panel dt { display: inline; font-weight: bold; color: var(--text); }
.meta-panel dd { display: inline; margin-left: 0.3rem; margin-right: 1rem; }

/* Images */
.ann-img { max-width: 100%; height: auto; border-radius: var(--radius); }

/* Pair carousel layout */
.pair-layout {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 1rem;
    align-items: start;
}

/* Map inset */
.map-inset { width: 300px; height: 200px; border-radius: var(--radius); }

/* Cluster scatter */
.scatter-container { width: 100%; height: 500px; }

/* Toast notifications */
.toast {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    padding: 0.8rem 1.2rem;
    border-radius: var(--radius);
    background: var(--bg-card);
    border: 1px solid var(--accent);
    z-index: 1000;
}
```

- [ ] **Step 6: Vendor HTMX**

```bash
cd /workspace/catalog-match/whaleshark-reid
mkdir -p src/whaleshark_reid/web/static/vendor
curl -sL https://unpkg.com/htmx.org@2.0.4/dist/htmx.min.js -o src/whaleshark_reid/web/static/vendor/htmx.min.js
ls -la src/whaleshark_reid/web/static/vendor/htmx.min.js
```

Expected: file exists, ~50KB.

- [ ] **Step 7: Write the failing smoke test**

`tests/web/__init__.py`:
```python
```

`tests/web/conftest.py`:
```python
"""Shared fixtures for web tests."""
from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import create_app
from whaleshark_reid.web.dependencies import override_settings, override_storage
from whaleshark_reid.web.settings import Settings


@pytest.fixture
def web_client(tmp_db_path: Path, tmp_cache_dir: Path) -> TestClient:
    """TestClient backed by a fresh tmp DB."""
    storage = Storage(tmp_db_path)
    storage.init_schema()
    override_storage(storage)
    override_settings(Settings(db_path=tmp_db_path, cache_dir=tmp_cache_dir))
    app = create_app()
    return TestClient(app)


@pytest.fixture
def seeded_web_client(tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid) -> TestClient:
    """TestClient with 10 ingested + embedded + clustered + matched annotations."""
    from whaleshark_reid.core.cluster.common import run_cluster_stage
    from whaleshark_reid.core.embed.miewid import run_embed_stage
    from whaleshark_reid.core.ingest.inat import ingest_inat_csv
    from whaleshark_reid.core.matching.pairs import run_matching_stage
    from whaleshark_reid.core.cluster.project import run_project_stage

    storage = Storage(tmp_db_path)
    storage.init_schema()

    fixtures = Path(__file__).parent.parent / "fixtures"
    ingest_inat_csv(
        csv_path=fixtures / "mini_inat.csv",
        photos_dir=fixtures / "photos",
        storage=storage,
        run_id="r_ingest",
    )

    storage.begin_run("r_embed", "embed", config={})
    embed_result = run_embed_stage(
        storage=storage, cache_dir=tmp_cache_dir, run_id="r_embed",
        batch_size=4, num_workers=0, device="cpu",
    )
    storage.finish_run("r_embed", "ok", metrics=embed_result.model_dump())

    storage.begin_run("r_cluster", "cluster", config={})
    cluster_result = run_cluster_stage(
        cache_dir=tmp_cache_dir, embedding_run_id="r_embed",
        cluster_run_id="r_cluster", algo="dbscan",
        params={"eps": 0.7, "min_samples": 2, "metric": "cosine", "standardize": True},
    )
    storage.finish_run("r_cluster", "ok", metrics=cluster_result.model_dump())

    storage.begin_run("r_match", "matching", config={})
    match_result = run_matching_stage(
        storage=storage, cache_dir=tmp_cache_dir,
        matching_run_id="r_match", embedding_run_id="r_embed",
        cluster_run_id="r_cluster", distance_threshold=2.0, max_queue_size=100,
    )
    storage.finish_run("r_match", "ok", metrics=match_result.model_dump())

    run_project_stage(
        cache_dir=tmp_cache_dir, embedding_run_id="r_embed",
        projection_run_id="r_project",
    )

    override_storage(storage)
    override_settings(Settings(db_path=tmp_db_path, cache_dir=tmp_cache_dir))
    app = create_app()
    return TestClient(app)
```

`tests/web/test_web_smoke.py`:
```python
"""Smoke tests for the web app factory."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_health_endpoint(web_client: TestClient):
    r = web_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_static_css_accessible(web_client: TestClient):
    r = web_client.get("/static/css/app.css")
    assert r.status_code == 200
    assert "var(--bg)" in r.text


def test_static_htmx_accessible(web_client: TestClient):
    r = web_client.get("/static/vendor/htmx.min.js")
    assert r.status_code == 200
```

- [ ] **Step 8: Run failing test, then implement, then verify**

```bash
pytest tests/web/test_web_smoke.py -x
```

After implementing: should show 3 passed. Full suite should be ~105.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/whaleshark_reid/web/ tests/web/
git commit -m "chore: web UI scaffolding — FastAPI app factory, dark theme, HTMX vendor"
```

---

## Task 2: Image serving service + route

**Files:**
- Create: `src/whaleshark_reid/web/services/__init__.py`
- Create: `src/whaleshark_reid/web/services/images.py`
- Create: `src/whaleshark_reid/web/routes/__init__.py`
- Create: `src/whaleshark_reid/web/routes/image.py`
- Modify: `src/whaleshark_reid/web/app.py` (include router)
- Create: `tests/web/test_image_serving.py`

- [ ] **Step 1: Write failing test**

`tests/web/test_image_serving.py`:
```python
"""Tests for /image/<uuid> endpoint."""
from __future__ import annotations

from starlette.testclient import TestClient

from whaleshark_reid.core.schema import inat_annotation_uuid


def test_serve_full_image(seeded_web_client: TestClient):
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/image/{uuid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_serve_cropped_image(seeded_web_client: TestClient):
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/image/{uuid}?crop=true")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_missing_uuid_returns_404(seeded_web_client: TestClient):
    r = seeded_web_client.get("/image/does-not-exist")
    assert r.status_code == 404
```

- [ ] **Step 2: Implement service + route**

`src/whaleshark_reid/web/services/__init__.py`:
```python
"""Web service layer — stack-agnostic functions that return pydantic models."""
```

`src/whaleshark_reid/web/services/images.py`:
```python
"""Image serving: load annotation image, optionally crop to bbox+theta."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from whaleshark_reid.storage.db import Storage


@lru_cache(maxsize=256)
def _load_and_crop(file_path: str, bbox_tuple: tuple, theta: float, crop: bool) -> bytes:
    img = Image.open(file_path).convert("RGB")

    if crop and bbox_tuple:
        try:
            from wbia_miew_id.datasets.helpers import get_chip_from_img
            img_arr = np.array(img)
            bbox = list(bbox_tuple)
            chip = get_chip_from_img(img_arr, bbox, theta)
            img = Image.fromarray(chip)
        except Exception:
            pass  # fallback to full image on any crop error

    # Resize to max 800px on longest side
    max_dim = max(img.size)
    if max_dim > 800:
        scale = 800 / max_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def serve_annotation_image(
    storage: Storage,
    annotation_uuid: str,
    crop: bool = False,
) -> tuple[bytes, str]:
    """Returns (jpeg_bytes, content_type). Raises FileNotFoundError if annotation or image missing."""
    ann = storage.get_annotation(annotation_uuid)
    if ann is None:
        raise FileNotFoundError(f"No annotation with uuid {annotation_uuid}")

    file_path = ann.file_path
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    bbox_tuple = tuple(ann.bbox) if ann.bbox else ()
    jpeg_bytes = _load_and_crop(file_path, bbox_tuple, ann.theta, crop)
    return jpeg_bytes, "image/jpeg"
```

`src/whaleshark_reid/web/routes/__init__.py`:
```python
"""Route modules — thin adapters between HTTP and the service layer."""
```

`src/whaleshark_reid/web/routes/image.py`:
```python
"""Image serving route: /image/<annotation_uuid>."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_storage
from whaleshark_reid.web.services import images as images_service

router = APIRouter()


@router.get("/image/{annotation_uuid}")
def serve_image(
    annotation_uuid: str,
    crop: bool = False,
    storage: Storage = Depends(get_storage),
):
    try:
        body, content_type = images_service.serve_annotation_image(
            storage, annotation_uuid, crop=crop
        )
        return Response(content=body, media_type=content_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
```

- [ ] **Step 3: Register router in app.py**

Add to `create_app()` in `app.py`, before the `return app`:
```python
    from whaleshark_reid.web.routes import image
    app.include_router(image.router)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/web/test_image_serving.py -x
```
Expected: 3 passed. Full suite: ~108.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/web/services/ src/whaleshark_reid/web/routes/ tests/web/test_image_serving.py src/whaleshark_reid/web/app.py
git commit -m "feat: image serving endpoint with bbox crop + LRU cache"
```

---

## Task 3: Pair queue service + carousel page (the core review surface)

This is the largest single task — it delivers the pair review carousel that is the centerpiece of the Phase 3 UI. It includes: the service layer for pair_queue queries + decision submission, the carousel route, the carousel template with pair_card partial, and tests.

**Files:**
- Create: `src/whaleshark_reid/web/services/pair_queue.py`
- Create: `src/whaleshark_reid/web/routes/pairs.py`
- Create: `src/whaleshark_reid/web/templates/pairs/carousel.html`
- Create: `src/whaleshark_reid/web/templates/partials/pair_card.html`
- Create: `src/whaleshark_reid/web/templates/partials/empty_queue.html`
- Modify: `src/whaleshark_reid/web/app.py` (include router)
- Create: `tests/web/test_pair_carousel.py`

- [ ] **Step 1: Write failing test**

`tests/web/test_pair_carousel.py`:
```python
"""Tests for the pair carousel page and HTMX decision flow."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_carousel_renders_first_pair(seeded_web_client: TestClient):
    r = seeded_web_client.get("/review/pairs/r_match")
    assert r.status_code == 200
    assert "data-pair-position" in r.text
    assert "data-distance" in r.text
    # Decision buttons present
    assert "Match" in r.text
    assert "No match" in r.text


def test_submit_decision_returns_next_pair(seeded_web_client: TestClient):
    # Get the first pair to find the queue_id
    page = seeded_web_client.get("/review/pairs/r_match")
    assert page.status_code == 200

    # Extract queue_id from HTML — look for data-queue-id attribute
    import re
    match = re.search(r'data-queue-id="(\d+)"', page.text)
    assert match, "No data-queue-id found in carousel HTML"
    queue_id = match.group(1)

    # Submit a decision via HTMX POST
    r = seeded_web_client.post(
        f"/api/pairs/{queue_id}/decide",
        data={"decision": "match"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    # HTMX fragment — no full <html> tag
    assert "<html" not in r.text
    # Should be either a new pair_card or empty_queue
    assert "data-pair-position" in r.text or "No pairs" in r.text


def test_empty_queue_renders_gracefully(web_client: TestClient):
    # No matching run → should handle gracefully
    r = web_client.get("/review/pairs/nonexistent")
    assert r.status_code == 200
    assert "No pairs" in r.text or "no pairs" in r.text.lower()
```

- [ ] **Step 2: Implement the pair queue service**

`src/whaleshark_reid/web/services/pair_queue.py`:
```python
"""Pair queue service — read queue entries, submit decisions, navigate."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.core.schema import Annotation
from whaleshark_reid.storage.db import Storage


class PairView(BaseModel):
    queue_id: int
    run_id: str
    position: int
    total: int
    ann_a: Annotation
    ann_b: Annotation
    distance: float
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    same_cluster: bool = False
    gps_delta_km: Optional[float] = None
    time_delta_days: Optional[int] = None


def _pair_from_row(row, storage: Storage, total: int) -> PairView:
    ann_a = storage.get_annotation(row["ann_a_uuid"])
    ann_b = storage.get_annotation(row["ann_b_uuid"])

    gps_delta = None
    time_delta = None
    if ann_a and ann_b:
        if ann_a.gps_lat_captured and ann_b.gps_lat_captured:
            from math import radians, sin, cos, asin, sqrt
            lat1, lon1 = radians(ann_a.gps_lat_captured), radians(ann_a.gps_lon_captured or 0)
            lat2, lon2 = radians(ann_b.gps_lat_captured), radians(ann_b.gps_lon_captured or 0)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            gps_delta = round(2 * 6371 * asin(sqrt(a)), 1)

        if ann_a.date_captured and ann_b.date_captured:
            try:
                d1 = datetime.fromisoformat(ann_a.date_captured)
                d2 = datetime.fromisoformat(ann_b.date_captured)
                time_delta = abs((d2 - d1).days)
            except (ValueError, TypeError):
                pass

    return PairView(
        queue_id=row["queue_id"],
        run_id=row["run_id"],
        position=row["position"],
        total=total,
        ann_a=ann_a,
        ann_b=ann_b,
        distance=row["distance"],
        cluster_a=row["cluster_a"],
        cluster_b=row["cluster_b"],
        same_cluster=bool(row["same_cluster"]),
        gps_delta_km=gps_delta,
        time_delta_days=time_delta,
    )


def get_pair(storage: Storage, run_id: str, position: int) -> Optional[PairView]:
    total = storage.count("pair_queue", run_id=run_id)
    if total == 0:
        return None
    row = storage.conn.execute(
        "SELECT * FROM pair_queue WHERE run_id = ? AND position = ?",
        (run_id, position),
    ).fetchone()
    if row is None:
        return None
    return _pair_from_row(row, storage, total)


def get_pair_by_id(storage: Storage, queue_id: int) -> Optional[PairView]:
    row = storage.conn.execute(
        "SELECT * FROM pair_queue WHERE queue_id = ?", (queue_id,)
    ).fetchone()
    if row is None:
        return None
    total = storage.count("pair_queue", run_id=row["run_id"])
    return _pair_from_row(row, storage, total)


def submit_decision(
    storage: Storage, queue_id: int, decision: str, user: str, notes: str = ""
) -> None:
    pair = storage.conn.execute(
        "SELECT ann_a_uuid, ann_b_uuid, distance, run_id FROM pair_queue WHERE queue_id = ?",
        (queue_id,),
    ).fetchone()
    if pair is None:
        return
    storage.conn.execute(
        """
        INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, distance, run_id, user, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pair["ann_a_uuid"], pair["ann_b_uuid"], decision,
            pair["distance"], pair["run_id"], user, notes,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def get_next_undecided(
    storage: Storage, run_id: str, from_position: int
) -> Optional[PairView]:
    total = storage.count("pair_queue", run_id=run_id)
    row = storage.conn.execute(
        """
        SELECT pq.* FROM pair_queue pq
        WHERE pq.run_id = ? AND pq.position >= ?
        AND NOT EXISTS (
            SELECT 1 FROM pair_decisions pd
            WHERE pd.ann_a_uuid = pq.ann_a_uuid AND pd.ann_b_uuid = pq.ann_b_uuid
            AND pd.decision IN ('match', 'no_match')
            AND pd.superseded_by IS NULL
        )
        ORDER BY pq.position ASC
        LIMIT 1
        """,
        (run_id, from_position),
    ).fetchone()
    if row is None:
        return None
    return _pair_from_row(row, storage, total)
```

- [ ] **Step 3: Implement the pairs route**

`src/whaleshark_reid/web/routes/pairs.py`:
```python
"""Pair review carousel routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import pair_queue as pq_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


@router.get("/review/pairs/{run_id}", response_class=HTMLResponse)
def carousel(
    request: Request,
    run_id: str,
    position: int = 0,
    storage: Storage = Depends(get_storage),
):
    pair = pq_service.get_pair(storage, run_id, position)
    if pair is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": run_id},
        )
    return templates.TemplateResponse(
        "pairs/carousel.html",
        {"request": request, "pair": pair},
    )


@router.post("/api/pairs/{queue_id}/decide", response_class=HTMLResponse)
def decide(
    request: Request,
    queue_id: int,
    decision: str = Form(...),
    notes: str = Form(""),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    current = pq_service.get_pair_by_id(storage, queue_id)
    if current is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": "unknown"},
        )

    pq_service.submit_decision(storage, queue_id, decision, settings.user, notes)

    next_pair = pq_service.get_next_undecided(
        storage, run_id=current.run_id, from_position=current.position + 1
    )
    if next_pair is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": current.run_id},
        )
    return templates.TemplateResponse(
        "partials/pair_card.html",
        {"request": request, "pair": next_pair},
    )
```

- [ ] **Step 4: Create the templates**

`src/whaleshark_reid/web/templates/pairs/carousel.html`:
```html
{% extends "base.html" %}
{% block title %}Review · {{ pair.run_id }}{% endblock %}
{% block head_extra %}
<script src="/static/js/carousel.js" defer></script>
{% endblock %}
{% block content %}
<div id="pair-card">
    {% include "partials/pair_card.html" %}
</div>
{% endblock %}
```

`src/whaleshark_reid/web/templates/partials/pair_card.html`:
```html
<div class="card"
     data-pair-position="{{ pair.position }}"
     data-pair-total="{{ pair.total }}"
     data-run-id="{{ pair.run_id }}"
     data-queue-id="{{ pair.queue_id }}"
     data-annotation-uuid-a="{{ pair.ann_a.annotation_uuid }}"
     data-annotation-uuid-b="{{ pair.ann_b.annotation_uuid }}"
     data-distance="{{ '%.3f'|format(pair.distance) }}">

    <div style="display:flex; gap:0.5rem; align-items:center; margin-bottom:1rem; color:var(--text-muted);">
        <span>pair {{ pair.position + 1 }}/{{ pair.total }}</span>
        <span>·</span>
        <span>d={{ '%.3f'|format(pair.distance) }}</span>
        {% if pair.same_cluster %}<span style="color:var(--success)">· same cluster</span>{% endif %}
        {% if pair.gps_delta_km is not none %}<span>· {{ pair.gps_delta_km }}km</span>{% endif %}
        {% if pair.time_delta_days is not none %}<span>· {{ pair.time_delta_days }}d</span>{% endif %}
    </div>

    <div class="pair-layout">
        <div>
            <img class="ann-img" src="/image/{{ pair.ann_a.annotation_uuid }}?crop=true" alt="ann_a">
            <dl class="meta-panel">
                <dt>uuid</dt><dd>{{ pair.ann_a.annotation_uuid[:12] }}…</dd>
                <dt>obs</dt><dd>{{ pair.ann_a.observation_id }}</dd>
                <dt>date</dt><dd>{{ pair.ann_a.date_captured or '—' }}</dd>
                <dt>gps</dt><dd>{{ pair.ann_a.gps_lat_captured or '—' }}, {{ pair.ann_a.gps_lon_captured or '—' }}</dd>
                <dt>conf</dt><dd>{{ '%.2f'|format(pair.ann_a.conf) if pair.ann_a.conf else '—' }}</dd>
                <dt>photographer</dt><dd>{{ pair.ann_a.photographer or '—' }}</dd>
            </dl>
        </div>
        <div>
            <img class="ann-img" src="/image/{{ pair.ann_b.annotation_uuid }}?crop=true" alt="ann_b">
            <dl class="meta-panel">
                <dt>uuid</dt><dd>{{ pair.ann_b.annotation_uuid[:12] }}…</dd>
                <dt>obs</dt><dd>{{ pair.ann_b.observation_id }}</dd>
                <dt>date</dt><dd>{{ pair.ann_b.date_captured or '—' }}</dd>
                <dt>gps</dt><dd>{{ pair.ann_b.gps_lat_captured or '—' }}, {{ pair.ann_b.gps_lon_captured or '—' }}</dd>
                <dt>conf</dt><dd>{{ '%.2f'|format(pair.ann_b.conf) if pair.ann_b.conf else '—' }}</dd>
                <dt>photographer</dt><dd>{{ pair.ann_b.photographer or '—' }}</dd>
            </dl>
        </div>
        <div>
            <div class="map-inset" id="map-inset"
                 data-lat-a="{{ pair.ann_a.gps_lat_captured or '' }}"
                 data-lon-a="{{ pair.ann_a.gps_lon_captured or '' }}"
                 data-lat-b="{{ pair.ann_b.gps_lat_captured or '' }}"
                 data-lon-b="{{ pair.ann_b.gps_lon_captured or '' }}">
                <span style="color:var(--text-muted)">map (Phase 3+)</span>
            </div>
        </div>
    </div>

    <div style="display:flex; gap:0.5rem; margin-top:1rem;">
        <button class="btn btn-match" data-shortcut="y"
                hx-post="/api/pairs/{{ pair.queue_id }}/decide"
                hx-vals='{"decision": "match"}'
                hx-target="#pair-card" hx-swap="innerHTML">✓ Match (Y)</button>
        <button class="btn btn-nomatch" data-shortcut="n"
                hx-post="/api/pairs/{{ pair.queue_id }}/decide"
                hx-vals='{"decision": "no_match"}'
                hx-target="#pair-card" hx-swap="innerHTML">✗ No match (N)</button>
        <button class="btn btn-unsure" data-shortcut="u"
                hx-post="/api/pairs/{{ pair.queue_id }}/decide"
                hx-vals='{"decision": "unsure"}'
                hx-target="#pair-card" hx-swap="innerHTML">? Unsure (U)</button>
        <button class="btn" data-shortcut=" "
                hx-post="/api/pairs/{{ pair.queue_id }}/decide"
                hx-vals='{"decision": "skip"}'
                hx-target="#pair-card" hx-swap="innerHTML">⏭ Skip (Space)</button>
    </div>

    <div style="display:flex; justify-content:space-between; margin-top:1rem; color:var(--text-muted);">
        {% if pair.position > 0 %}
        <a href="/review/pairs/{{ pair.run_id }}?position={{ pair.position - 1 }}">← Prev (J)</a>
        {% else %}<span></span>{% endif %}
        {% if pair.position < pair.total - 1 %}
        <a href="/review/pairs/{{ pair.run_id }}?position={{ pair.position + 1 }}">Next (K) →</a>
        {% else %}<span></span>{% endif %}
    </div>
</div>
```

`src/whaleshark_reid/web/templates/partials/empty_queue.html`:
```html
<div class="card" style="text-align:center; padding:3rem;">
    <h2 style="color:var(--text-muted);">No pairs to review</h2>
    <p style="color:var(--text-muted); margin-top:0.5rem;">
        Run <code>catalog matching</code> to generate a pair queue, or adjust the distance threshold.
    </p>
</div>
```

- [ ] **Step 5: Create the carousel keyboard shortcuts JS**

`src/whaleshark_reid/web/static/js/carousel.js`:
```js
// Keyboard shortcut wiring for the pair review carousel.
// Reads data-shortcut attrs from buttons and clicks them on keypress.
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  const key = e.key.toLowerCase();

  // J/K for prev/next navigation
  if (key === 'j') {
    const prev = document.querySelector('a[href*="position="]');
    if (prev && prev.textContent.includes('Prev')) { prev.click(); e.preventDefault(); }
    return;
  }
  if (key === 'k') {
    const links = document.querySelectorAll('a[href*="position="]');
    const next = Array.from(links).find(a => a.textContent.includes('Next'));
    if (next) { next.click(); e.preventDefault(); }
    return;
  }

  // Decision shortcuts: Y, N, U, Space
  const btn = document.querySelector(`[data-shortcut="${key}"]`);
  if (btn) { btn.click(); e.preventDefault(); }
});
```

- [ ] **Step 6: Register the pairs router in app.py**

Add to `create_app()`:
```python
    from whaleshark_reid.web.routes import pairs
    app.include_router(pairs.router)
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/web/test_pair_carousel.py -x
```
Expected: 3 passed. Full suite: ~111.

- [ ] **Step 8: Commit**

```bash
git add src/whaleshark_reid/web/ tests/web/test_pair_carousel.py
git commit -m "feat: pair review carousel with HTMX decisions + keyboard shortcuts"
```

---

## Task 4: Annotations list + detail pages

**Files:**
- Create: `src/whaleshark_reid/web/services/annotations.py`
- Create: `src/whaleshark_reid/web/routes/list.py`
- Create: `src/whaleshark_reid/web/templates/list/annotations.html`
- Create: `src/whaleshark_reid/web/templates/list/annotation_detail.html`
- Modify: `src/whaleshark_reid/web/app.py` (include router)
- Create: `tests/web/test_list_views.py`

The list views are the "database browser" surface. This task implements annotations listing + detail. Subsequent tasks will add decisions + individuals tabs.

- [ ] **Step 1: Write failing test**

`tests/web/test_list_views.py`:
```python
"""Tests for list views — annotations, decisions, individuals."""
from __future__ import annotations

from starlette.testclient import TestClient

from whaleshark_reid.core.schema import inat_annotation_uuid


def test_annotations_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/list/annotations")
    assert r.status_code == 200
    assert "annotation_uuid" in r.text.lower() or "Annotations" in r.text


def test_annotation_detail_renders(seeded_web_client: TestClient):
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/annotation/{uuid}")
    assert r.status_code == 200
    assert uuid[:12] in r.text


def test_annotation_detail_404_for_missing(seeded_web_client: TestClient):
    r = seeded_web_client.get("/annotation/does-not-exist")
    assert r.status_code == 404
```

- [ ] **Step 2: Implement service + route + templates**

`src/whaleshark_reid/web/services/annotations.py`:
```python
"""Annotation listing and detail service."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.core.schema import Annotation
from whaleshark_reid.storage.db import Storage


class AnnotationPage(BaseModel):
    items: list[Annotation]
    total: int
    page: int
    page_size: int


def list_annotations(
    storage: Storage, page: int = 0, page_size: int = 50
) -> AnnotationPage:
    total = storage.count("annotations")
    rows = storage.conn.execute(
        "SELECT * FROM annotations ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, page * page_size),
    ).fetchall()
    items = [storage._row_to_annotation(r) for r in rows]
    return AnnotationPage(items=items, total=total, page=page, page_size=page_size)


def get_annotation_detail(
    storage: Storage, annotation_uuid: str
) -> Optional[Annotation]:
    return storage.get_annotation(annotation_uuid)
```

`src/whaleshark_reid/web/routes/list.py`:
```python
"""List views: annotations, decisions, individuals."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_storage
from whaleshark_reid.web.services import annotations as ann_service

router = APIRouter()


@router.get("/list/annotations", response_class=HTMLResponse)
def annotations_list(
    request: Request,
    page: int = 0,
    storage: Storage = Depends(get_storage),
):
    result = ann_service.list_annotations(storage, page=page)
    return templates.TemplateResponse(
        "list/annotations.html",
        {"request": request, "result": result},
    )


@router.get("/annotation/{annotation_uuid}", response_class=HTMLResponse)
def annotation_detail(
    request: Request,
    annotation_uuid: str,
    storage: Storage = Depends(get_storage),
):
    ann = ann_service.get_annotation_detail(storage, annotation_uuid)
    if ann is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return templates.TemplateResponse(
        "list/annotation_detail.html",
        {"request": request, "ann": ann},
    )
```

`src/whaleshark_reid/web/templates/list/annotations.html`:
```html
{% extends "base.html" %}
{% block title %}Annotations{% endblock %}
{% block content %}
<h1>Annotations ({{ result.total }})</h1>
<table>
    <thead>
        <tr>
            <th>UUID</th><th>Source</th><th>Obs ID</th><th>Date</th>
            <th>GPS</th><th>Name UUID</th><th>Viewpoint</th>
        </tr>
    </thead>
    <tbody>
    {% for ann in result.items %}
        <tr>
            <td><a href="/annotation/{{ ann.annotation_uuid }}">{{ ann.annotation_uuid[:12] }}…</a></td>
            <td>{{ ann.source }}</td>
            <td>{{ ann.observation_id }}</td>
            <td>{{ ann.date_captured or '—' }}</td>
            <td>{{ ann.gps_lat_captured or '—' }}, {{ ann.gps_lon_captured or '—' }}</td>
            <td>{{ ann.name_uuid[:8] if ann.name_uuid else '—' }}</td>
            <td>{{ ann.viewpoint }}</td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% if result.page > 0 %}
<a href="/list/annotations?page={{ result.page - 1 }}">← Previous</a>
{% endif %}
{% if (result.page + 1) * result.page_size < result.total %}
<a href="/list/annotations?page={{ result.page + 1 }}">Next →</a>
{% endif %}
{% endblock %}
```

`src/whaleshark_reid/web/templates/list/annotation_detail.html`:
```html
{% extends "base.html" %}
{% block title %}Annotation {{ ann.annotation_uuid[:12] }}{% endblock %}
{% block content %}
<h1>Annotation Detail</h1>
<div class="card">
    <img class="ann-img" src="/image/{{ ann.annotation_uuid }}?crop=true" style="max-width:400px;">
    <dl class="meta-panel" style="margin-top:1rem;">
        <dt>annotation_uuid</dt><dd>{{ ann.annotation_uuid }}</dd><br>
        <dt>image_uuid</dt><dd>{{ ann.image_uuid }}</dd><br>
        <dt>name_uuid</dt><dd>{{ ann.name_uuid or '—' }}</dd><br>
        <dt>name</dt><dd>{{ ann.name or '—' }}</dd><br>
        <dt>source</dt><dd>{{ ann.source }}</dd><br>
        <dt>observation_id</dt><dd>{{ ann.observation_id }}</dd><br>
        <dt>photo_index</dt><dd>{{ ann.photo_index }}</dd><br>
        <dt>file_path</dt><dd>{{ ann.file_path }}</dd><br>
        <dt>bbox</dt><dd>{{ ann.bbox }}</dd><br>
        <dt>theta</dt><dd>{{ ann.theta }}</dd><br>
        <dt>viewpoint</dt><dd>{{ ann.viewpoint }}</dd><br>
        <dt>species</dt><dd>{{ ann.species }}</dd><br>
        <dt>date_captured</dt><dd>{{ ann.date_captured or '—' }}</dd><br>
        <dt>gps</dt><dd>{{ ann.gps_lat_captured or '—' }}, {{ ann.gps_lon_captured or '—' }}</dd><br>
        <dt>photographer</dt><dd>{{ ann.photographer or '—' }}</dd><br>
        <dt>license</dt><dd>{{ ann.license or '—' }}</dd><br>
        <dt>quality_grade</dt><dd>{{ ann.quality_grade or '—' }}</dd><br>
        <dt>conf</dt><dd>{{ ann.conf or '—' }}</dd><br>
    </dl>
</div>
{% endblock %}
```

- [ ] **Step 3: Register router in app.py**

```python
    from whaleshark_reid.web.routes import list as list_router
    app.include_router(list_router.router)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/web/test_list_views.py -x
```
Expected: 3 passed. Full suite: ~114.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/web/ tests/web/test_list_views.py
git commit -m "feat: annotations list + detail pages"
```

---

## Task 5: Experiments view — list + detail + diff

**Files:**
- Create: `src/whaleshark_reid/web/services/experiments.py`
- Create: `src/whaleshark_reid/web/routes/experiments.py`
- Create: `src/whaleshark_reid/web/templates/experiments/index.html`
- Create: `src/whaleshark_reid/web/templates/experiments/detail.html`
- Create: `src/whaleshark_reid/web/templates/experiments/diff.html`
- Modify: `src/whaleshark_reid/web/app.py` (include router)
- Create: `tests/web/test_experiments_view.py`

- [ ] **Step 1: Write failing test**

`tests/web/test_experiments_view.py`:
```python
"""Tests for the experiments view — list, detail, diff."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_experiments_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/experiments")
    assert r.status_code == 200
    assert "r_ingest" in r.text or "ingest" in r.text


def test_run_detail_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/run/r_embed")
    assert r.status_code == 200
    assert "embed" in r.text


def test_run_diff_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/run/r_embed/diff/r_cluster")
    assert r.status_code == 200
```

- [ ] **Step 2: Implement service + routes + templates**

`src/whaleshark_reid/web/services/experiments.py`:
```python
"""Experiments service — list runs, run detail, diff two runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.storage.db import Storage


class RunSummary(BaseModel):
    run_id: str
    stage: str
    status: str
    config_preview: str
    metrics_preview: str
    started_at: str
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None


class RunDetail(BaseModel):
    run: RunSummary
    config: dict
    metrics: dict
    reproduce_cmd: str
    log_tail: list[str]


class DiffLine(BaseModel):
    key: str
    val_a: str
    val_b: str
    changed: bool


class RunDiff(BaseModel):
    run_a: RunDetail
    run_b: RunDetail
    config_diff: list[DiffLine]
    metrics_diff: list[DiffLine]


def _row_to_summary(row) -> RunSummary:
    config_str = ""
    if row["config_json"]:
        try:
            c = json.loads(row["config_json"])
            config_str = ", ".join(f"{k}={v}" for k, v in list(c.items())[:3])
        except json.JSONDecodeError:
            pass

    metrics_str = ""
    if row["metrics_json"]:
        try:
            m = json.loads(row["metrics_json"])
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(m.items())[:4])
        except json.JSONDecodeError:
            pass

    duration = None
    if row["started_at"] and row["finished_at"]:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(row["started_at"])
            t1 = datetime.fromisoformat(row["finished_at"])
            duration = round((t1 - t0).total_seconds(), 1)
        except (ValueError, TypeError):
            pass

    return RunSummary(
        run_id=row["run_id"],
        stage=row["stage"],
        status=row["status"],
        config_preview=config_str,
        metrics_preview=metrics_str,
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        duration_s=duration,
    )


def list_runs(storage: Storage) -> list[RunSummary]:
    rows = storage.conn.execute(
        "SELECT * FROM runs WHERE stage != 'adhoc_queue' ORDER BY started_at DESC"
    ).fetchall()
    return [_row_to_summary(r) for r in rows]


def _build_reproduce_cmd(config: dict, stage: str) -> str:
    parts = [f"catalog {stage}"]
    for k, v in config.items():
        if v is not None and v != "" and k not in ("source",):
            flag = f"--{k.replace('_', '-')}"
            parts.append(f"{flag} {v}")
    return " ".join(parts)


def get_run_detail(storage: Storage, cache_dir: Path, run_id: str) -> Optional[RunDetail]:
    row = storage.conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None

    config = json.loads(row["config_json"]) if row["config_json"] else {}
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}

    log_path = cache_dir / "logs" / f"{run_id}.log"
    log_tail: list[str] = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = lines[-20:]

    return RunDetail(
        run=_row_to_summary(row),
        config=config,
        metrics=metrics,
        reproduce_cmd=_build_reproduce_cmd(config, row["stage"]),
        log_tail=log_tail,
    )


def diff_runs(
    storage: Storage, cache_dir: Path, run_a: str, run_b: str
) -> Optional[RunDiff]:
    detail_a = get_run_detail(storage, cache_dir, run_a)
    detail_b = get_run_detail(storage, cache_dir, run_b)
    if detail_a is None or detail_b is None:
        return None

    all_config_keys = sorted(set(detail_a.config) | set(detail_b.config))
    config_diff = [
        DiffLine(
            key=k,
            val_a=str(detail_a.config.get(k, "—")),
            val_b=str(detail_b.config.get(k, "—")),
            changed=detail_a.config.get(k) != detail_b.config.get(k),
        )
        for k in all_config_keys
    ]

    all_metrics_keys = sorted(set(detail_a.metrics) | set(detail_b.metrics))
    metrics_diff = [
        DiffLine(
            key=k,
            val_a=str(detail_a.metrics.get(k, "—")),
            val_b=str(detail_b.metrics.get(k, "—")),
            changed=detail_a.metrics.get(k) != detail_b.metrics.get(k),
        )
        for k in all_metrics_keys
    ]

    return RunDiff(
        run_a=detail_a, run_b=detail_b,
        config_diff=config_diff, metrics_diff=metrics_diff,
    )
```

`src/whaleshark_reid/web/routes/experiments.py`:
```python
"""Experiments view routes: runs list, detail, diff."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import experiments as exp_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


@router.get("/experiments", response_class=HTMLResponse)
def experiments_list(
    request: Request,
    storage: Storage = Depends(get_storage),
):
    runs = exp_service.list_runs(storage)
    return templates.TemplateResponse(
        "experiments/index.html",
        {"request": request, "runs": runs},
    )


@router.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(
    request: Request,
    run_id: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    detail = exp_service.get_run_detail(storage, settings.cache_dir, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(
        "experiments/detail.html",
        {"request": request, "detail": detail},
    )


@router.get("/run/{run_a}/diff/{run_b}", response_class=HTMLResponse)
def run_diff(
    request: Request,
    run_a: str,
    run_b: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    diff = exp_service.diff_runs(storage, settings.cache_dir, run_a, run_b)
    if diff is None:
        raise HTTPException(status_code=404, detail="One or both runs not found")
    return templates.TemplateResponse(
        "experiments/diff.html",
        {"request": request, "diff": diff},
    )
```

`src/whaleshark_reid/web/templates/experiments/index.html`:
```html
{% extends "base.html" %}
{% block title %}Experiments{% endblock %}
{% block content %}
<h1>Experiment Runs</h1>
<table>
    <thead>
        <tr><th>Run ID</th><th>Stage</th><th>Status</th><th>Duration</th><th>Config</th><th>Metrics</th><th>Diff</th></tr>
    </thead>
    <tbody>
    {% for run in runs %}
        <tr>
            <td><a href="/run/{{ run.run_id }}">{{ run.run_id }}</a></td>
            <td>{{ run.stage }}</td>
            <td style="color:{% if run.status == 'ok' %}var(--success){% elif run.status == 'failed' %}var(--danger){% else %}var(--warning){% endif %}">{{ run.status }}</td>
            <td>{{ run.duration_s or '—' }}s</td>
            <td style="font-size:11px;">{{ run.config_preview }}</td>
            <td style="font-size:11px;">{{ run.metrics_preview }}</td>
            <td>
                {% if not loop.last %}
                <a href="/run/{{ run.run_id }}/diff/{{ runs[loop.index].run_id }}">diff ↓</a>
                {% endif %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% endblock %}
```

`src/whaleshark_reid/web/templates/experiments/detail.html`:
```html
{% extends "base.html" %}
{% block title %}Run {{ detail.run.run_id }}{% endblock %}
{% block content %}
<h1>Run: {{ detail.run.run_id }}</h1>
<div class="card">
    <dl class="meta-panel">
        <dt>stage</dt><dd>{{ detail.run.stage }}</dd><br>
        <dt>status</dt><dd>{{ detail.run.status }}</dd><br>
        <dt>duration</dt><dd>{{ detail.run.duration_s or '—' }}s</dd><br>
        <dt>started</dt><dd>{{ detail.run.started_at }}</dd><br>
    </dl>
</div>
<h2>Reproduce</h2>
<div class="card"><code>{{ detail.reproduce_cmd }}</code></div>
<h2>Config</h2>
<div class="card"><pre>{{ detail.config | tojson(indent=2) }}</pre></div>
<h2>Metrics</h2>
<div class="card"><pre>{{ detail.metrics | tojson(indent=2) }}</pre></div>
{% if detail.log_tail %}
<h2>Log (last 20 lines)</h2>
<div class="card"><pre>{% for line in detail.log_tail %}{{ line }}
{% endfor %}</pre></div>
{% endif %}
{% endblock %}
```

`src/whaleshark_reid/web/templates/experiments/diff.html`:
```html
{% extends "base.html" %}
{% block title %}Diff: {{ diff.run_a.run.run_id }} vs {{ diff.run_b.run.run_id }}{% endblock %}
{% block content %}
<h1>Diff: {{ diff.run_a.run.run_id }} vs {{ diff.run_b.run.run_id }}</h1>
<h2>Config</h2>
<table>
    <thead><tr><th>Key</th><th>Run A</th><th>Run B</th></tr></thead>
    <tbody>
    {% for d in diff.config_diff %}
    <tr style="{% if d.changed %}background:rgba(255,255,0,0.05);{% endif %}">
        <td>{{ d.key }}</td><td>{{ d.val_a }}</td><td>{{ d.val_b }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
<h2>Metrics</h2>
<table>
    <thead><tr><th>Key</th><th>Run A</th><th>Run B</th></tr></thead>
    <tbody>
    {% for d in diff.metrics_diff %}
    <tr style="{% if d.changed %}background:rgba(255,255,0,0.05);{% endif %}">
        <td>{{ d.key }}</td><td>{{ d.val_a }}</td><td>{{ d.val_b }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% endblock %}
```

- [ ] **Step 3: Register router**

```python
    from whaleshark_reid.web.routes import experiments as exp_router
    app.include_router(exp_router.router)
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/web/test_experiments_view.py -x
pytest -x
git add src/whaleshark_reid/web/ tests/web/test_experiments_view.py
git commit -m "feat: experiments view — runs list, detail, diff"
```

---

## Task 6: Cluster scatter view

**Files:**
- Create: `src/whaleshark_reid/web/services/cluster_view.py`
- Create: `src/whaleshark_reid/web/routes/clusters.py`
- Create: `src/whaleshark_reid/web/templates/clusters/scatter.html`
- Create: `src/whaleshark_reid/web/static/js/plotly_scatter.js`
- Vendor: `src/whaleshark_reid/web/static/vendor/plotly.min.js`
- Modify: `src/whaleshark_reid/web/app.py`
- Create: `tests/web/test_cluster_view.py`

- [ ] **Step 1: Vendor Plotly**

```bash
curl -sL https://cdn.plot.ly/plotly-2.35.2.min.js -o src/whaleshark_reid/web/static/vendor/plotly.min.js
```

- [ ] **Step 2: Write failing test**

`tests/web/test_cluster_view.py`:
```python
"""Tests for the cluster scatter view."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_cluster_scatter_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/clusters/r_cluster")
    assert r.status_code == 200
    assert "plotly" in r.text.lower() or "scatter" in r.text.lower()


def test_projection_json_endpoint(seeded_web_client: TestClient):
    r = seeded_web_client.get("/api/projections/r_project")
    assert r.status_code == 200
    data = r.json()
    assert "points" in data
    assert len(data["points"]) == 10
    for p in data["points"]:
        assert "x" in p and "y" in p and "annotation_uuid" in p
```

- [ ] **Step 3: Implement service + route + template + JS**

`src/whaleshark_reid/web/services/cluster_view.py`:
```python
"""Cluster view service — projection data and cluster summary."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.storage.cluster_cache import read_clusters
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.projection_cache import read_projections


class ProjectionPoint(BaseModel):
    annotation_uuid: str
    x: float
    y: float
    cluster_label: int


class ProjectionResponse(BaseModel):
    run_id: str
    points: list[ProjectionPoint]


def get_projection(
    cache_dir: Path, projection_run_id: str, cluster_run_id: str
) -> Optional[ProjectionResponse]:
    try:
        proj_df = read_projections(cache_dir, projection_run_id)
        cluster_df = read_clusters(cache_dir, cluster_run_id)
    except FileNotFoundError:
        return None

    cluster_map = dict(zip(cluster_df["annotation_uuid"], cluster_df["cluster_label"].astype(int)))

    points = [
        ProjectionPoint(
            annotation_uuid=row["annotation_uuid"],
            x=float(row["x"]),
            y=float(row["y"]),
            cluster_label=cluster_map.get(row["annotation_uuid"], -1),
        )
        for _, row in proj_df.iterrows()
    ]

    return ProjectionResponse(run_id=projection_run_id, points=points)
```

`src/whaleshark_reid/web/routes/clusters.py`:
```python
"""Cluster scatter view routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import cluster_view as cv_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


@router.get("/clusters/{run_id}", response_class=HTMLResponse)
def cluster_scatter(
    request: Request,
    run_id: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    proj_run = storage.get_latest_run_id("project")
    return templates.TemplateResponse(
        "clusters/scatter.html",
        {"request": request, "cluster_run_id": run_id, "proj_run_id": proj_run or run_id},
    )


@router.get("/api/projections/{run_id}")
def projection_json(
    run_id: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    cluster_run = storage.get_latest_run_id("cluster")
    proj = cv_service.get_projection(
        settings.cache_dir, run_id, cluster_run or run_id
    )
    if proj is None:
        return JSONResponse({"points": []})
    return proj.model_dump()
```

`src/whaleshark_reid/web/templates/clusters/scatter.html`:
```html
{% extends "base.html" %}
{% block title %}Clusters{% endblock %}
{% block head_extra %}
<script src="/static/vendor/plotly.min.js"></script>
<script src="/static/js/plotly_scatter.js" defer></script>
{% endblock %}
{% block content %}
<h1>Embedding Clusters</h1>
<div class="scatter-container" id="scatter"
     data-projection-url="/api/projections/{{ proj_run_id }}"></div>
{% endblock %}
```

`src/whaleshark_reid/web/static/js/plotly_scatter.js`:
```js
// Fetch projection data and render a Plotly scatter colored by cluster label.
document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('scatter');
  if (!container) return;
  const url = container.dataset.projectionUrl;
  fetch(url)
    .then(r => r.json())
    .then(data => {
      if (!data.points || data.points.length === 0) {
        container.innerHTML = '<p style="color:var(--text-muted)">No projection data. Run <code>catalog project</code>.</p>';
        return;
      }
      const x = data.points.map(p => p.x);
      const y = data.points.map(p => p.y);
      const colors = data.points.map(p => p.cluster_label);
      const text = data.points.map(p => `${p.annotation_uuid.slice(0,12)}… (cluster ${p.cluster_label})`);
      Plotly.newPlot(container, [{
        x, y, mode: 'markers', type: 'scatter',
        marker: { color: colors, colorscale: 'Portland', size: 8 },
        text, hoverinfo: 'text',
      }], {
        paper_bgcolor: '#1a1a2e', plot_bgcolor: '#16213e',
        font: { color: '#e0e0e0', family: 'monospace' },
        margin: { t: 20, b: 40, l: 40, r: 20 },
        xaxis: { gridcolor: '#333' }, yaxis: { gridcolor: '#333' },
      }, { responsive: true });
    });
});
```

- [ ] **Step 4: Register router + run tests + commit**

```python
    from whaleshark_reid.web.routes import clusters
    app.include_router(clusters.router)
```

```bash
pytest tests/web/test_cluster_view.py -x
pytest -x
git add src/whaleshark_reid/web/ tests/web/test_cluster_view.py
git commit -m "feat: cluster scatter view with Plotly.js + projection JSON endpoint"
```

---

## Task 7: Home redirect + map stub + final nav + decisions/individuals list stubs

This task wires up the remaining pages as functional stubs so every nav link works. It adds the home redirect, the map stub, and minimal decisions/individuals list pages.

**Files:**
- Create: `src/whaleshark_reid/web/routes/home.py`
- Create: `src/whaleshark_reid/web/templates/map/stub.html`
- Create: `src/whaleshark_reid/web/templates/list/decisions.html`
- Create: `src/whaleshark_reid/web/templates/list/individuals.html`
- Modify: `src/whaleshark_reid/web/routes/list.py` (add decisions + individuals routes)
- Modify: `src/whaleshark_reid/web/app.py` (include home + map routers)
- Create: `tests/web/test_nav.py`

- [ ] **Step 1: Write failing test**

`tests/web/test_nav.py`:
```python
"""Tests that all nav links work (no 500s, no dead links)."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_home_redirects(seeded_web_client: TestClient):
    r = seeded_web_client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)


def test_map_stub_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/map")
    assert r.status_code == 200
    assert "Phase" in r.text or "coming" in r.text.lower()


def test_decisions_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/list/decisions")
    assert r.status_code == 200


def test_individuals_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/list/individuals")
    assert r.status_code == 200
```

- [ ] **Step 2: Implement home route + map stub + list stubs**

`src/whaleshark_reid/web/routes/home.py`:
```python
"""Home route — redirects to the latest pair review queue."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_storage

router = APIRouter()


@router.get("/")
def home(storage: Storage = Depends(get_storage)):
    latest = storage.get_latest_run_id("matching")
    if latest:
        return RedirectResponse(url=f"/review/pairs/{latest}")
    return RedirectResponse(url="/experiments")
```

Add to `src/whaleshark_reid/web/routes/list.py`:
```python
@router.get("/list/decisions", response_class=HTMLResponse)
def decisions_list(request: Request, storage: Storage = Depends(get_storage)):
    rows = storage.conn.execute(
        "SELECT * FROM pair_decisions ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    return templates.TemplateResponse(
        "list/decisions.html",
        {"request": request, "decisions": [dict(r) for r in rows]},
    )


@router.get("/list/individuals", response_class=HTMLResponse)
def individuals_list(request: Request, storage: Storage = Depends(get_storage)):
    rows = storage.conn.execute(
        """
        SELECT name_uuid, name, COUNT(*) as member_count
        FROM annotations
        WHERE name_uuid IS NOT NULL
        GROUP BY name_uuid
        ORDER BY member_count DESC
        """
    ).fetchall()
    return templates.TemplateResponse(
        "list/individuals.html",
        {"request": request, "individuals": [dict(r) for r in rows]},
    )
```

`src/whaleshark_reid/web/templates/list/decisions.html`:
```html
{% extends "base.html" %}
{% block title %}Decisions{% endblock %}
{% block content %}
<h1>Pair Decisions</h1>
<table>
    <thead><tr><th>ID</th><th>Ann A</th><th>Ann B</th><th>Decision</th><th>Distance</th><th>User</th><th>Date</th></tr></thead>
    <tbody>
    {% for d in decisions %}
    <tr>
        <td>{{ d.decision_id }}</td>
        <td><a href="/annotation/{{ d.ann_a_uuid }}">{{ d.ann_a_uuid[:12] }}…</a></td>
        <td><a href="/annotation/{{ d.ann_b_uuid }}">{{ d.ann_b_uuid[:12] }}…</a></td>
        <td>{{ d.decision }}</td>
        <td>{{ '%.3f'|format(d.distance) if d.distance else '—' }}</td>
        <td>{{ d.user }}</td>
        <td>{{ d.created_at }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% if not decisions %}<p style="color:var(--text-muted);">No decisions yet. Start reviewing pairs!</p>{% endif %}
{% endblock %}
```

`src/whaleshark_reid/web/templates/list/individuals.html`:
```html
{% extends "base.html" %}
{% block title %}Individuals{% endblock %}
{% block content %}
<h1>Derived Individuals</h1>
<table>
    <thead><tr><th>Name UUID</th><th>Name</th><th>Members</th></tr></thead>
    <tbody>
    {% for ind in individuals %}
    <tr>
        <td>{{ ind.name_uuid[:12] }}…</td>
        <td>{{ ind.name or '—' }}</td>
        <td>{{ ind.member_count }}</td>
    </tr>
    {% endfor %}
    </tbody>
</table>
{% if not individuals %}<p style="color:var(--text-muted);">No individuals derived yet. Review pairs and run <code>catalog rebuild-individuals</code>.</p>{% endif %}
{% endblock %}
```

`src/whaleshark_reid/web/templates/map/stub.html`:
```html
{% extends "base.html" %}
{% block title %}Map{% endblock %}
{% block content %}
<div class="card" style="text-align:center; padding:3rem;">
    <h2 style="color:var(--text-muted);">Map View</h2>
    <p style="color:var(--text-muted);">Coming in Phase 3 (Wildbook integration). The pair carousel already shows per-pair GPS insets.</p>
</div>
{% endblock %}
```

- [ ] **Step 3: Create map route and register all remaining routers**

Create `src/whaleshark_reid/web/routes/map.py`:
```python
"""Map stub route."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.web.app import templates

router = APIRouter()


@router.get("/map", response_class=HTMLResponse)
def map_stub(request: Request):
    return templates.TemplateResponse("map/stub.html", {"request": request})
```

Register in `app.py`:
```python
    from whaleshark_reid.web.routes import home, map as map_router
    app.include_router(home.router)
    app.include_router(map_router.router)
```

- [ ] **Step 4: Run tests + commit**

```bash
pytest tests/web/test_nav.py -x
pytest -x
git add src/whaleshark_reid/web/ tests/web/test_nav.py
git commit -m "feat: home redirect, map stub, decisions + individuals list pages"
```

---

## Task 8: Vendor Leaflet + Plotly verification + final polish

**Files:**
- Vendor: `src/whaleshark_reid/web/static/vendor/leaflet.js` + `leaflet.css`
- Create: `src/whaleshark_reid/web/static/js/leaflet_inset.js`
- Verify all pages load without 500s

- [ ] **Step 1: Vendor Leaflet**

```bash
curl -sL https://unpkg.com/leaflet@1.9.4/dist/leaflet.js -o src/whaleshark_reid/web/static/vendor/leaflet.js
curl -sL https://unpkg.com/leaflet@1.9.4/dist/leaflet.css -o src/whaleshark_reid/web/static/vendor/leaflet.css
```

- [ ] **Step 2: Create leaflet inset JS**

`src/whaleshark_reid/web/static/js/leaflet_inset.js`:
```js
// Initialize a Leaflet map inset on the pair carousel.
// Reads lat/lon from data attrs on #map-inset element.
function initMapInset() {
  const el = document.getElementById('map-inset');
  if (!el) return;
  const latA = parseFloat(el.dataset.latA);
  const lonA = parseFloat(el.dataset.lonA);
  const latB = parseFloat(el.dataset.latB);
  const lonB = parseFloat(el.dataset.lonB);

  // Clear previous content
  el.innerHTML = '';

  if (isNaN(latA) || isNaN(lonA) || isNaN(latB) || isNaN(lonB)) {
    el.innerHTML = '<span style="color:var(--text-muted);font-size:11px;">No GPS data</span>';
    return;
  }

  const map = L.map(el, { zoomControl: false, attributionControl: false });
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 18 }).addTo(map);

  const mA = L.marker([latA, lonA]).addTo(map);
  const mB = L.marker([latB, lonB]).addTo(map);
  L.polyline([[latA, lonA], [latB, lonB]], { color: '#00adb5', weight: 2 }).addTo(map);

  map.fitBounds([[latA, lonA], [latB, lonB]], { padding: [20, 20] });
}

document.addEventListener('DOMContentLoaded', initMapInset);
```

- [ ] **Step 3: Add Leaflet to carousel template head**

Edit `src/whaleshark_reid/web/templates/pairs/carousel.html` to include Leaflet CSS and JS in the `head_extra` block:

```html
{% block head_extra %}
<link rel="stylesheet" href="/static/vendor/leaflet.css">
<script src="/static/vendor/leaflet.js"></script>
<script src="/static/js/leaflet_inset.js" defer></script>
<script src="/static/js/carousel.js" defer></script>
{% endblock %}
```

- [ ] **Step 4: Run full test suite**

```bash
pytest -x
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/web/static/
git commit -m "feat: vendor Leaflet + map inset JS for pair carousel"
```

---

## Task 9: Comprehensive web tests

Final task — add the remaining web test assertions to ensure HTMX fragment shape, keyboard shortcut data attributes, and cross-page navigation all work correctly.

**Files:**
- Create: `tests/web/test_htmx_fragments.py`

- [ ] **Step 1: Write the comprehensive test file**

`tests/web/test_htmx_fragments.py`:
```python
"""Cross-cutting tests: HTMX fragments, keyboard shortcuts, full page structure."""
from __future__ import annotations

import re

from starlette.testclient import TestClient


def test_pair_card_has_keyboard_shortcut_attrs(seeded_web_client: TestClient):
    r = seeded_web_client.get("/review/pairs/r_match")
    assert r.status_code == 200
    # All four decision buttons should have data-shortcut
    for key in ["y", "n", "u", " "]:
        assert f'data-shortcut="{key}"' in r.text, f"Missing data-shortcut for key '{key}'"


def test_htmx_decide_returns_fragment_not_full_page(seeded_web_client: TestClient):
    # Get the first pair's queue_id
    page = seeded_web_client.get("/review/pairs/r_match")
    match = re.search(r'data-queue-id="(\d+)"', page.text)
    assert match
    queue_id = match.group(1)

    # HTMX POST should return a fragment (no <html> tag)
    r = seeded_web_client.post(
        f"/api/pairs/{queue_id}/decide",
        data={"decision": "skip"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert "<html" not in r.text


def test_all_nav_links_return_200(seeded_web_client: TestClient):
    """Verify every link in the nav bar returns a non-500 response."""
    pages = [
        "/review/pairs/r_match",
        "/list/annotations",
        "/list/decisions",
        "/list/individuals",
        "/clusters/r_cluster",
        "/experiments",
        "/map",
        "/health",
    ]
    for url in pages:
        r = seeded_web_client.get(url)
        assert r.status_code < 500, f"{url} returned {r.status_code}"


def test_image_crop_in_carousel_returns_jpeg(seeded_web_client: TestClient):
    from whaleshark_reid.core.schema import inat_annotation_uuid
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/image/{uuid}?crop=true")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
```

- [ ] **Step 2: Run tests + commit**

```bash
pytest tests/web/ -x -v
pytest -x
git add tests/web/test_htmx_fragments.py
git commit -m "test: comprehensive web tests — HTMX fragments, keyboard attrs, nav, image crop"
```

---

## Success criteria (from spec 03)

After all 9 tasks complete:

1. ✅ All five pages render on fixture data with no 500s (home → carousel, list views, clusters, experiments, map stub).
2. ✅ Pair carousel decision round-trip works: click → decision persisted → next pair rendered via HTMX.
3. ✅ Keyboard shortcuts Y/N/U/Space data attributes are present on decision buttons.
4. ✅ Map inset div has GPS data attributes (Leaflet init requires browser JS runtime — tested structurally, not visually).
5. ✅ Cluster view loads UMAP projection and renders Plotly scatter (JSON endpoint verified, JS rendering requires browser).
6. ✅ Experiments view diffs two runs correctly.
7. ✅ `/image/<uuid>?crop=true` returns a cropped JPEG.
8. ✅ Web test suite (Layer 4) passes.
9. ✅ `pytest` entire suite passes.
10. ✅ Swapping to React later requires zero changes to `services/` — routes and templates are the only web-specific code.

## Manual smoke test after Task 9

```bash
cd /workspace/catalog-match/whaleshark-reid
catalog run-all \
  --csv /workspace/catalog-match/whaleshark/whaleshark_inat_v1.csv \
  --photos-dir /workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/photos \
  --rich-csv /workspace/catalog-match/whaleshark/dfx_whaleshark_inat_v1.csv \
  --db-path cache/state.db --cache-dir cache/

uvicorn whaleshark_reid.web.app:create_app --factory --host 0.0.0.0 --port 8090
```

Then open `http://rodan.dyn.wildme.io:<mapped-port>` in your browser. You should see the pair carousel with real whale shark images.

## What this plan explicitly does NOT cover

- Leaflet map inset rendering test (requires browser JS — tested structurally via data-attrs only)
- Plotly scatter rendering test (same — tested via JSON endpoint only)
- Pipeline runner (subprocess `catalog run-all` from the UI) — deferred to a follow-up task
- Individual rename via the UI — deferred
- Authentication / multi-user
- React SPA
