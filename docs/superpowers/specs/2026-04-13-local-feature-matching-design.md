# Local Feature Matching for Pair Review — Design

**Date:** 2026-04-13
**Status:** Approved for implementation planning

## Goal

Give the human reviewer a second, independent signal when verifying MiewID-suggested pairs: an overlay of matched local features (keypoints + connecting lines) drawn directly on the two chip images. Reviewer can trace corresponding points between the two sharks to gain or lose confidence in a match.

This complements MiewID distance. It does not replace it, and it does not yet feed back into candidate generation or reranking.

## V1 Scope

- One pair, one interactive visualization, triggered on demand per pair.
- Extractor + matcher: **ALIKED + LightGlue** (cvg/LightGlue, installed directly from GitHub, not via wbia-plugin-lightglue).
- Visualization: client-side **SVG overlay** on the live chip images, with confidence slider, line/keypoint toggles, click-to-hide, shift-click-to-hide-below.
- **Hybrid caching:** manual "Run local match" button on first view; cached results auto-render on subsequent views.
- **Bulk precompute CLI** leveraging LightGlue's batched inference and per-annotation feature dedup, so a whole review queue can be warmed on the GPU ahead of time.

Explicitly out of scope: score fusion with MiewID, reranking MiewID candidates with LightGlue, PNG export, SuperPoint / DISK extractor wiring (structure supports them but only ALIKED ships initially).

## Architecture

### 1. Core module — `whaleshark_reid.core.match.lightglue`

Mirrors the shape of `core.embed.miewid` (a reviewer familiar with one will recognize the other).

**Public API:**

- `MatchResult` dataclass: `n_matches: int`, `mean_score: float`, `median_score: float`, `kpts_a: list[[x, y]]`, `kpts_b: list[[x, y]]`, `matches: list[[i, j, score]]`, `img_a_size: [w, h]`, `img_b_size: [w, h]`, `extractor: str`.
- `class LocalMatcher`: holds the extractor + matcher on a device. Lazy singleton per extractor via `get_matcher(extractor="aliked")`; model is loaded once per process and held in VRAM. CUDA if available, else CPU.
- `match_pair(img_a_path, img_b_path, extractor="aliked") -> MatchResult`: load images, resize to 440×440, extract features, match, return result. Used by the interactive route.
- `extract_features_batch(image_paths, batch_size=8) -> dict[path → feats]`: batch feature extraction for bulk precompute.
- `match_pairs_batch(pairs, feats_cache) -> list[MatchResult]`: batched matcher pass over precomputed feats.

**Design note — per-annotation dedup.** In a pair queue, each annotation appears in many pairs. Extracting features per pair would waste GPU time. The batch API extracts once per unique annotation UUID, caches feats in a dict keyed by annotation UUID, then the matcher pass joins feats pairwise. Expected speedup is proportional to the average pair-queue fan-out per annotation.

**Device + install fallback.** LightGlue is an optional extra; if the import fails, the matcher endpoints return 503 with a clear error. Everything else (MiewID, pair queue, review UI) keeps working.

### 2. Storage

New table, added via additive migration in `_apply_migrations`:

```sql
CREATE TABLE IF NOT EXISTS pair_matches (
    queue_id       INTEGER NOT NULL,
    extractor      TEXT    NOT NULL,
    n_matches      INTEGER NOT NULL,
    mean_score     REAL,
    median_score   REAL,
    match_data     TEXT    NOT NULL,   -- JSON: {kpts_a, kpts_b, matches}
    img_a_size     TEXT    NOT NULL,   -- '[w, h]' — coord space of kpts_a (match space, e.g. 440x440)
    img_b_size     TEXT    NOT NULL,   -- '[w, h]' — coord space of kpts_b
    computed_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (queue_id, extractor),
    FOREIGN KEY (queue_id) REFERENCES pair_queue(queue_id)
);
```

**Why a new table:**
- Multiple extractors can coexist, keyed by `(queue_id, extractor)`.
- Keeps a JSON blob out of the hot `pair_queue` table that every review filter query hits.
- Clear ownership: anything under this PK is a computed cache and is safe to delete/rebuild.

**JSON shape in `match_data`:**
```json
{
  "kpts_a":  [[x, y], ...],
  "kpts_b":  [[x, y], ...],
  "matches": [[i, j, score], ...]
}
```
Always read/written as a unit — no need for SQLite JSON indexing.

**`n_matches`** is the count of matches with `score >= 0.5` (the same confidence threshold used in the default UI slider), so the displayed headline stat matches what the reviewer initially sees without requiring any client-side recomputation when lists are rendered.

### 3. Web API

Two endpoints (new router `web/routes/local_match.py` mounted in `web/app.py`), backed by a thin service layer in `web/services/local_match.py`.

- `GET /api/pairs/{queue_id}/local-match?extractor=aliked`
  - 200 + JSON if cached in `pair_matches`.
  - 404 if not cached.
  - Client calls this on every pair render; a 200 triggers auto-overlay (the "hybrid" behavior).
- `POST /api/pairs/{queue_id}/local-match?extractor=aliked[&overwrite=1]`
  - Runs the matcher, writes to `pair_matches`, returns JSON.
  - Synchronous. Expected ~50–200 ms on GPU.
  - `overwrite=1` is how the client "Re-run" button forces a recompute.

**Response shape** (both GET and POST on cache hit / compute):
```json
{
  "extractor": "aliked",
  "n_matches": 142,
  "mean_score": 0.71,
  "median_score": 0.68,
  "img_a_size": [440, 440],
  "img_b_size": [440, 440],
  "kpts_a":  [[x, y], ...],
  "kpts_b":  [[x, y], ...],
  "matches": [[i, j, score], ...]
}
```

**Error modes:** 404 (queue_id unknown), 503 (LightGlue not installed), 500 (model crash — logged, returned as JSON error).

### 4. Frontend

**Template changes (`web/templates/partials/pair_card.html`):**

- Each chip image wrapped in `<div class="chip-wrap">` with `position: relative`, so an SVG overlay can be absolutely positioned over it.
- A **single** `<svg>` per pair spans both chips — one viewBox, two coordinate regions — so lines can cross the gap between images naturally.
- A `<div class="local-match-controls">` holds: confidence slider (0.0–1.0, default 0.5), lines/keypoints toggle checkboxes, "Show hidden (N)" button, "Re-run" button, stats readout (`"142 matches · mean 0.71"`).
- A hidden data island carries the queue_id and initial state; JS reads it on load.

**Client module (`web/static/js/local_match.js`):**

- On pair render, issue `GET /api/pairs/{queue_id}/local-match`.
  - 200 → render overlay directly.
  - 404 → show idle "Run local match" button.
- POST fires on button click; UI transitions idle → loading (spinner) → rendered.
- Coordinate transform: LightGlue returns keypoints in 440×440 match space; rescale to the rendered chip's current pixel size. A `ResizeObserver` reapplies the transform if the viewport changes.
- Color scheme: lines colored by match confidence using a viridis gradient (same as the notebook, so manual spot-checks are visually consistent).
- Interactions:
  - Slider → filter lines by confidence, pure client-side.
  - Click a line → add its match index to a `hidden` Set; fade it out. Also hide its two endpoint keypoints if they aren't shared with another visible match.
  - Shift-click a line → hide it *and* all lines with lower confidence (fast tail cleanup).
  - "Show hidden (N)" → clears the Set.
  - Hidden state is per-session, client-only. It does not persist across reloads and does not mutate the cache. (Rationale: a reviewer's visual decluttering preference is not a property of the pair match itself.)

**Styling:** a small `local_match.css` (or appended to `app.css` with the existing `?v=` cache-bust pattern). Overlay SVG has `pointer-events: none`; individual `<line>` elements opt back in with `pointer-events: auto` so clicks register without stealing events from the underlying image.

### 5. CLI — bulk precompute

New subcommand in `cli.py`:

```
python -m whaleshark_reid.cli precompute-matches \
  --run-id <id> [--extractor aliked] [--limit N] [--overwrite]
```

Flow:
1. Select pairs from `pair_queue` for `run_id`, skipping those already present in `pair_matches` (unless `--overwrite`).
2. Collect the unique set of annotation UUIDs across those pairs.
3. `extract_features_batch` runs the extractor in minibatches, building a feats dict keyed by UUID.
4. `match_pairs_batch` runs the matcher over the pairs using cached feats. LightGlue batches pair inputs when images are same-sized, and we already pad/resize to 440×440, so this is a real batch and not a Python loop in disguise.
5. Writes each `MatchResult` into `pair_matches`. Resume-safe because of the PK; a Ctrl-C mid-run leaves consistent state.

### 6. Install / deps

- `lightglue` added as an optional extra in `pyproject.toml`: `[project.optional-dependencies] local-match = ["lightglue @ git+https://github.com/cvg/LightGlue.git", "torch"]`.
- A single import site (`whaleshark_reid.core.match.lightglue`) so the rest of the codebase is unaware of LightGlue.
- The matcher endpoints + CLI command return a clear 503 / error message when the extra is missing — nothing crashes on import.

## Testing

- **Unit**
  - `MatchResult` ↔ JSON round-trip (service-layer serde).
  - Feature-batch dedup: passing a list with repeated annotation UUIDs extracts each only once.
  - Cache hit/miss behavior in the service layer (mocked `LocalMatcher`).
- **Integration (FastAPI test client)**
  - GET on a fresh pair → 404. POST → 200. GET → 200 with same payload.
  - `pair_matches` PK prevents duplicate rows for `(queue_id, extractor)`; POST without `overwrite` on an existing row returns the cached row (idempotent). POST with `overwrite=1` updates the row.
  - CLI `precompute-matches` is resume-safe (run twice without `--overwrite` is a no-op on second run).
- **Migration**
  - Fresh DB → `pair_matches` exists after schema init.
  - Pre-existing DB → `pair_matches` added by `_apply_migrations` without touching other tables.
- **LightGlue-gated tests**
  - Any test that actually calls the real matcher is marked `@pytest.mark.lightglue` and skipped by default.
  - Route/service tests use a `FakeLocalMatcher` fixture returning canned `MatchResult`s. This keeps CI fast and independent of GPU / LightGlue install.

## File layout

**New:**
- `src/whaleshark_reid/core/match/__init__.py`
- `src/whaleshark_reid/core/match/lightglue.py`
- `src/whaleshark_reid/web/routes/local_match.py`
- `src/whaleshark_reid/web/services/local_match.py`
- `src/whaleshark_reid/web/static/js/local_match.js`
- `src/whaleshark_reid/web/static/css/local_match.css` (or appended to `app.css`)

**Touched:**
- `src/whaleshark_reid/storage/schema.sql` (add `pair_matches`)
- `src/whaleshark_reid/storage/db.py` (migration branch for `pair_matches`)
- `src/whaleshark_reid/web/templates/partials/pair_card.html` (chip-wrap, SVG, controls, JS hook)
- `src/whaleshark_reid/web/app.py` (mount new router)
- `src/whaleshark_reid/cli.py` (new `precompute-matches` subcommand)
- `pyproject.toml` (`local-match` optional extra)

## Deferred (future work, not this spec)

- MiewID + LightGlue score fusion (the "hybrid" approach).
- Reranking top-K MiewID candidates with LightGlue.
- PNG export of a rendered visualization (for sharing / reports).
- SuperPoint and DISK extractor wiring (module supports them via `get_matcher(extractor=...)`, but only ALIKED is initially exercised and tested).
- Persisting per-user "hidden line" preferences.
