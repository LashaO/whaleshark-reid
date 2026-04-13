# Local Feature Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an interactive ALIKED+LightGlue local-feature overlay on the pair review card — single-pair on-demand, hybrid caching (auto-render if cached), client-side confidence slider + line toggles + click-to-hide, plus a bulk-precompute CLI.

**Architecture:** New `core/match/lightglue.py` holds a lazy-singleton matcher with per-annotation feature dedup. A new `pair_matches` SQLite table caches results keyed by `(queue_id, extractor)`. Two FastAPI endpoints (GET/POST `/api/pairs/{id}/local-match`) plus a `catalog match-local` CLI. The pair card renders a per-pair SVG overlay with a small JS module.

**Tech Stack:** Python 3.10+, FastAPI, Jinja2, SQLite, Typer, ALIKED+LightGlue (cvg/LightGlue, installed as optional extra), pytest, vanilla JS + SVG.

**Spec:** `docs/superpowers/specs/2026-04-13-local-feature-matching-design.md`

---

## File Structure

**New:**
- `src/whaleshark_reid/core/match/__init__.py` — package marker
- `src/whaleshark_reid/core/match/lightglue.py` — `MatchResult`, `LocalMatcher`, `match_pair`, `extract_features_batch`, `match_pairs_batch`, `get_matcher`
- `src/whaleshark_reid/web/services/local_match.py` — cache read/write + glue to `core.match.lightglue`
- `src/whaleshark_reid/web/routes/local_match.py` — GET + POST `/api/pairs/{queue_id}/local-match`
- `src/whaleshark_reid/web/static/js/local_match.js` — client rendering + interactions
- `src/whaleshark_reid/cli/commands/match_local.py` — `match-local` CLI command
- `tests/core/test_match_lightglue.py`
- `tests/web/test_local_match_routes.py`
- `tests/cli/test_cli_match_local.py`

**Modified:**
- `src/whaleshark_reid/storage/schema.sql` — add `pair_matches` table
- `src/whaleshark_reid/storage/db.py` — migration branch in `_apply_migrations`
- `src/whaleshark_reid/web/app.py` — mount new router
- `src/whaleshark_reid/web/templates/partials/pair_card.html` — chip-wrap + SVG host + controls + JS hook
- `src/whaleshark_reid/web/static/css/app.css` — styles for chip-wrap, overlay, controls
- `src/whaleshark_reid/cli/main.py` — register `match-local`
- `pyproject.toml` — `[local-match]` optional extra

---

## Task 1: Add `pair_matches` schema + migration

**Files:**
- Modify: `src/whaleshark_reid/storage/schema.sql`
- Modify: `src/whaleshark_reid/storage/db.py` (`_apply_migrations`)
- Test: `tests/storage/test_migrations.py` (extend or create)

- [ ] **Step 1: Write the failing test**

Add `tests/storage/test_pair_matches_migration.py`:

```python
"""Verify pair_matches table is created by schema init + additive migration."""
from pathlib import Path

from whaleshark_reid.storage.db import Storage


def _table_columns(storage: Storage, table: str) -> set[str]:
    return {r["name"] for r in storage.conn.execute(f"PRAGMA table_info({table})").fetchall()}


def test_pair_matches_created_on_fresh_db(tmp_db_path: Path):
    s = Storage(tmp_db_path)
    s.init_schema()
    cols = _table_columns(s, "pair_matches")
    assert {"queue_id", "extractor", "n_matches", "mean_score",
            "median_score", "match_data", "img_a_size", "img_b_size",
            "computed_at"}.issubset(cols)


def test_pair_matches_added_by_migration(tmp_db_path: Path):
    """A DB that pre-dates pair_matches should get it via _apply_migrations."""
    s = Storage(tmp_db_path)
    # Bare schema without pair_matches
    s.conn.executescript("""
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, stage TEXT, config_json TEXT,
            metrics_json TEXT, notes TEXT, git_sha TEXT, started_at TEXT,
            finished_at TEXT, status TEXT, error TEXT);
        CREATE TABLE annotations (annotation_uuid TEXT PRIMARY KEY);
        CREATE TABLE pair_queue (queue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT, ann_a_uuid TEXT, ann_b_uuid TEXT,
            distance REAL, cluster_a INTEGER, cluster_b INTEGER,
            same_cluster INTEGER, position INTEGER);
    """)
    assert "pair_matches" not in {
        r["name"] for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    s._apply_migrations()
    assert "pair_matches" in {
        r["name"] for r in s.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def test_pair_matches_pk_prevents_duplicate_extractor(tmp_db_path: Path):
    import pytest, sqlite3
    s = Storage(tmp_db_path)
    s.init_schema()
    s.conn.execute("INSERT INTO runs(run_id, stage, config_json, started_at, status) VALUES('r','match','{}', '2026-04-13', 'ok')")
    s.conn.execute("INSERT INTO annotations(annotation_uuid) VALUES('a'),('b')")
    s.conn.execute(
        "INSERT INTO pair_queue(run_id, ann_a_uuid, ann_b_uuid, distance, position) VALUES('r','a','b', 0.1, 0)"
    )
    qid = s.conn.execute("SELECT queue_id FROM pair_queue").fetchone()["queue_id"]
    s.conn.execute(
        "INSERT INTO pair_matches(queue_id, extractor, n_matches, match_data, img_a_size, img_b_size) "
        "VALUES(?, 'aliked', 12, '{}', '[440,440]', '[440,440]')", (qid,)
    )
    with pytest.raises(sqlite3.IntegrityError):
        s.conn.execute(
            "INSERT INTO pair_matches(queue_id, extractor, n_matches, match_data, img_a_size, img_b_size) "
            "VALUES(?, 'aliked', 99, '{}', '[440,440]', '[440,440]')", (qid,)
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/storage/test_pair_matches_migration.py -v`
Expected: FAIL — no `pair_matches` table.

- [ ] **Step 3: Add table to schema.sql**

Append to `src/whaleshark_reid/storage/schema.sql`:

```sql
CREATE TABLE IF NOT EXISTS pair_matches (
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
CREATE INDEX IF NOT EXISTS idx_pm_queue ON pair_matches(queue_id);
```

- [ ] **Step 4: Add migration branch in db.py**

In `src/whaleshark_reid/storage/db.py`, extend `_apply_migrations`:

```python
def _apply_migrations(self) -> None:
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
```

- [ ] **Step 5: Run tests to verify pass**

Run: `pytest tests/storage/test_pair_matches_migration.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/whaleshark_reid/storage/schema.sql src/whaleshark_reid/storage/db.py tests/storage/test_pair_matches_migration.py
git commit -m "Add pair_matches cache table with additive migration"
```

---

## Task 2: `MatchResult` dataclass + JSON serde

**Files:**
- Create: `src/whaleshark_reid/core/match/__init__.py` (empty)
- Create: `src/whaleshark_reid/core/match/lightglue.py`
- Create: `tests/core/test_match_lightglue.py`

- [ ] **Step 1: Write failing tests**

Create `tests/core/test_match_lightglue.py`:

```python
"""Unit tests for core.match.lightglue that do NOT require the real LightGlue install.

Anything that calls the real extractor/matcher is marked `@pytest.mark.lightglue`
and skipped by default; CI/GPU runs can opt in with `-m lightglue`.
"""
import json

import pytest

from whaleshark_reid.core.match.lightglue import MatchResult


def test_match_result_roundtrip_json():
    mr = MatchResult(
        extractor="aliked",
        n_matches=2,
        mean_score=0.6,
        median_score=0.6,
        kpts_a=[[1.0, 2.0], [3.0, 4.0]],
        kpts_b=[[5.0, 6.0], [7.0, 8.0]],
        matches=[[0, 0, 0.8], [1, 1, 0.4]],
        img_a_size=[440, 440],
        img_b_size=[440, 440],
    )
    blob = mr.to_json_dict()
    assert set(blob.keys()) == {
        "extractor", "n_matches", "mean_score", "median_score",
        "kpts_a", "kpts_b", "matches", "img_a_size", "img_b_size",
    }
    # Round trip through real JSON
    roundtripped = MatchResult.from_json_dict(json.loads(json.dumps(blob)))
    assert roundtripped == mr


def test_n_matches_counts_confidence_threshold():
    """n_matches counts score >= threshold."""
    from whaleshark_reid.core.match.lightglue import count_confident_matches
    assert count_confident_matches([[0, 0, 0.8], [1, 1, 0.4], [2, 2, 0.5]], thr=0.5) == 2
    assert count_confident_matches([], thr=0.5) == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/core/test_match_lightglue.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement dataclass + helper**

Create `src/whaleshark_reid/core/match/__init__.py` (empty file).

Create `src/whaleshark_reid/core/match/lightglue.py`:

```python
"""ALIKED + LightGlue local feature matching for the pair review UI.

This module isolates the only code path that imports `lightglue`. Everything
else in the project can be unaware of it. Import failure (the optional extra
isn't installed) raises LightGlueUnavailable only when the matcher is actually
requested — not at module import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


class LightGlueUnavailable(RuntimeError):
    """Raised when LightGlue is not installed and the matcher is requested."""


@dataclass
class MatchResult:
    extractor: str
    n_matches: int
    mean_score: float | None
    median_score: float | None
    kpts_a: list[list[float]]
    kpts_b: list[list[float]]
    matches: list[list[float]]  # [i, j, score]
    img_a_size: list[int]       # [w, h] — coord space of kpts_a
    img_b_size: list[int]

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict) -> "MatchResult":
        return cls(**d)


def count_confident_matches(matches: Iterable[list[float]], thr: float = 0.5) -> int:
    return sum(1 for m in matches if m[2] >= thr)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/core/test_match_lightglue.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/core/match/ tests/core/test_match_lightglue.py
git commit -m "Add MatchResult dataclass and confidence-threshold helper"
```

---

## Task 3: `LocalMatcher` singleton + `match_pair` (with fakeable seam)

**Files:**
- Modify: `src/whaleshark_reid/core/match/lightglue.py`
- Modify: `tests/core/test_match_lightglue.py`

- [ ] **Step 1: Write failing test for the fake seam**

The real extractor/matcher classes are monkeypatchable via `get_matcher`. Callers never instantiate them directly; they call `match_pair(img_a, img_b, extractor=...)`, which defers to `get_matcher(extractor).match_pair(img_a, img_b)`. That lets tests inject a stub.

Append to `tests/core/test_match_lightglue.py`:

```python
def test_match_pair_delegates_to_singleton(monkeypatch):
    """match_pair must call get_matcher(extractor).match_pair(a, b)."""
    from whaleshark_reid.core.match import lightglue as lg

    calls = []

    class FakeMatcher:
        def match_pair(self, a, b):
            calls.append((a, b))
            return MatchResult(
                extractor="aliked", n_matches=1, mean_score=0.9, median_score=0.9,
                kpts_a=[[0, 0]], kpts_b=[[1, 1]], matches=[[0, 0, 0.9]],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )

    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": FakeMatcher())
    out = lg.match_pair("/path/a.jpg", "/path/b.jpg", extractor="aliked")
    assert out.n_matches == 1
    assert calls == [("/path/a.jpg", "/path/b.jpg")]


def test_get_matcher_caches_per_extractor(monkeypatch):
    """get_matcher should return the same instance on repeat calls for same extractor."""
    from whaleshark_reid.core.match import lightglue as lg

    # Reset the cache so the test is independent of ordering.
    lg._MATCHER_CACHE.clear()

    build_calls = []

    class FakeMatcher:
        def __init__(self, extractor):
            build_calls.append(extractor)
            self.extractor = extractor

    monkeypatch.setattr(lg, "_build_matcher", lambda extractor: FakeMatcher(extractor))

    a1 = lg.get_matcher("aliked")
    a2 = lg.get_matcher("aliked")
    b1 = lg.get_matcher("superpoint")
    assert a1 is a2
    assert a1 is not b1
    assert build_calls == ["aliked", "superpoint"]
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/core/test_match_lightglue.py -v`
Expected: last 2 FAIL — names not defined.

- [ ] **Step 3: Add seam + cache + real builder**

Append to `src/whaleshark_reid/core/match/lightglue.py`:

```python
# ---- Matcher singleton / seam ------------------------------------------------
#
# Tests inject a fake via monkeypatch on `_build_matcher` or `get_matcher`.
# Real matcher construction lives in _build_matcher, which imports lightglue
# lazily so the module stays importable without the optional dep.

_MATCHER_CACHE: dict[str, "LocalMatcher"] = {}


class LocalMatcher:
    """ALIKED+LightGlue (or SuperPoint+LightGlue, etc.) bound to a device."""

    def __init__(self, extractor: str):
        import torch

        try:
            from lightglue import ALIKED, SuperPoint, DISK, LightGlue
            from lightglue.utils import load_image, rbd
        except ImportError as e:
            raise LightGlueUnavailable(
                "LightGlue is not installed. Install the optional extra: "
                "`pip install 'whaleshark-reid[local-match]'`"
            ) from e

        self.extractor_name = extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._load_image = load_image
        self._rbd = rbd

        if extractor == "aliked":
            self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
            features = "aliked"
        elif extractor == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            features = "superpoint"
        elif extractor == "disk":
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
            features = "disk"
        else:
            raise ValueError(f"unknown extractor: {extractor!r}")
        self.matcher = LightGlue(features=features).eval().to(self.device)

    @staticmethod
    def _statsf(scores: list[float]) -> tuple[float | None, float | None]:
        if not scores:
            return None, None
        xs = sorted(scores)
        mean = sum(xs) / len(xs)
        median = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
        return float(mean), float(median)

    def _extract(self, img_path: str):
        img = self._load_image(img_path, resize=440).to(self.device)
        with self._torch.inference_mode():
            feats = self.extractor.extract(img)
        return feats, img.shape[-1], img.shape[-2]  # width, height

    def match_pair(self, img_a_path: str, img_b_path: str) -> MatchResult:
        feats_a, wa, ha = self._extract(img_a_path)
        feats_b, wb, hb = self._extract(img_b_path)
        with self._torch.inference_mode():
            out = self.matcher({"image0": feats_a, "image1": feats_b})
        feats_a, feats_b, out = [self._rbd(x) for x in (feats_a, feats_b, out)]

        kpts_a = feats_a["keypoints"].cpu().tolist()
        kpts_b = feats_b["keypoints"].cpu().tolist()
        pairs = out["matches"].cpu().tolist()          # list of [i, j]
        scores = out["scores"].cpu().tolist()          # list of float

        matches = [[int(i), int(j), float(s)] for (i, j), s in zip(pairs, scores)]
        mean, median = self._statsf(scores)
        return MatchResult(
            extractor=self.extractor_name,
            n_matches=count_confident_matches(matches, thr=0.5),
            mean_score=mean,
            median_score=median,
            kpts_a=kpts_a,
            kpts_b=kpts_b,
            matches=matches,
            img_a_size=[int(wa), int(ha)],
            img_b_size=[int(wb), int(hb)],
        )


def _build_matcher(extractor: str) -> LocalMatcher:
    return LocalMatcher(extractor)


def get_matcher(extractor: str = "aliked") -> LocalMatcher:
    if extractor not in _MATCHER_CACHE:
        _MATCHER_CACHE[extractor] = _build_matcher(extractor)
    return _MATCHER_CACHE[extractor]


def match_pair(img_a_path: str, img_b_path: str, extractor: str = "aliked") -> MatchResult:
    return get_matcher(extractor).match_pair(img_a_path, img_b_path)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_match_lightglue.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/core/match/lightglue.py tests/core/test_match_lightglue.py
git commit -m "Add LocalMatcher with ALIKED+LightGlue and cached singleton"
```

---

## Task 4: Batched feature extraction + pairs batch

**Files:**
- Modify: `src/whaleshark_reid/core/match/lightglue.py`
- Modify: `tests/core/test_match_lightglue.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/core/test_match_lightglue.py`:

```python
def test_extract_features_batch_dedupes_by_path(monkeypatch):
    """Passing repeated paths must only extract each unique path once."""
    from whaleshark_reid.core.match import lightglue as lg
    lg._MATCHER_CACHE.clear()

    extract_calls = []

    class FakeMatcher:
        extractor_name = "aliked"
        def _extract(self, path):
            extract_calls.append(path)
            return {"keypoints": [[1, 2]]}, 440, 440  # feats, w, h

    monkeypatch.setattr(lg, "_build_matcher", lambda e: FakeMatcher())
    feats_by_path = lg.extract_features_batch(
        ["a.jpg", "b.jpg", "a.jpg", "b.jpg", "c.jpg"], extractor="aliked",
    )
    assert set(feats_by_path.keys()) == {"a.jpg", "b.jpg", "c.jpg"}
    assert sorted(extract_calls) == ["a.jpg", "b.jpg", "c.jpg"]


def test_match_pairs_batch_uses_cached_feats(monkeypatch):
    from whaleshark_reid.core.match import lightglue as lg
    lg._MATCHER_CACHE.clear()

    match_calls = []

    class FakeMatcher:
        extractor_name = "aliked"
        def _match_prebuilt(self, feats_a, feats_b, size_a, size_b):
            match_calls.append((feats_a, feats_b))
            return MatchResult(
                extractor="aliked", n_matches=3, mean_score=0.7, median_score=0.7,
                kpts_a=[], kpts_b=[], matches=[[0, 0, 0.9]],
                img_a_size=list(size_a), img_b_size=list(size_b),
            )

    monkeypatch.setattr(lg, "_build_matcher", lambda e: FakeMatcher())
    feats = {"a.jpg": ("FA", (440, 440)), "b.jpg": ("FB", (440, 440))}
    results = lg.match_pairs_batch(
        [("a.jpg", "b.jpg"), ("a.jpg", "b.jpg")], feats, extractor="aliked"
    )
    assert len(results) == 2
    assert all(r.n_matches == 3 for r in results)
    # Each pair triggers one match call (no re-extraction)
    assert len(match_calls) == 2
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/core/test_match_lightglue.py -v`
Expected: 2 FAIL — `extract_features_batch`, `match_pairs_batch` not defined.

- [ ] **Step 3: Add batch functions + `_match_prebuilt`**

Append to `src/whaleshark_reid/core/match/lightglue.py`:

```python
def extract_features_batch(
    image_paths: list[str], extractor: str = "aliked",
) -> dict[str, tuple[object, tuple[int, int]]]:
    """Extract features once per unique path.

    Returns mapping path -> (feats, (w, h)).
    """
    m = get_matcher(extractor)
    out: dict[str, tuple[object, tuple[int, int]]] = {}
    for path in set(image_paths):
        feats, w, h = m._extract(path)
        out[path] = (feats, (w, h))
    return out


def match_pairs_batch(
    pairs: list[tuple[str, str]],
    feats_by_path: dict[str, tuple[object, tuple[int, int]]],
    extractor: str = "aliked",
) -> list[MatchResult]:
    m = get_matcher(extractor)
    results: list[MatchResult] = []
    for a, b in pairs:
        fa, sa = feats_by_path[a]
        fb, sb = feats_by_path[b]
        results.append(m._match_prebuilt(fa, fb, sa, sb))
    return results
```

Also add the `_match_prebuilt` method to `LocalMatcher` (refactor of `match_pair`) — insert inside the class, and have `match_pair` call it:

```python
    def _match_prebuilt(self, feats_a, feats_b, size_a, size_b) -> MatchResult:
        with self._torch.inference_mode():
            out = self.matcher({"image0": feats_a, "image1": feats_b})
        fa, fb, out = [self._rbd(x) for x in (feats_a, feats_b, out)]

        kpts_a = fa["keypoints"].cpu().tolist()
        kpts_b = fb["keypoints"].cpu().tolist()
        pairs_ij = out["matches"].cpu().tolist()
        scores = out["scores"].cpu().tolist()
        matches = [[int(i), int(j), float(s)] for (i, j), s in zip(pairs_ij, scores)]
        mean, median = self._statsf(scores)
        return MatchResult(
            extractor=self.extractor_name, n_matches=count_confident_matches(matches, 0.5),
            mean_score=mean, median_score=median,
            kpts_a=kpts_a, kpts_b=kpts_b, matches=matches,
            img_a_size=[int(size_a[0]), int(size_a[1])],
            img_b_size=[int(size_b[0]), int(size_b[1])],
        )
```

And refactor `match_pair` on `LocalMatcher` to:

```python
    def match_pair(self, img_a_path: str, img_b_path: str) -> MatchResult:
        feats_a, wa, ha = self._extract(img_a_path)
        feats_b, wb, hb = self._extract(img_b_path)
        return self._match_prebuilt(feats_a, feats_b, (wa, ha), (wb, hb))
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/core/test_match_lightglue.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/core/match/lightglue.py tests/core/test_match_lightglue.py
git commit -m "Add batched feature extraction and pairs matching with per-path dedup"
```

---

## Task 5: Service layer — cache read/write

**Files:**
- Create: `src/whaleshark_reid/web/services/local_match.py`
- Create: `tests/web/test_local_match_service.py`

- [ ] **Step 1: Write failing tests**

Create `tests/web/test_local_match_service.py`:

```python
from pathlib import Path

from whaleshark_reid.core.match.lightglue import MatchResult
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.services import local_match as svc


def _seed_pair(s: Storage) -> int:
    s.conn.execute("INSERT INTO runs(run_id, stage, config_json, started_at, status) VALUES('r','match','{}', '2026-04-13', 'ok')")
    s.conn.execute("INSERT INTO annotations(annotation_uuid, file_path, bbox_json, theta) VALUES('a','/x/a.jpg','[0,0,10,10]', 0),('b','/x/b.jpg','[0,0,10,10]', 0)")
    s.conn.execute("INSERT INTO pair_queue(run_id, ann_a_uuid, ann_b_uuid, distance, position) VALUES('r','a','b',0.1,0)")
    return s.conn.execute("SELECT queue_id FROM pair_queue").fetchone()["queue_id"]


def _make_result() -> MatchResult:
    return MatchResult(
        extractor="aliked", n_matches=2, mean_score=0.7, median_score=0.7,
        kpts_a=[[1, 2]], kpts_b=[[3, 4]], matches=[[0, 0, 0.9], [0, 0, 0.6]],
        img_a_size=[440, 440], img_b_size=[440, 440],
    )


def test_read_returns_none_if_uncached(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    assert svc.read_cached(s, qid, "aliked") is None


def test_write_then_read(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    svc.write_cached(s, qid, _make_result())
    got = svc.read_cached(s, qid, "aliked")
    assert got is not None
    assert got.n_matches == 2
    assert got.kpts_a == [[1, 2]]


def test_write_overwrite_replaces_row(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    svc.write_cached(s, qid, _make_result())
    updated = MatchResult(
        extractor="aliked", n_matches=99, mean_score=0.1, median_score=0.1,
        kpts_a=[], kpts_b=[], matches=[],
        img_a_size=[440, 440], img_b_size=[440, 440],
    )
    svc.write_cached(s, qid, updated)  # upsert
    got = svc.read_cached(s, qid, "aliked")
    assert got.n_matches == 99


def test_lookup_pair_paths(tmp_db_path: Path):
    s = Storage(tmp_db_path); s.init_schema()
    qid = _seed_pair(s)
    paths = svc.lookup_pair_image_paths(s, qid)
    assert paths == ("/x/a.jpg", "/x/b.jpg")
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/web/test_local_match_service.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement service**

Create `src/whaleshark_reid/web/services/local_match.py`:

```python
"""Service layer: cache read/write for local feature matches."""
from __future__ import annotations

import json

from whaleshark_reid.core.match.lightglue import MatchResult
from whaleshark_reid.storage.db import Storage


def read_cached(storage: Storage, queue_id: int, extractor: str) -> MatchResult | None:
    row = storage.conn.execute(
        "SELECT * FROM pair_matches WHERE queue_id = ? AND extractor = ?",
        (queue_id, extractor),
    ).fetchone()
    if row is None:
        return None
    blob = json.loads(row["match_data"])
    return MatchResult(
        extractor=row["extractor"],
        n_matches=row["n_matches"],
        mean_score=row["mean_score"],
        median_score=row["median_score"],
        kpts_a=blob["kpts_a"],
        kpts_b=blob["kpts_b"],
        matches=blob["matches"],
        img_a_size=json.loads(row["img_a_size"]),
        img_b_size=json.loads(row["img_b_size"]),
    )


def write_cached(storage: Storage, queue_id: int, result: MatchResult) -> None:
    blob = json.dumps({
        "kpts_a": result.kpts_a,
        "kpts_b": result.kpts_b,
        "matches": result.matches,
    })
    storage.conn.execute(
        """
        INSERT INTO pair_matches(queue_id, extractor, n_matches, mean_score,
                                 median_score, match_data, img_a_size, img_b_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(queue_id, extractor) DO UPDATE SET
            n_matches = excluded.n_matches,
            mean_score = excluded.mean_score,
            median_score = excluded.median_score,
            match_data = excluded.match_data,
            img_a_size = excluded.img_a_size,
            img_b_size = excluded.img_b_size,
            computed_at = CURRENT_TIMESTAMP
        """,
        (
            queue_id, result.extractor, result.n_matches,
            result.mean_score, result.median_score, blob,
            json.dumps(result.img_a_size), json.dumps(result.img_b_size),
        ),
    )


def lookup_pair_image_paths(storage: Storage, queue_id: int) -> tuple[str, str]:
    """Returns (path_a, path_b) for the two annotations in a pair."""
    row = storage.conn.execute(
        """
        SELECT a.file_path AS pa, b.file_path AS pb
        FROM pair_queue pq
        JOIN annotations a ON a.annotation_uuid = pq.ann_a_uuid
        JOIN annotations b ON b.annotation_uuid = pq.ann_b_uuid
        WHERE pq.queue_id = ?
        """,
        (queue_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"no pair_queue row with queue_id={queue_id}")
    return row["pa"], row["pb"]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/web/test_local_match_service.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whaleshark_reid/web/services/local_match.py tests/web/test_local_match_service.py
git commit -m "Add local-match cache service with upsert semantics"
```

---

## Task 6: GET/POST routes

**Files:**
- Create: `src/whaleshark_reid/web/routes/local_match.py`
- Modify: `src/whaleshark_reid/web/app.py`
- Create: `tests/web/test_local_match_routes.py`

- [ ] **Step 1: Write failing tests**

Create `tests/web/test_local_match_routes.py`:

```python
"""Route tests using a fake matcher (no LightGlue required)."""
import re

from starlette.testclient import TestClient

from whaleshark_reid.core.match import lightglue as lg
from whaleshark_reid.core.match.lightglue import MatchResult


def _install_fake(monkeypatch):
    """Replace get_matcher with a deterministic stub."""
    class FakeMatcher:
        def match_pair(self, a, b):
            return MatchResult(
                extractor="aliked", n_matches=3, mean_score=0.8, median_score=0.8,
                kpts_a=[[1, 2], [3, 4]], kpts_b=[[5, 6], [7, 8]],
                matches=[[0, 0, 0.9], [1, 1, 0.7], [0, 1, 0.3]],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )
    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": FakeMatcher())


def _find_queue_id(html: str) -> int:
    m = re.search(r'data-queue-id="(\d+)"', html)
    assert m, "no data-queue-id in carousel"
    return int(m.group(1))


def test_get_returns_404_when_uncached(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)
    r = seeded_web_client.get(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 404


def test_post_runs_matcher_and_get_returns_cached(seeded_web_client: TestClient, monkeypatch):
    _install_fake(monkeypatch)
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 200
    body = r.json()
    assert body["extractor"] == "aliked"
    assert body["n_matches"] == 3
    assert body["matches"][0] == [0, 0, 0.9]

    # Second call hits cache and returns same payload
    r2 = seeded_web_client.get(f"/api/pairs/{qid}/local-match")
    assert r2.status_code == 200
    assert r2.json()["n_matches"] == 3


def test_post_without_overwrite_is_idempotent(seeded_web_client: TestClient, monkeypatch):
    _install_fake(monkeypatch)
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    seeded_web_client.post(f"/api/pairs/{qid}/local-match")

    call_count = {"n": 0}
    class CountingMatcher:
        def match_pair(self, a, b):
            call_count["n"] += 1
            return MatchResult(
                extractor="aliked", n_matches=99, mean_score=0.1, median_score=0.1,
                kpts_a=[], kpts_b=[], matches=[],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )
    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": CountingMatcher())

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 200
    # Cached, so matcher was NOT called again
    assert call_count["n"] == 0
    assert r.json()["n_matches"] == 3


def test_post_with_overwrite_recomputes(seeded_web_client: TestClient, monkeypatch):
    _install_fake(monkeypatch)
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    seeded_web_client.post(f"/api/pairs/{qid}/local-match")

    class NewerMatcher:
        def match_pair(self, a, b):
            return MatchResult(
                extractor="aliked", n_matches=42, mean_score=0.2, median_score=0.2,
                kpts_a=[], kpts_b=[], matches=[],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )
    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": NewerMatcher())

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match?overwrite=1")
    assert r.status_code == 200
    assert r.json()["n_matches"] == 42


def test_get_404_for_unknown_queue_id(seeded_web_client: TestClient):
    r = seeded_web_client.get("/api/pairs/999999/local-match")
    assert r.status_code == 404


def test_lightglue_unavailable_returns_503(seeded_web_client: TestClient, monkeypatch):
    from whaleshark_reid.core.match.lightglue import LightGlueUnavailable
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    def raiser(extractor="aliked"):
        raise LightGlueUnavailable("not installed")
    monkeypatch.setattr(lg, "get_matcher", raiser)

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 503
```

- [ ] **Step 2: Run tests to verify fail**

Run: `pytest tests/web/test_local_match_routes.py -v`
Expected: FAIL — endpoints missing.

- [ ] **Step 3: Create the route**

Create `src/whaleshark_reid/web/routes/local_match.py`:

```python
"""GET + POST /api/pairs/{queue_id}/local-match."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from whaleshark_reid.core.match import lightglue as lg_module
from whaleshark_reid.core.match.lightglue import LightGlueUnavailable
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_storage
from whaleshark_reid.web.services import local_match as svc

router = APIRouter()


@router.get("/api/pairs/{queue_id}/local-match")
def get_local_match(
    queue_id: int,
    extractor: str = "aliked",
    storage: Storage = Depends(get_storage),
):
    # Verify the pair exists before answering cache miss vs unknown
    try:
        svc.lookup_pair_image_paths(storage, queue_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="pair not found")

    cached = svc.read_cached(storage, queue_id, extractor)
    if cached is None:
        raise HTTPException(status_code=404, detail="not cached")
    return cached.to_json_dict()


@router.post("/api/pairs/{queue_id}/local-match")
def post_local_match(
    queue_id: int,
    extractor: str = "aliked",
    overwrite: int = 0,
    storage: Storage = Depends(get_storage),
):
    try:
        path_a, path_b = svc.lookup_pair_image_paths(storage, queue_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="pair not found")

    if not overwrite:
        cached = svc.read_cached(storage, queue_id, extractor)
        if cached is not None:
            return cached.to_json_dict()

    try:
        matcher = lg_module.get_matcher(extractor)
    except LightGlueUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = matcher.match_pair(path_a, path_b)
    svc.write_cached(storage, queue_id, result)
    return result.to_json_dict()
```

- [ ] **Step 4: Mount in `web/app.py`**

Inside `create_app()` near the other `include_router` calls (after `pairs` is mounted):

```python
    from whaleshark_reid.web.routes import local_match
    app.include_router(local_match.router)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/web/test_local_match_routes.py -v`
Expected: 6 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/whaleshark_reid/web/routes/local_match.py src/whaleshark_reid/web/app.py tests/web/test_local_match_routes.py
git commit -m "Add GET/POST /api/pairs/{id}/local-match endpoints with cache + overwrite"
```

---

## Task 7: Frontend — SVG overlay, slider, toggles, click-to-hide

**Files:**
- Create: `src/whaleshark_reid/web/static/js/local_match.js`
- Modify: `src/whaleshark_reid/web/static/css/app.css`
- Modify: `src/whaleshark_reid/web/templates/partials/pair_card.html`
- Create: `tests/web/test_local_match_frontend_hooks.py`

- [ ] **Step 1: Write failing test for template hooks**

Create `tests/web/test_local_match_frontend_hooks.py`:

```python
"""Confirm the pair card exposes the DOM hooks the JS module relies on."""
from starlette.testclient import TestClient


def test_pair_card_has_chip_wrap_hooks(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    assert page.status_code == 200
    html = page.text
    # Both chips wrapped for overlay positioning
    assert html.count('class="chip-wrap"') >= 2 or html.count("chip-wrap") >= 2
    # Controls container
    assert 'class="local-match-controls"' in html
    # Shared SVG host for lines across the gap
    assert 'id="local-match-overlay"' in html
    # JS included
    assert 'local_match.js' in html


def test_pair_card_loads_local_match_css_with_cache_bust(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    # existing app.css version bumped OR new file, either way: must appear
    assert "app.css?v=" in page.text
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/web/test_local_match_frontend_hooks.py -v`
Expected: FAIL — hooks not in template.

- [ ] **Step 3: Edit `pair_card.html`**

Find the existing `.pair-layout` block and replace the two chip image blocks and map div with this structure (keep `meta-panel` dl's as-is; only wrap the images):

```html
<div class="pair-layout">
    <div>
        <div class="chip-wrap" data-chip="a">
            <img class="ann-img" src="/image/{{ pair.ann_a.annotation_uuid }}?crop=true" alt="ann_a">
        </div>
        <dl class="meta-panel"> ... unchanged ... </dl>
    </div>
    <div>
        <div class="chip-wrap" data-chip="b">
            <img class="ann-img" src="/image/{{ pair.ann_b.annotation_uuid }}?crop=true" alt="ann_b">
        </div>
        <dl class="meta-panel"> ... unchanged ... </dl>
    </div>
    <div>
        <div class="map-inset" id="map-inset" ...> ... </div>
    </div>
</div>

<!-- Shared overlay spanning the whole layout so lines can cross the gap -->
<svg id="local-match-overlay" class="local-match-overlay" aria-hidden="true"></svg>

<div class="local-match-controls" data-queue-id="{{ pair.queue_id }}" data-extractor="aliked">
    <button type="button" class="btn lm-run" style="display:none;">🔍 Run local match</button>
    <span class="lm-status" style="color:var(--text-muted)"></span>
    <label>conf ≥ <input type="range" class="lm-conf" min="0" max="1" step="0.01" value="0.5"></label>
    <span class="lm-conf-val">0.50</span>
    <label><input type="checkbox" class="lm-lines" checked> lines</label>
    <label><input type="checkbox" class="lm-kpts" checked> keypoints</label>
    <button type="button" class="btn lm-show-hidden" style="display:none;">Show hidden (<span class="lm-hidden-count">0</span>)</button>
    <button type="button" class="btn lm-rerun" style="display:none;">re-run</button>
    <span class="lm-stats" style="color:var(--text-muted); margin-left:auto;"></span>
</div>
```

At the bottom of the file (or in base.html, but keep it local since the card re-renders via HTMX), include the JS:

```html
<script src="/static/js/local_match.js?v=1"></script>
```

Bump `app.css?v=4` → `app.css?v=5` in `base.html`.

- [ ] **Step 4: Add styles**

Append to `src/whaleshark_reid/web/static/css/app.css`:

```css
.chip-wrap {
    position: relative;
    display: inline-block;
}
.local-match-overlay {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 5;
}
.local-match-overlay line { pointer-events: auto; cursor: pointer; }
.local-match-overlay line.hidden { opacity: 0.1; pointer-events: none; }
.local-match-overlay circle { pointer-events: none; }
.local-match-controls {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 0.75rem;
    padding: 0.5rem;
    background: var(--bg-card-alt, rgba(255,255,255,0.02));
    border-radius: 4px;
    font-size: 12px;
}
.local-match-controls input[type=range] { vertical-align: middle; }
```

- [ ] **Step 5: Implement JS**

Create `src/whaleshark_reid/web/static/js/local_match.js`:

```javascript
// Local feature match overlay for the pair card.
// Renders an SVG overlay spanning both chip images. On mount, tries GET
// /api/pairs/{id}/local-match; 200 → render, 404 → show "Run local match"
// button that POSTs. Controls for conf slider, line/keypoint toggles,
// click-to-hide, shift-click-to-hide-below.
(function () {
    const VIRIDIS = [
        [253, 231, 37], [180, 222, 44], [95, 201, 98],
        [33, 145, 140], [59, 82, 139], [68, 1, 84],
    ];
    function scoreColor(s) {
        const t = Math.max(0, Math.min(1, 1 - s)); // high score => warm end
        const i = t * (VIRIDIS.length - 1);
        const a = Math.floor(i), b = Math.min(a + 1, VIRIDIS.length - 1);
        const f = i - a;
        const c = VIRIDIS[a].map((v, k) => Math.round(v + (VIRIDIS[b][k] - v) * f));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
    }

    function init(card) {
        const ctrl = card.querySelector(".local-match-controls");
        if (!ctrl || ctrl.dataset.initialized) return;
        ctrl.dataset.initialized = "1";
        const queueId = ctrl.dataset.queueId;
        const extractor = ctrl.dataset.extractor || "aliked";

        const overlay = card.querySelector("#local-match-overlay");
        const chipA = card.querySelector('.chip-wrap[data-chip="a"]');
        const chipB = card.querySelector('.chip-wrap[data-chip="b"]');
        const runBtn = ctrl.querySelector(".lm-run");
        const rerunBtn = ctrl.querySelector(".lm-rerun");
        const status = ctrl.querySelector(".lm-status");
        const stats = ctrl.querySelector(".lm-stats");
        const conf = ctrl.querySelector(".lm-conf");
        const confVal = ctrl.querySelector(".lm-conf-val");
        const linesCk = ctrl.querySelector(".lm-lines");
        const kptsCk = ctrl.querySelector(".lm-kpts");
        const showHidden = ctrl.querySelector(".lm-show-hidden");
        const hiddenCount = ctrl.querySelector(".lm-hidden-count");

        const state = { result: null, hidden: new Set() };

        function chipBox(wrap) {
            const img = wrap.querySelector("img");
            const wrapRect = wrap.getBoundingClientRect();
            const cardRect = card.getBoundingClientRect();
            return {
                x: wrapRect.left - cardRect.left,
                y: wrapRect.top - cardRect.top,
                w: img.clientWidth,
                h: img.clientHeight,
            };
        }

        function render() {
            if (!state.result) {
                overlay.innerHTML = "";
                return;
            }
            const r = state.result;
            const boxA = chipBox(chipA), boxB = chipBox(chipB);
            const sxA = boxA.w / r.img_a_size[0], syA = boxA.h / r.img_a_size[1];
            const sxB = boxB.w / r.img_b_size[0], syB = boxB.h / r.img_b_size[1];
            const cardRect = card.getBoundingClientRect();
            overlay.setAttribute("viewBox", `0 0 ${cardRect.width} ${cardRect.height}`);
            overlay.setAttribute("width", cardRect.width);
            overlay.setAttribute("height", cardRect.height);

            const thr = parseFloat(conf.value);
            const showLines = linesCk.checked;
            const showKpts = kptsCk.checked;

            let html = "";
            if (showLines) {
                r.matches.forEach((m, idx) => {
                    const [i, j, s] = m;
                    if (s < thr) return;
                    if (state.hidden.has(idx)) {
                        html += `<line class="hidden" data-idx="${idx}" x1="0" y1="0" x2="0" y2="0"/>`;
                        return;
                    }
                    const pa = r.kpts_a[i], pb = r.kpts_b[j];
                    const x1 = boxA.x + pa[0] * sxA, y1 = boxA.y + pa[1] * syA;
                    const x2 = boxB.x + pb[0] * sxB, y2 = boxB.y + pb[1] * syB;
                    html += `<line data-idx="${idx}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${scoreColor(s)}" stroke-width="1" stroke-opacity="0.9"/>`;
                });
            }
            if (showKpts) {
                const visible = new Set();
                r.matches.forEach((m, idx) => {
                    if (m[2] < thr || state.hidden.has(idx)) return;
                    visible.add(`a${m[0]}`); visible.add(`b${m[1]}`);
                });
                visible.forEach(k => {
                    const side = k[0], n = parseInt(k.slice(1), 10);
                    const p = side === "a" ? r.kpts_a[n] : r.kpts_b[n];
                    const box = side === "a" ? boxA : boxB;
                    const sx = side === "a" ? sxA : sxB, sy = side === "a" ? syA : syB;
                    html += `<circle cx="${box.x + p[0] * sx}" cy="${box.y + p[1] * sy}" r="2" fill="#fff" opacity="0.7"/>`;
                });
            }
            overlay.innerHTML = html;
            hiddenCount.textContent = state.hidden.size;
            showHidden.style.display = state.hidden.size ? "" : "none";
        }

        overlay.addEventListener("click", (e) => {
            const line = e.target.closest("line[data-idx]");
            if (!line || !state.result) return;
            const idx = parseInt(line.dataset.idx, 10);
            if (e.shiftKey) {
                const thisScore = state.result.matches[idx][2];
                state.result.matches.forEach((m, i) => {
                    if (m[2] <= thisScore) state.hidden.add(i);
                });
            } else {
                state.hidden.add(idx);
            }
            render();
        });

        showHidden.addEventListener("click", () => { state.hidden.clear(); render(); });
        conf.addEventListener("input", () => { confVal.textContent = parseFloat(conf.value).toFixed(2); render(); });
        linesCk.addEventListener("change", render);
        kptsCk.addEventListener("change", render);

        function setStats(r) {
            stats.textContent = `${r.n_matches} matches @≥0.5 · mean ${(r.mean_score ?? 0).toFixed(2)}`;
        }

        function onResult(r) {
            state.result = r;
            state.hidden.clear();
            setStats(r);
            runBtn.style.display = "none";
            rerunBtn.style.display = "";
            render();
        }

        async function runMatch(overwrite) {
            status.textContent = "…matching";
            runBtn.disabled = rerunBtn.disabled = true;
            try {
                const q = overwrite ? "?overwrite=1" : "";
                const r = await fetch(`/api/pairs/${queueId}/local-match${q}`, { method: "POST" });
                if (!r.ok) {
                    status.textContent = `error (${r.status})`;
                    return;
                }
                status.textContent = "";
                onResult(await r.json());
            } finally {
                runBtn.disabled = rerunBtn.disabled = false;
            }
        }

        runBtn.addEventListener("click", () => runMatch(false));
        rerunBtn.addEventListener("click", () => runMatch(true));

        const ro = new ResizeObserver(render);
        ro.observe(card);

        // Hybrid caching: fetch cached result on mount.
        fetch(`/api/pairs/${queueId}/local-match?extractor=${extractor}`).then(async (r) => {
            if (r.status === 200) {
                onResult(await r.json());
            } else {
                runBtn.style.display = "";
            }
        });
    }

    function initAll(root) {
        (root || document).querySelectorAll(".card[data-queue-id]").forEach(init);
    }

    document.addEventListener("DOMContentLoaded", () => initAll());
    document.body.addEventListener("htmx:afterSwap", (e) => initAll(e.target));
})();
```

- [ ] **Step 6: Run template-hook tests**

Run: `pytest tests/web/test_local_match_frontend_hooks.py -v`
Expected: 2 PASS.

- [ ] **Step 7: Smoke-run the existing pair carousel tests to confirm no regressions**

Run: `pytest tests/web/test_pair_carousel.py tests/web/test_htmx_fragments.py -v`
Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/whaleshark_reid/web/static/js/local_match.js src/whaleshark_reid/web/static/css/app.css src/whaleshark_reid/web/templates/partials/pair_card.html src/whaleshark_reid/web/templates/base.html tests/web/test_local_match_frontend_hooks.py
git commit -m "Add interactive local-match SVG overlay with slider, toggles, click-to-hide"
```

---

## Task 8: CLI `match-local` bulk precompute

**Files:**
- Create: `src/whaleshark_reid/cli/commands/match_local.py`
- Modify: `src/whaleshark_reid/cli/main.py`
- Create: `tests/cli/test_cli_match_local.py`

- [ ] **Step 1: Write failing tests**

Create `tests/cli/test_cli_match_local.py`:

```python
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.cli.main import app
from whaleshark_reid.core.match import lightglue as lg_module
from whaleshark_reid.core.match.lightglue import MatchResult
from whaleshark_reid.storage.db import Storage


def _seed(s: Storage):
    s.conn.execute("INSERT INTO runs(run_id, stage, config_json, started_at, status) VALUES('r','match','{}', '2026-04-13', 'ok')")
    s.conn.execute("INSERT INTO annotations(annotation_uuid, file_path, bbox_json, theta) VALUES('a','/x/a.jpg','[0,0,10,10]',0),('b','/x/b.jpg','[0,0,10,10]',0),('c','/x/c.jpg','[0,0,10,10]',0)")
    s.conn.execute("INSERT INTO pair_queue(run_id, ann_a_uuid, ann_b_uuid, distance, position) VALUES('r','a','b',0.1,0),('r','a','c',0.2,1)")


def _fake_matcher(monkeypatch):
    class FM:
        extractor_name = "aliked"
        def _extract(self, path):
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
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/cli/test_cli_match_local.py -v`
Expected: FAIL — command not registered.

- [ ] **Step 3: Implement command**

Create `src/whaleshark_reid/cli/commands/match_local.py`:

```python
"""CLI command: bulk-precompute local feature matches for every pair in a queue.

Feature extraction is deduped per unique annotation UUID; each pair then
reuses cached features.
"""
from __future__ import annotations

from pathlib import Path

import typer

from whaleshark_reid.core.match import lightglue as lg
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.services.local_match import read_cached, write_cached


def match_local_command(
    run_id: str = typer.Option(..., "--run-id", help="pair_queue run_id to process"),
    db_path: Path = typer.Option(..., "--db-path", help="path to catalog SQLite DB"),
    extractor: str = typer.Option("aliked", "--extractor"),
    limit: int = typer.Option(0, "--limit", help="0 = all pairs"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    storage = Storage(db_path)
    storage.init_schema()

    rows = storage.conn.execute(
        """
        SELECT pq.queue_id, a.file_path AS pa, b.file_path AS pb,
               pq.ann_a_uuid, pq.ann_b_uuid
        FROM pair_queue pq
        JOIN annotations a ON a.annotation_uuid = pq.ann_a_uuid
        JOIN annotations b ON b.annotation_uuid = pq.ann_b_uuid
        WHERE pq.run_id = ?
        ORDER BY pq.position
        """, (run_id,),
    ).fetchall()

    if limit > 0:
        rows = rows[:limit]

    # Skip already-cached pairs unless --overwrite
    if not overwrite:
        rows = [r for r in rows if read_cached(storage, r["queue_id"], extractor) is None]

    if not rows:
        typer.echo("nothing to do")
        return

    # Dedup annotation paths (annotation UUID as identity; same UUID => same file)
    path_by_uuid: dict[str, str] = {}
    for r in rows:
        path_by_uuid[r["ann_a_uuid"]] = r["pa"]
        path_by_uuid[r["ann_b_uuid"]] = r["pb"]

    typer.echo(f"extracting features for {len(path_by_uuid)} unique annotations")
    feats_by_path = lg.extract_features_batch(
        list(path_by_uuid.values()), extractor=extractor,
    )

    typer.echo(f"matching {len(rows)} pairs")
    pair_paths = [(r["pa"], r["pb"]) for r in rows]
    results = lg.match_pairs_batch(pair_paths, feats_by_path, extractor=extractor)

    with storage.transaction():
        for r, res in zip(rows, results):
            write_cached(storage, r["queue_id"], res)
    typer.echo(f"wrote {len(results)} rows to pair_matches")
```

- [ ] **Step 4: Register in `cli/main.py`**

Add import and registration:

```python
from whaleshark_reid.cli.commands.match_local import match_local_command
# ...
app.command(name="match-local")(match_local_command)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/cli/test_cli_match_local.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add src/whaleshark_reid/cli/commands/match_local.py src/whaleshark_reid/cli/main.py tests/cli/test_cli_match_local.py
git commit -m "Add catalog match-local CLI for bulk precompute with feature dedup"
```

---

## Task 9: `pyproject.toml` optional extra + pytest marker

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add optional extra**

In `pyproject.toml`, under `[project.optional-dependencies]`:

```toml
local-match = [
  "lightglue @ git+https://github.com/cvg/LightGlue.git",
]
```

(torch is already a top-level dependency.)

- [ ] **Step 2: Register `lightglue` marker to silence pytest warning**

Under `[tool.pytest.ini_options]` (add the section if missing):

```toml
[tool.pytest.ini_options]
markers = [
    "lightglue: tests that require the real LightGlue install (skipped by default)",
]
addopts = "-m 'not lightglue'"
```

- [ ] **Step 3: Run the full suite**

Run: `pytest -q`
Expected: all new tests pass, no regressions.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "Register local-match optional extra and pytest marker"
```

---

## Task 10: Manual smoke + commit docs

**Files:** none new; this task is a sanity check.

- [ ] **Step 1: Install optional extra in the dev env**

```bash
pip install -e '.[local-match]'
python -c "from lightglue import ALIKED; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 2: Run the dev server and test manually**

```bash
uvicorn whaleshark_reid.web.app:create_app --factory --reload
```

In the browser:
1. Navigate to `/review/pairs/<run-id>`.
2. Verify "Run local match" button appears on a fresh pair.
3. Click it — spinner shows, then overlay appears with colored lines.
4. Drag the confidence slider — lines filter instantly.
5. Click a line — it fades. Shift-click another — all below it fade. Click "Show hidden (N)" — they return.
6. Toggle `lines` / `keypoints` checkboxes.
7. Submit a decision (match/skip), advance to the next pair — ensure no console errors.
8. Come back to the first pair (decisions log → annotation link → back) — overlay should auto-render from cache, no button press needed.

- [ ] **Step 3: Bulk precompute sanity check**

```bash
catalog match-local --run-id <run> --db-path /workspace/cache/state.db --limit 5
```

Expected: "extracting features for N unique annotations", "matching M pairs", "wrote M rows to pair_matches". Second run with same args prints "nothing to do".

- [ ] **Step 4: Final commit (if any manual tweaks needed)**

```bash
git add -u
git commit -m "Polish local-match UI after manual smoke" || echo "nothing to polish"
```

---

## Self-Review Pass

- **Spec coverage:** new module, pair_matches schema+migration, JSON shape, GET/POST endpoints, frontend SVG overlay with slider/toggles/click-to-hide/shift-click, hybrid caching via GET-then-auto-render, bulk CLI with per-annotation dedup, optional extra, LightGlue-gated tests — all present.
- **Placeholder scan:** no TBD/TODO; every code step has full code. Template edit in Task 7 references `... unchanged ...` for the existing meta-panel, which is intentional and clear.
- **Type consistency:** `MatchResult` field names consistent across tests, service, routes, JS. `get_matcher(extractor=...)` signature stable. `_build_matcher` is the seam both batch and singleton go through.
- **Ambiguity:** `img_a_size` / `img_b_size` are the *match-space* coord sizes (where kpts live), not rendered chip sizes — documented both in the spec and schema comment.
