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
