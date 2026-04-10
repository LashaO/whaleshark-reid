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
    labels = [r.cluster_label for r in results]
    assert all(isinstance(l, int) for l in labels)
    assert "n_clusters" in metrics
