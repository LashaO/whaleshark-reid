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
    results, metrics = run_dbscan(mat, uuids, eps=0.3, min_samples=2, metric="euclidean", standardize=True)

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
