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
