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
