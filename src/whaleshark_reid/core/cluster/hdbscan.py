"""HDBSCAN clustering wrapper (alternative to DBSCAN).

HDBSCAN does not require eps; instead it uses a minimum cluster size and picks
clusters at multiple density levels. Works well when cluster densities vary.
"""
from __future__ import annotations

import numpy as np

from whaleshark_reid.core.cluster.common import cluster_metrics, labels_to_results
from whaleshark_reid.core.schema import ClusterResult


def run_hdbscan(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    min_cluster_size: int = 3,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> tuple[list[ClusterResult], dict]:
    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
    )
    labels = clusterer.fit_predict(embeddings)

    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "metric": metric,
    }
    results = labels_to_results(annotation_uuids, labels, algo="hdbscan", params=params)
    metrics = cluster_metrics(labels)
    metrics["algo"] = "hdbscan"
    return results, metrics
