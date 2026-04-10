"""DBSCAN clustering wrapper.

Matches extract_and_evaluate_whalesharks.ipynb exactly:
  StandardScaler().fit_transform(embeddings) → DBSCAN(eps=0.7, min_samples=2, metric='cosine')
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from whaleshark_reid.core.cluster.common import cluster_metrics, labels_to_results
from whaleshark_reid.core.schema import ClusterResult


def run_dbscan(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    eps: float = 0.7,
    min_samples: int = 2,
    metric: str = "cosine",
    standardize: bool = True,
) -> tuple[list[ClusterResult], dict]:
    data = StandardScaler().fit_transform(embeddings) if standardize else embeddings
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
    labels = model.labels_

    params = {
        "eps": eps,
        "min_samples": min_samples,
        "metric": metric,
        "standardize": standardize,
    }
    results = labels_to_results(annotation_uuids, labels, algo="dbscan", params=params)
    metrics = cluster_metrics(labels)
    metrics["algo"] = "dbscan"
    return results, metrics
