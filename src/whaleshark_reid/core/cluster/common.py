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


def run_cluster_stage(
    cache_dir: Path,
    embedding_run_id: str,
    cluster_run_id: str,
    algo: str = "dbscan",
    params: dict | None = None,
) -> ClusterStageResult:
    """Load embeddings parquet → run DBSCAN or HDBSCAN → write cluster parquet → return stage result."""
    import json

    from whaleshark_reid.storage.cluster_cache import write_clusters
    from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array

    params = params or {}
    uuids, mat = read_embeddings_as_array(cache_dir, embedding_run_id)

    if algo == "dbscan":
        from whaleshark_reid.core.cluster.dbscan import run_dbscan

        results, metrics = run_dbscan(
            mat, uuids,
            eps=params.get("eps", 0.7),
            min_samples=params.get("min_samples", 2),
            metric=params.get("metric", "cosine"),
            standardize=params.get("standardize", True),
        )
    elif algo == "hdbscan":
        from whaleshark_reid.core.cluster.hdbscan import run_hdbscan

        results, metrics = run_hdbscan(
            mat, uuids,
            min_cluster_size=params.get("min_cluster_size", 3),
            min_samples=params.get("min_samples"),
            metric=params.get("metric", "euclidean"),
        )
    else:
        raise ValueError(f"Unknown cluster algo: {algo}")

    rows = [
        {
            "annotation_uuid": r.annotation_uuid,
            "cluster_label": int(r.cluster_label),
            "cluster_algo": r.cluster_algo,
            "cluster_params_json": json.dumps(r.cluster_params, default=str),
        }
        for r in results
    ]
    write_clusters(cache_dir, cluster_run_id, rows)

    return ClusterStageResult(
        algo=metrics["algo"],
        n_clusters=metrics["n_clusters"],
        n_noise=metrics["n_noise"],
        largest_cluster_size=metrics["largest_cluster_size"],
        singleton_fraction=metrics["singleton_fraction"],
        median_cluster_size=metrics["median_cluster_size"],
    )
