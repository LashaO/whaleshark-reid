"""Distance distribution + cluster quality + queue priority metrics."""
from __future__ import annotations

import numpy as np


def distance_distribution_stats(distmat: np.ndarray) -> dict:
    """Compute descriptive stats on the upper triangle of a square distance matrix.

    Returns dict with n, mean, std, median, percentiles, histogram (20 bins, [0, 2]).
    """
    n = distmat.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    d = distmat[iu, ju]
    hist, edges = np.histogram(d, bins=20, range=(0.0, 2.0))
    return {
        "n": int(len(d)),
        "mean": float(d.mean()) if len(d) else 0.0,
        "std": float(d.std()) if len(d) else 0.0,
        "median": float(np.median(d)) if len(d) else 0.0,
        "p10": float(np.percentile(d, 10)) if len(d) else 0.0,
        "p25": float(np.percentile(d, 25)) if len(d) else 0.0,
        "p50": float(np.percentile(d, 50)) if len(d) else 0.0,
        "p75": float(np.percentile(d, 75)) if len(d) else 0.0,
        "p90": float(np.percentile(d, 90)) if len(d) else 0.0,
        "histogram": hist.tolist(),
        "histogram_edges": edges.tolist(),
    }


def cluster_quality_stats(embeddings: np.ndarray, cluster_labels: np.ndarray) -> dict:
    """Compute silhouette score (if possible), noise fraction, cluster size histogram."""
    from collections import Counter

    labels = np.asarray(cluster_labels)
    n = len(labels)
    noise = int((labels == -1).sum())
    non_noise_mask = labels != -1
    n_unique_non_noise = len(set(labels[non_noise_mask].tolist()))

    silhouette: float | None = None
    if n_unique_non_noise > 1 and non_noise_mask.sum() > 1:
        from sklearn.metrics import silhouette_score
        try:
            silhouette = float(silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask]))
        except Exception:
            silhouette = None

    sizes = list(Counter(labels[non_noise_mask].tolist()).values())

    return {
        "silhouette_score": silhouette,
        "noise_fraction": noise / n if n else 0.0,
        "n_clusters": n_unique_non_noise,
        "cluster_size_histogram": sizes,
        "median_cluster_size": float(np.median(sizes)) if sizes else 0.0,
    }


def queue_priority_stats(pair_queue) -> dict:
    """Summary stats over a list of PairCandidate."""
    if not pair_queue:
        return {
            "n_pairs": 0,
            "fraction_same_cluster": 0.0,
            "median_distance": 0.0,
        }
    distances = np.array([p.distance for p in pair_queue], dtype=np.float64)
    same = sum(1 for p in pair_queue if p.same_cluster)
    return {
        "n_pairs": len(pair_queue),
        "fraction_same_cluster": same / len(pair_queue),
        "median_distance": float(np.median(distances)),
    }
