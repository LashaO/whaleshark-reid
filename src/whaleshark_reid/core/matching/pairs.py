"""Pair candidate generation + filtering + cluster annotation + matching stage entry point."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.schema import MatchingResult, PairCandidate


def pairs_below_threshold(
    distmat: np.ndarray,
    annotation_uuids: list[str],
    threshold: float,
) -> list[PairCandidate]:
    """Return PairCandidates for every (i, j) with i < j and distmat[i,j] < threshold.
    Sorted ascending by distance.
    """
    n = len(annotation_uuids)
    assert distmat.shape == (n, n)
    iu, ju = np.triu_indices(n, k=1)
    mask = distmat[iu, ju] <= threshold
    iu, ju = iu[mask], ju[mask]

    pairs = [
        PairCandidate(
            ann_a_uuid=annotation_uuids[i],
            ann_b_uuid=annotation_uuids[j],
            distance=float(distmat[i, j]),
        )
        for i, j in zip(iu, ju)
    ]
    pairs.sort(key=lambda p: p.distance)
    return pairs


def filter_by_decisions(
    candidates: list[PairCandidate],
    active_decisions: list[tuple[str, str, str]],
) -> list[PairCandidate]:
    """Drop candidates whose (a, b) or (b, a) already has a match/no_match decision.
    skip and unsure decisions do NOT filter — those pairs stay in the queue.
    """
    blocked: set[frozenset[str]] = set()
    for a, b, decision in active_decisions:
        if decision in ("match", "no_match"):
            blocked.add(frozenset((a, b)))
    return [p for p in candidates if frozenset((p.ann_a_uuid, p.ann_b_uuid)) not in blocked]


def annotate_with_clusters(
    candidates: list[PairCandidate],
    cluster_by_uuid: dict[str, int],
) -> list[PairCandidate]:
    """Fill in cluster_a, cluster_b, same_cluster for each candidate."""
    out = []
    for p in candidates:
        ca = cluster_by_uuid.get(p.ann_a_uuid)
        cb = cluster_by_uuid.get(p.ann_b_uuid)
        same = ca is not None and cb is not None and ca == cb and ca != -1
        out.append(p.model_copy(update={"cluster_a": ca, "cluster_b": cb, "same_cluster": same}))
    return out


def _distance_percentiles(distances: np.ndarray) -> dict:
    if len(distances) == 0:
        return {"min": 0.0, "max": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "min": float(distances.min()),
        "max": float(distances.max()),
        "median": float(np.median(distances)),
        "p10": float(np.percentile(distances, 10)),
        "p25": float(np.percentile(distances, 25)),
        "p50": float(np.percentile(distances, 50)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
    }


def run_matching_stage(
    storage,
    cache_dir: Path,
    matching_run_id: str,
    embedding_run_id: str,
    cluster_run_id: str,
    distance_threshold: float = 1.0,
    max_queue_size: int = 2000,
) -> MatchingResult:
    """Compute pairwise distances, filter, annotate with clusters, write pair_queue."""
    from wbia_miew_id.metrics.distance import compute_distance_matrix

    from whaleshark_reid.storage.cluster_cache import read_clusters
    from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array

    uuids, mat = read_embeddings_as_array(cache_dir, embedding_run_id)
    cluster_df = read_clusters(cache_dir, cluster_run_id)
    cluster_by_uuid = dict(zip(cluster_df["annotation_uuid"], cluster_df["cluster_label"].astype(int)))

    distmat_tensor = compute_distance_matrix(mat, mat, metric="cosine")
    distmat = distmat_tensor.numpy() if hasattr(distmat_tensor, "numpy") else np.asarray(distmat_tensor)

    candidates = pairs_below_threshold(distmat, uuids, threshold=distance_threshold)

    # Pull active pair decisions
    rows = storage.conn.execute(
        """
        SELECT ann_a_uuid, ann_b_uuid, decision FROM pair_decisions
        WHERE superseded_by IS NULL
        """
    ).fetchall()
    active = [(r["ann_a_uuid"], r["ann_b_uuid"], r["decision"]) for r in rows]
    n_before_filter = len(candidates)
    candidates = filter_by_decisions(candidates, active)
    n_filtered = n_before_filter - len(candidates)

    candidates = annotate_with_clusters(candidates, cluster_by_uuid)
    candidates = candidates[:max_queue_size]

    # Replace pair_queue rows for this matching run
    storage.conn.execute("DELETE FROM pair_queue WHERE run_id = ?", (matching_run_id,))
    for position, p in enumerate(candidates):
        storage.conn.execute(
            """
            INSERT INTO pair_queue (
                run_id, ann_a_uuid, ann_b_uuid, distance,
                cluster_a, cluster_b, same_cluster, position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                matching_run_id,
                p.ann_a_uuid,
                p.ann_b_uuid,
                p.distance,
                p.cluster_a,
                p.cluster_b,
                1 if p.same_cluster else 0,
                position,
            ),
        )

    dists = np.array([p.distance for p in candidates], dtype=np.float64)
    pctiles = _distance_percentiles(dists)
    n_same = sum(1 for p in candidates if p.same_cluster)
    n_cross = len(candidates) - n_same

    return MatchingResult(
        n_pairs=len(candidates),
        n_same_cluster=n_same,
        n_cross_cluster=n_cross,
        n_filtered_out_by_decisions=n_filtered,
        median_distance=pctiles["median"],
        min_distance=pctiles["min"],
        max_distance=pctiles["max"],
        p10=pctiles["p10"],
        p25=pctiles["p25"],
        p50=pctiles["p50"],
        p75=pctiles["p75"],
        p90=pctiles["p90"],
    )
