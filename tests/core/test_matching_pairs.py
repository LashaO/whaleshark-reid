"""Tests for matching/pairs.py — pair candidate generation, filtering, annotation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.matching.pairs import (
    annotate_with_clusters,
    filter_by_decisions,
    pairs_below_threshold,
)
from whaleshark_reid.core.schema import PairCandidate


def _distmat_4x4() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1, 0.5, 0.9],
            [0.1, 0.0, 0.4, 0.8],
            [0.5, 0.4, 0.0, 0.3],
            [0.9, 0.8, 0.3, 0.0],
        ]
    )


def test_pairs_below_threshold_returns_upper_triangle_only():
    distmat = _distmat_4x4()
    uuids = ["a", "b", "c", "d"]
    pairs = pairs_below_threshold(distmat, uuids, threshold=0.6)

    # (a,b)=0.1, (a,c)=0.5, (b,c)=0.4, (c,d)=0.3
    # Upper triangle only, no (b,a) duplicate, no self-pair
    dists = sorted(p.distance for p in pairs)
    assert dists == [0.1, 0.3, 0.4, 0.5]
    assert all(p.ann_a_uuid != p.ann_b_uuid for p in pairs)


def test_pairs_below_threshold_sorted_ascending():
    distmat = _distmat_4x4()
    uuids = ["a", "b", "c", "d"]
    pairs = pairs_below_threshold(distmat, uuids, threshold=1.0)
    dists = [p.distance for p in pairs]
    assert dists == sorted(dists)


def test_filter_by_decisions_drops_match_and_no_match_but_keeps_skip():
    pairs = [
        PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1),
        PairCandidate(ann_a_uuid="a", ann_b_uuid="c", distance=0.5),
        PairCandidate(ann_a_uuid="c", ann_b_uuid="d", distance=0.3),
    ]
    decisions = [
        ("a", "b", "match"),
        ("a", "c", "no_match"),
        ("c", "d", "skip"),  # skip does NOT filter — still in queue
    ]
    filtered = filter_by_decisions(pairs, decisions)
    kept_pairs = {(p.ann_a_uuid, p.ann_b_uuid) for p in filtered}
    assert kept_pairs == {("c", "d")}


def test_filter_by_decisions_is_order_insensitive():
    pairs = [PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1)]
    decisions = [("b", "a", "match")]  # flipped order
    filtered = filter_by_decisions(pairs, decisions)
    assert filtered == []


def test_annotate_with_clusters():
    pairs = [
        PairCandidate(ann_a_uuid="a", ann_b_uuid="b", distance=0.1),
        PairCandidate(ann_a_uuid="a", ann_b_uuid="c", distance=0.5),
    ]
    cluster_by_uuid = {"a": 3, "b": 3, "c": 7}
    annotated = annotate_with_clusters(pairs, cluster_by_uuid)

    assert annotated[0].cluster_a == 3
    assert annotated[0].cluster_b == 3
    assert annotated[0].same_cluster is True

    assert annotated[1].cluster_a == 3
    assert annotated[1].cluster_b == 7
    assert annotated[1].same_cluster is False


# --- run_matching_stage end-to-end ---

def test_run_matching_stage_writes_pair_queue(tmp_cache_dir: Path, tmp_db_path: Path):
    from whaleshark_reid.core.matching.pairs import run_matching_stage
    from whaleshark_reid.storage.cluster_cache import write_clusters
    from whaleshark_reid.storage.db import Storage
    from whaleshark_reid.storage.embedding_cache import write_embeddings

    storage = Storage(tmp_db_path)
    storage.init_schema()

    # Seed 4 annotations so foreign key constraints are satisfied
    from whaleshark_reid.core.schema import (
        Annotation,
        inat_annotation_uuid,
        inat_image_uuid,
    )
    for i in range(4):
        ann = Annotation(
            annotation_uuid=inat_annotation_uuid(100 + i, 0),
            image_uuid=inat_image_uuid(100 + i, 0),
            source="inat",
            observation_id=100 + i,
            photo_index=0,
            file_path=f"/tmp/{100+i}.jpg",
            file_name=f"{100+i}.jpg",
            bbox=[0, 0, 10, 10],
        )
        storage.upsert_annotation(ann, run_id="r_ingest")
    uuids = [inat_annotation_uuid(100 + i, 0) for i in range(4)]

    # Seed a run so the pair_queue foreign key is satisfied
    storage.begin_run(run_id="r_match", stage="matching", config={})

    # Fake embeddings: two tight pairs
    mat = np.array(
        [
            [1.0, 0.0],
            [0.95, 0.05],
            [-1.0, 0.0],
            [-0.95, -0.05],
        ],
        dtype=np.float32,
    )
    emb_rows = [
        {"annotation_uuid": uuids[i], "embedding": mat[i].tolist(),
         "model_id": "m", "model_version": "v", "created_at": "t"}
        for i in range(4)
    ]
    write_embeddings(tmp_cache_dir, "r_embed", emb_rows)

    # Fake clusters
    cluster_rows = [
        {"annotation_uuid": uuids[i], "cluster_label": 0 if i < 2 else 1,
         "cluster_algo": "dbscan", "cluster_params_json": "{}"}
        for i in range(4)
    ]
    write_clusters(tmp_cache_dir, "r_cluster", cluster_rows)

    result = run_matching_stage(
        storage=storage,
        cache_dir=tmp_cache_dir,
        matching_run_id="r_match",
        embedding_run_id="r_embed",
        cluster_run_id="r_cluster",
        distance_threshold=2.0,   # loose, keep everything
        max_queue_size=1000,
    )

    # 6 upper-triangle pairs below threshold
    assert result.n_pairs == 6
    assert storage.count("pair_queue", run_id="r_match") == 6

    # Cross-cluster pairs: (0,2),(0,3),(1,2),(1,3) = 4; same-cluster: (0,1),(2,3) = 2
    assert result.n_same_cluster == 2
    assert result.n_cross_cluster == 4
