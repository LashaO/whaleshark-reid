"""Tests for the cluster stage entry point — reads embeddings, writes clusters parquet."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.cluster.common import run_cluster_stage
from whaleshark_reid.storage.cluster_cache import read_clusters
from whaleshark_reid.storage.embedding_cache import write_embeddings


def test_run_cluster_stage_writes_parquet_and_returns_metrics(tmp_cache_dir: Path):
    rng = np.random.default_rng(42)
    mat = np.vstack([
        rng.normal(loc=10.0, scale=0.1, size=(5, 8)),
        rng.normal(loc=-10.0, scale=0.1, size=(5, 8)),
    ]).astype(np.float32)

    rows = [
        {
            "annotation_uuid": f"u{i}",
            "embedding": mat[i].tolist(),
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        }
        for i in range(10)
    ]
    write_embeddings(tmp_cache_dir, "run_abc", rows)

    result = run_cluster_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="run_abc",
        cluster_run_id="run_xyz",
        algo="dbscan",
        params={"eps": 0.7, "min_samples": 2, "metric": "cosine", "standardize": True},
    )

    assert result.algo == "dbscan"
    assert result.n_clusters == 2

    df = read_clusters(tmp_cache_dir, "run_xyz")
    assert len(df) == 10
    assert set(df["cluster_algo"]) == {"dbscan"}
