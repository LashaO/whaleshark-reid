"""Tests for parquet-backed cluster label cache."""
from __future__ import annotations

from pathlib import Path

from whaleshark_reid.storage.cluster_cache import read_clusters, write_clusters


def test_write_and_read_clusters(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": "a",
            "cluster_label": 0,
            "cluster_algo": "dbscan",
            "cluster_params_json": '{"eps": 0.7}',
        },
        {
            "annotation_uuid": "b",
            "cluster_label": -1,
            "cluster_algo": "dbscan",
            "cluster_params_json": '{"eps": 0.7}',
        },
    ]
    write_clusters(tmp_cache_dir, "run_abc", rows)

    df = read_clusters(tmp_cache_dir, "run_abc")
    assert len(df) == 2
    assert set(df["cluster_label"]) == {0, -1}
