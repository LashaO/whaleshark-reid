"""Tests for parquet-backed UMAP projection cache."""
from __future__ import annotations

from pathlib import Path

from whaleshark_reid.storage.projection_cache import read_projections, write_projections


def test_write_and_read_projections(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": "a",
            "x": 1.2,
            "y": -3.4,
            "algo": "umap",
            "params_json": '{"n_neighbors": 15}',
        },
    ]
    write_projections(tmp_cache_dir, "run_abc", rows)

    df = read_projections(tmp_cache_dir, "run_abc")
    assert len(df) == 1
    assert df.iloc[0]["x"] == 1.2
