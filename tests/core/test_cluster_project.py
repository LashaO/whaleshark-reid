"""Tests for UMAP 2D projection wrapper."""
from __future__ import annotations

import numpy as np

from whaleshark_reid.core.cluster.project import run_umap


def test_run_umap_returns_points():
    rng = np.random.default_rng(0)
    mat = rng.normal(size=(20, 8)).astype(np.float32)
    uuids = [f"u{i}" for i in range(20)]

    points = run_umap(mat, uuids, n_neighbors=5, min_dist=0.1, random_state=42)

    assert len(points) == 20
    for p in points:
        assert p.algo == "umap"
        assert isinstance(p.x, float)
        assert isinstance(p.y, float)


def test_run_project_stage_writes_parseable_params_json(tmp_cache_dir, tmp_path):
    """Verify that run_project_stage writes params_json as valid JSON."""
    import json as json_lib

    import numpy as np

    from whaleshark_reid.core.cluster.project import run_project_stage
    from whaleshark_reid.storage.embedding_cache import write_embeddings
    from whaleshark_reid.storage.projection_cache import read_projections

    rng = np.random.default_rng(0)
    mat = rng.normal(size=(20, 8)).astype(np.float32)
    rows = [
        {
            "annotation_uuid": f"u{i}",
            "embedding": mat[i].tolist(),
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        }
        for i in range(20)
    ]
    write_embeddings(tmp_cache_dir, "r_emb", rows)

    run_project_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="r_emb",
        projection_run_id="r_proj",
        n_neighbors=5,
    )

    df = read_projections(tmp_cache_dir, "r_proj")
    assert len(df) == 20
    # The fix: params_json must be parseable JSON
    parsed = json_lib.loads(df.iloc[0]["params_json"])
    assert parsed["n_neighbors"] == 5
    assert "min_dist" in parsed
