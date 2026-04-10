"""End-to-end pipeline test: ingest → embed (stubbed) → cluster → match → project → rebuild.

Uses a stubbed MiewID so the test runs on CPU in <5 seconds without hitting HuggingFace.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from whaleshark_reid.core.cluster.common import run_cluster_stage
from whaleshark_reid.core.cluster.project import run_project_stage
from whaleshark_reid.core.embed.miewid import run_embed_stage
from whaleshark_reid.core.feedback.unionfind import rebuild_individuals_cache
from whaleshark_reid.core.ingest.inat import ingest_inat_csv
from whaleshark_reid.core.matching.pairs import run_matching_stage
from whaleshark_reid.core.schema import inat_annotation_uuid
from whaleshark_reid.storage.cluster_cache import read_clusters
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.embedding_cache import read_embeddings
from whaleshark_reid.storage.projection_cache import read_projections

FIXTURES = Path(__file__).parent.parent / "fixtures"


class _StubMiewId(torch.nn.Module):
    """Same stub as test_embed_miewid — deterministic embed_dim=8."""
    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        per_sample = x.mean(dim=(1, 2, 3))
        return per_sample.unsqueeze(1).expand(-1, self.embed_dim)

    def forward(self, x, label=None):
        return self.extract_feat(x)


@pytest.fixture
def stub_miewid(monkeypatch):
    from transformers import AutoModel

    monkeypatch.setattr(
        AutoModel, "from_pretrained",
        lambda *a, **k: _StubMiewId(),
    )
    yield


def test_full_pipeline_on_mini_fixture(
    stub_miewid,
    tmp_db_path: Path,
    tmp_cache_dir: Path,
):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    # --- Stage 1: ingest ---
    ingest_result = ingest_inat_csv(
        csv_path=FIXTURES / "mini_inat.csv",
        photos_dir=FIXTURES / "photos",
        storage=storage,
        run_id="r_ingest",
        rich_csv_path=FIXTURES / "mini_inat_rich.csv",
    )
    assert ingest_result.n_ingested == 10

    # --- Stage 2: embed ---
    storage.begin_run(run_id="r_embed", stage="embed", config={})
    embed_result = run_embed_stage(
        storage=storage,
        cache_dir=tmp_cache_dir,
        run_id="r_embed",
        batch_size=4,
        num_workers=0,
        device="cpu",
    )
    assert embed_result.n_embedded == 10
    assert embed_result.embed_dim == 8
    storage.finish_run("r_embed", "ok", metrics=embed_result.model_dump())

    df_emb = read_embeddings(tmp_cache_dir, "r_embed")
    assert len(df_emb) == 10

    # --- Stage 3: cluster ---
    storage.begin_run(run_id="r_cluster", stage="cluster", config={})
    cluster_result = run_cluster_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="r_embed",
        cluster_run_id="r_cluster",
        algo="dbscan",
        params={"eps": 0.7, "min_samples": 2, "metric": "cosine", "standardize": True},
    )
    storage.finish_run("r_cluster", "ok", metrics=cluster_result.model_dump())

    df_clu = read_clusters(tmp_cache_dir, "r_cluster")
    assert len(df_clu) == 10

    # --- Stage 4: matching ---
    storage.begin_run(run_id="r_match", stage="matching", config={})
    match_result = run_matching_stage(
        storage=storage,
        cache_dir=tmp_cache_dir,
        matching_run_id="r_match",
        embedding_run_id="r_embed",
        cluster_run_id="r_cluster",
        distance_threshold=2.0,  # loose, include everything
        max_queue_size=1000,
    )
    storage.finish_run("r_match", "ok", metrics=match_result.model_dump())
    assert storage.count("pair_queue", run_id="r_match") > 0

    # --- Stage 5: project (UMAP) ---
    storage.begin_run(run_id="r_project", stage="project", config={})
    project_result = run_project_stage(
        cache_dir=tmp_cache_dir,
        embedding_run_id="r_embed",
        projection_run_id="r_project",
        n_neighbors=5,
    )
    storage.finish_run("r_project", "ok", metrics=project_result.model_dump())

    df_proj = read_projections(tmp_cache_dir, "r_project")
    assert len(df_proj) == 10

    # --- Stage 6: seed one pair decision, rebuild individuals ---
    u0 = inat_annotation_uuid(100, 0)
    u1 = inat_annotation_uuid(101, 0)
    storage.conn.execute(
        "INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, created_at) "
        "VALUES (?, ?, 'match', ?)",
        (u0, u1, datetime.now(timezone.utc).isoformat()),
    )
    rebuild_result = rebuild_individuals_cache(storage)
    assert rebuild_result.n_components == 1

    a = storage.get_annotation(u0)
    b = storage.get_annotation(u1)
    assert a.name_uuid is not None
    assert a.name_uuid == b.name_uuid
