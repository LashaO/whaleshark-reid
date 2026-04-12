"""Shared fixtures for web tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from starlette.testclient import TestClient

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import create_app
from whaleshark_reid.web.dependencies import override_settings, override_storage
from whaleshark_reid.web.settings import Settings


class _StubMiewId(torch.nn.Module):
    """Deterministic stand-in for MiewIdNet so web tests don't hit HuggingFace."""

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

    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **k: _StubMiewId())
    yield


@pytest.fixture
def web_client(tmp_db_path: Path, tmp_cache_dir: Path) -> TestClient:
    """TestClient backed by a fresh tmp DB."""
    storage = Storage(tmp_db_path)
    storage.init_schema()
    override_storage(storage)
    override_settings(Settings(db_path=tmp_db_path, cache_dir=tmp_cache_dir))
    app = create_app()
    return TestClient(app)


@pytest.fixture
def seeded_web_client(tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid) -> TestClient:
    """TestClient with 10 ingested + embedded + clustered + matched annotations."""
    from whaleshark_reid.core.cluster.common import run_cluster_stage
    from whaleshark_reid.core.embed.miewid import run_embed_stage
    from whaleshark_reid.core.ingest.inat import ingest_inat_csv
    from whaleshark_reid.core.matching.pairs import run_matching_stage
    from whaleshark_reid.core.cluster.project import run_project_stage

    storage = Storage(tmp_db_path)
    storage.init_schema()

    fixtures = Path(__file__).parent.parent / "fixtures"
    ingest_inat_csv(
        csv_path=fixtures / "mini_inat.csv",
        photos_dir=fixtures / "photos",
        storage=storage,
        run_id="r_ingest",
    )

    storage.begin_run("r_embed", "embed", config={})
    embed_result = run_embed_stage(
        storage=storage, cache_dir=tmp_cache_dir, run_id="r_embed",
        batch_size=4, num_workers=0, device="cpu",
    )
    storage.finish_run("r_embed", "ok", metrics=embed_result.model_dump())

    storage.begin_run("r_cluster", "cluster", config={})
    cluster_result = run_cluster_stage(
        cache_dir=tmp_cache_dir, embedding_run_id="r_embed",
        cluster_run_id="r_cluster", algo="dbscan",
        params={"eps": 0.7, "min_samples": 2, "metric": "cosine", "standardize": True},
    )
    storage.finish_run("r_cluster", "ok", metrics=cluster_result.model_dump())

    storage.begin_run("r_match", "matching", config={})
    match_result = run_matching_stage(
        storage=storage, cache_dir=tmp_cache_dir,
        matching_run_id="r_match", embedding_run_id="r_embed",
        cluster_run_id="r_cluster", distance_threshold=2.0, max_queue_size=100,
    )
    storage.finish_run("r_match", "ok", metrics=match_result.model_dump())

    run_project_stage(
        cache_dir=tmp_cache_dir, embedding_run_id="r_embed",
        projection_run_id="r_project",
    )

    override_storage(storage)
    override_settings(Settings(db_path=tmp_db_path, cache_dir=tmp_cache_dir))
    app = create_app()
    return TestClient(app)
