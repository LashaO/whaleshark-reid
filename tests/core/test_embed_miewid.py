"""Tests for core.embed.miewid with a stubbed HF model.

We monkeypatch transformers.AutoModel so the test suite does NOT hit HuggingFace
or require a GPU. The stub returns deterministic pseudo-embeddings so we can
verify pipeline plumbing (DataFrame construction, DataLoader, return shape).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from whaleshark_reid.core.schema import (
    Annotation,
    inat_annotation_uuid,
    inat_image_uuid,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


class _StubMiewId(torch.nn.Module):
    """Deterministic stand-in for MiewIdNet.

    extract_feat(x) returns a (B, D) tensor where each row is the mean
    of the corresponding image tensor, broadcast to D=8. Deterministic so
    tests can assert specific values.
    """

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        # A single parameter so .to(device) works
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        per_sample = x.mean(dim=(1, 2, 3))  # (B,)
        return per_sample.unsqueeze(1).expand(-1, self.embed_dim)  # (B, 8)

    def forward(self, x, label=None):
        return self.extract_feat(x)


@pytest.fixture
def stub_miewid(monkeypatch):
    from transformers import AutoModel

    def _fake_from_pretrained(*args, **kwargs):
        return _StubMiewId()

    monkeypatch.setattr(AutoModel, "from_pretrained", _fake_from_pretrained)
    yield


def _make_ann(obs_id: int, image_path: Path) -> Annotation:
    return Annotation(
        annotation_uuid=inat_annotation_uuid(obs_id, 0),
        image_uuid=inat_image_uuid(obs_id, 0),
        source="inat",
        observation_id=obs_id,
        photo_index=0,
        file_path=str(image_path),
        file_name=image_path.name,
        bbox=[0.0, 0.0, 64.0, 64.0],
        theta=0.0,
        viewpoint="unknown",
        species="whaleshark",
    )


def test_embed_annotations_returns_correct_shape(stub_miewid):
    from whaleshark_reid.core.embed.miewid import embed_annotations

    anns = [
        _make_ann(100 + i, FIXTURES / "photos" / f"obs{100 + i}_1.jpg")
        for i in range(5)
    ]
    mat = embed_annotations(anns, batch_size=2, num_workers=0, device="cpu")
    assert mat.shape == (5, 8)
    assert not np.isnan(mat).any()


def test_embed_annotations_deterministic_for_same_input(stub_miewid):
    from whaleshark_reid.core.embed.miewid import embed_annotations

    anns = [_make_ann(100, FIXTURES / "photos" / "obs100_1.jpg")]
    a = embed_annotations(anns, batch_size=1, num_workers=0, device="cpu")
    b = embed_annotations(anns, batch_size=1, num_workers=0, device="cpu")
    assert np.allclose(a, b)
