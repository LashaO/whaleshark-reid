"""Shared fixtures for CLI tests."""
from __future__ import annotations

import pytest
import torch
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


class _StubMiewId(torch.nn.Module):
    """Deterministic stand-in for MiewIdNet so CLI tests don't hit HuggingFace."""

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
