"""Shared pytest fixtures for all tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_cache_dir(tmp_path: Path) -> Path:
    """Temporary cache directory with embeddings/, clusters/, projections/, logs/ subdirs."""
    for sub in ("embeddings", "clusters", "projections", "logs"):
        (tmp_path / sub).mkdir()
    return tmp_path


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    """Temporary SQLite database path (file does not exist yet)."""
    return tmp_path / "state.db"
