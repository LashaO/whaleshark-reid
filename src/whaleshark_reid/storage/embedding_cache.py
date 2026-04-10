"""Parquet-backed embedding cache.

One file per run: cache_dir/embeddings/<run_id>.parquet
Columns: annotation_uuid, embedding (list[float]), model_id, model_version, created_at
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _embeddings_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / "embeddings" / f"{run_id}.parquet"


def write_embeddings(cache_dir: Path, run_id: str, rows: Iterable[dict]) -> Path:
    out = _embeddings_path(cache_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows))
    df.to_parquet(out, engine="pyarrow", index=False)
    return out


def read_embeddings(cache_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_embeddings_path(cache_dir, run_id), engine="pyarrow")


def read_embeddings_as_array(cache_dir: Path, run_id: str) -> tuple[list[str], np.ndarray]:
    df = read_embeddings(cache_dir, run_id)
    uuids = df["annotation_uuid"].tolist()
    mat = np.vstack([np.asarray(e, dtype=np.float32) for e in df["embedding"]])
    return uuids, mat


def existing_annotation_uuids(cache_dir: Path, run_id: str) -> set[str]:
    path = _embeddings_path(cache_dir, run_id)
    if not path.exists():
        return set()
    df = pd.read_parquet(path, engine="pyarrow", columns=["annotation_uuid"])
    return set(df["annotation_uuid"].tolist())
