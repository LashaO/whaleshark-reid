"""Parquet-backed 2D projection cache (UMAP/t-SNE).

One file per run: cache_dir/projections/<run_id>.parquet
Columns: annotation_uuid, x, y, algo, params_json
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _projections_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / "projections" / f"{run_id}.parquet"


def write_projections(cache_dir: Path, run_id: str, rows: Iterable[dict]) -> Path:
    out = _projections_path(cache_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(out, engine="pyarrow", index=False)
    return out


def read_projections(cache_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_projections_path(cache_dir, run_id), engine="pyarrow")
