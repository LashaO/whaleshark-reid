"""Parquet-backed cluster label cache.

One file per run: cache_dir/clusters/<run_id>.parquet
Columns: annotation_uuid, cluster_label, cluster_algo, cluster_params_json
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _clusters_path(cache_dir: Path, run_id: str) -> Path:
    return cache_dir / "clusters" / f"{run_id}.parquet"


def write_clusters(cache_dir: Path, run_id: str, rows: Iterable[dict]) -> Path:
    out = _clusters_path(cache_dir, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(out, engine="pyarrow", index=False)
    return out


def read_clusters(cache_dir: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(_clusters_path(cache_dir, run_id), engine="pyarrow")
