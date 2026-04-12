"""Cluster view service — projection data and cluster summary."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.storage.cluster_cache import read_clusters
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.projection_cache import read_projections


class ProjectionPoint(BaseModel):
    annotation_uuid: str
    x: float
    y: float
    cluster_label: int


class ProjectionResponse(BaseModel):
    run_id: str
    points: list[ProjectionPoint]


def get_projection(
    cache_dir: Path, projection_run_id: str, cluster_run_id: str
) -> Optional[ProjectionResponse]:
    try:
        proj_df = read_projections(cache_dir, projection_run_id)
        cluster_df = read_clusters(cache_dir, cluster_run_id)
    except FileNotFoundError:
        return None

    cluster_map = dict(zip(cluster_df["annotation_uuid"], cluster_df["cluster_label"].astype(int)))

    points = [
        ProjectionPoint(
            annotation_uuid=row["annotation_uuid"],
            x=float(row["x"]),
            y=float(row["y"]),
            cluster_label=cluster_map.get(row["annotation_uuid"], -1),
        )
        for _, row in proj_df.iterrows()
    ]

    return ProjectionResponse(run_id=projection_run_id, points=points)
