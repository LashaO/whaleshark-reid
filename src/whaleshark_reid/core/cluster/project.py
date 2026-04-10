"""2D UMAP projection for the web cluster view."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.core.schema import ProjectionPoint, ProjectStageResult


def run_umap(
    embeddings: np.ndarray,
    annotation_uuids: list[str],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> list[ProjectionPoint]:
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }
    return [
        ProjectionPoint(
            annotation_uuid=uuid,
            x=float(coords[i, 0]),
            y=float(coords[i, 1]),
            algo="umap",
            params=params,
        )
        for i, uuid in enumerate(annotation_uuids)
    ]


def run_project_stage(
    cache_dir: Path,
    embedding_run_id: str,
    projection_run_id: str,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> ProjectStageResult:
    from whaleshark_reid.storage.embedding_cache import read_embeddings_as_array
    from whaleshark_reid.storage.projection_cache import write_projections

    uuids, mat = read_embeddings_as_array(cache_dir, embedding_run_id)
    points = run_umap(
        mat, uuids,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    rows = [
        {
            "annotation_uuid": p.annotation_uuid,
            "x": p.x,
            "y": p.y,
            "algo": p.algo,
            "params_json": str(p.params),
        }
        for p in points
    ]
    write_projections(cache_dir, projection_run_id, rows)

    return ProjectStageResult(
        algo="umap",
        n_points=len(points),
        params={
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": metric,
            "random_state": random_state,
        },
    )
