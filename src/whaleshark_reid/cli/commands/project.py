"""`catalog project` — 2D UMAP projection of embeddings for the cluster scatter view."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.cluster.project import run_project_stage
from whaleshark_reid.storage.db import Storage


def project_command(
    algo: str = typer.Option("umap", "--algo"),
    n_neighbors: int = typer.Option(15, "--n-neighbors"),
    min_dist: float = typer.Option(0.1, "--min-dist"),
    metric: str = typer.Option("cosine", "--metric"),
    random_state: int = typer.Option(42, "--random-state"),
    embedding_run_id: Optional[str] = typer.Option(
        None, "--embedding-run-id", help="default: latest successful embed run"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Project embeddings to 2D for the cluster view."""
    storage = Storage(db_path)
    storage.init_schema()

    if embedding_run_id is None:
        embedding_run_id = storage.get_latest_run_id(stage="embed")
        if embedding_run_id is None:
            typer.echo("error: no successful embed run found. Run `catalog embed` first.", err=True)
            raise typer.Exit(code=2)

    config = {
        "algo": algo,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
        "embedding_run_id": embedding_run_id,
    }
    with run_context(stage="project", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(
            f"projecting algo={algo} n_neighbors={n_neighbors} embedding_run={embedding_run_id}"
        )
        result = run_project_stage(
            cache_dir=cache_dir,
            embedding_run_id=embedding_run_id,
            projection_run_id=ctx.run_id,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        ctx.logger.info(f"projected n_points={result.n_points}")
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(f"projected {result.n_points} points / algo={result.algo}")
