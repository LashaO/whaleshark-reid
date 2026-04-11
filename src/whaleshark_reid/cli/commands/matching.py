"""`catalog matching` — compute pair candidates and write the review queue."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.matching.pairs import run_matching_stage
from whaleshark_reid.storage.db import Storage


def matching_command(
    distance_threshold: float = typer.Option(1.0, "--distance-threshold"),
    max_queue_size: int = typer.Option(2000, "--max-queue-size"),
    embedding_run_id: Optional[str] = typer.Option(
        None, "--embedding-run-id", help="default: latest successful embed run"
    ),
    cluster_run_id: Optional[str] = typer.Option(
        None, "--cluster-run-id", help="default: latest successful cluster run"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Build the sorted pair_queue from embeddings + cluster labels."""
    storage = Storage(db_path)
    storage.init_schema()

    if embedding_run_id is None:
        embedding_run_id = storage.get_latest_run_id(stage="embed")
        if embedding_run_id is None:
            typer.echo("error: no successful embed run found. Run `catalog embed` first.", err=True)
            raise typer.Exit(code=2)
    if cluster_run_id is None:
        cluster_run_id = storage.get_latest_run_id(stage="cluster")
        if cluster_run_id is None:
            typer.echo("error: no successful cluster run found. Run `catalog cluster` first.", err=True)
            raise typer.Exit(code=2)

    config = {
        "distance_threshold": distance_threshold,
        "max_queue_size": max_queue_size,
        "embedding_run_id": embedding_run_id,
        "cluster_run_id": cluster_run_id,
    }
    with run_context(stage="matching", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(
            f"matching threshold={distance_threshold} max_queue={max_queue_size} "
            f"embed={embedding_run_id} cluster={cluster_run_id}"
        )
        result = run_matching_stage(
            storage=storage,
            cache_dir=cache_dir,
            matching_run_id=ctx.run_id,
            embedding_run_id=embedding_run_id,
            cluster_run_id=cluster_run_id,
            distance_threshold=distance_threshold,
            max_queue_size=max_queue_size,
        )
        ctx.logger.info(
            f"n_pairs={result.n_pairs} same_cluster={result.n_same_cluster} "
            f"cross={result.n_cross_cluster} median_dist={result.median_distance:.3f}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"queued {result.n_pairs} pairs / "
        f"same_cluster={result.n_same_cluster} cross_cluster={result.n_cross_cluster} / "
        f"median_dist={result.median_distance:.3f}"
    )
