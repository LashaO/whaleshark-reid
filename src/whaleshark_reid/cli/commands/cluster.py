"""`catalog cluster` — cluster embeddings via DBSCAN or HDBSCAN."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.cluster.common import run_cluster_stage
from whaleshark_reid.storage.db import Storage


def cluster_command(
    algo: str = typer.Option("dbscan", "--algo", help="dbscan | hdbscan"),
    eps: float = typer.Option(0.7, "--eps", help="DBSCAN only"),
    min_samples: int = typer.Option(2, "--min-samples"),
    metric: str = typer.Option("cosine", "--metric"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize"),
    min_cluster_size: int = typer.Option(3, "--min-cluster-size", help="HDBSCAN only"),
    embedding_run_id: Optional[str] = typer.Option(
        None, "--embedding-run-id", help="default: latest successful embed run"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Cluster embeddings into proposed individual groups."""
    storage = Storage(db_path)
    storage.init_schema()

    if embedding_run_id is None:
        embedding_run_id = storage.get_latest_run_id(stage="embed")
        if embedding_run_id is None:
            typer.echo("error: no successful embed run found. Run `catalog embed` first.", err=True)
            raise typer.Exit(code=2)

    if algo == "dbscan":
        params = {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "standardize": standardize,
        }
    elif algo == "hdbscan":
        params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": None,
            "metric": "euclidean" if metric == "cosine" else metric,
        }
    else:
        typer.echo(f"error: unknown cluster algo: {algo}", err=True)
        raise typer.Exit(code=2)

    config = {
        "algo": algo,
        "params": params,
        "embedding_run_id": embedding_run_id,
    }
    with run_context(stage="cluster", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(f"clustering algo={algo} params={params} embedding_run={embedding_run_id}")
        result = run_cluster_stage(
            cache_dir=cache_dir,
            embedding_run_id=embedding_run_id,
            cluster_run_id=ctx.run_id,
            algo=algo,
            params=params,
        )
        ctx.logger.info(
            f"n_clusters={result.n_clusters} n_noise={result.n_noise} "
            f"largest={result.largest_cluster_size}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"clustered: {result.n_clusters} clusters / "
        f"{result.n_noise} noise / "
        f"largest={result.largest_cluster_size} / "
        f"singleton_fraction={result.singleton_fraction:.2f}"
    )
