"""`catalog run-all` — runs ingest → embed → cluster → matching → project in sequence.

Each stage gets its own run_id (no hierarchical grouping). Related runs are
discovered later by timestamp window / config overlap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command
from whaleshark_reid.cli.commands.project import project_command


def run_all_command(
    csv: Path = typer.Option(..., "--csv", exists=True),
    photos_dir: Path = typer.Option(..., "--photos-dir", exists=True, file_okay=False),
    rich_csv: Optional[Path] = typer.Option(None, "--rich-csv", exists=True),
    eps: float = typer.Option(0.7, "--eps"),
    min_samples: int = typer.Option(2, "--min-samples"),
    distance_threshold: float = typer.Option(1.0, "--distance-threshold"),
    max_queue_size: int = typer.Option(2000, "--max-queue-size"),
    use_bbox: bool = typer.Option(True, "--use-bbox/--no-use-bbox"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    device: Optional[str] = typer.Option(None, "--device"),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Run ingest → embed → cluster → matching → project as five separate runs."""
    # ⚠️  MAINTAINER WARNING: each sub-command is a Typer-decorated function whose
    # "default" values are actually OptionInfo instances, not real Python values.
    # When calling these functions directly (as we do below), you MUST pass every
    # parameter explicitly — if you omit one, Python will use the OptionInfo as the
    # actual argument value, which will crash deep inside the core stage function
    # with a cryptic "OptionInfo has no attribute" error.
    #
    # If you add a new parameter to any sub-command, YOU MUST also thread it
    # through the corresponding call site here.
    typer.echo("→ ingest")
    ingest_command(
        csv=csv,
        photos_dir=photos_dir,
        source="inat",
        rich_csv=rich_csv,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ embed")
    embed_command(
        model="conservationxlabs/miewid-msv3",
        batch_size=batch_size,
        num_workers=num_workers,
        use_bbox=use_bbox,
        only_missing=True,
        device=device,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ cluster")
    cluster_command(
        algo="dbscan",
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        standardize=True,
        min_cluster_size=3,
        embedding_run_id=None,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ matching")
    matching_command(
        distance_threshold=distance_threshold,
        max_queue_size=max_queue_size,
        embedding_run_id=None,
        cluster_run_id=None,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ project")
    project_command(
        algo="umap",
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        embedding_run_id=None,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("✓ run-all complete")
