"""`catalog embed` — extract MiewID embeddings for ingested annotations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.embed.miewid import run_embed_stage
from whaleshark_reid.storage.db import Storage


def embed_command(
    model: str = typer.Option("conservationxlabs/miewid-msv3", "--model"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    use_bbox: bool = typer.Option(True, "--use-bbox/--no-use-bbox"),
    only_missing: bool = typer.Option(
        True, "--only-missing/--force-reembed",
        help="Skip annotations already in THIS run's parquet (per-run cache, not global). "
             "Currently every new run_id starts with an empty cache, so this flag is "
             "effectively a no-op. Use --force-reembed to be explicit.",
    ),
    device: Optional[str] = typer.Option(None, "--device", help="cuda | cpu | mps; default: auto"),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Embed all ingested annotations not yet in the cache.

    Writes to cache_dir/embeddings/<run_id>.parquet.
    """
    storage = Storage(db_path)
    storage.init_schema()
    config = {
        "model": model,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_bbox": use_bbox,
        "only_missing": only_missing,
        "device": device,
    }
    with run_context(stage="embed", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(f"embedding model={model} batch_size={batch_size} use_bbox={use_bbox}")
        # NOTE: only_missing checks against this run's own parquet (per-run_id isolation),
        # NOT across all prior embed runs. A future enhancement would union across all
        # prior successful embed parquets to skip truly-already-embedded annotations.
        result = run_embed_stage(
            storage=storage,
            cache_dir=cache_dir,
            run_id=ctx.run_id,
            model_id=model,
            batch_size=batch_size,
            num_workers=num_workers,
            use_bbox=use_bbox,
            only_missing=only_missing,
            device=device,
        )
        ctx.logger.info(
            f"embedded={result.n_embedded} skipped={result.n_skipped_existing} "
            f"failed={result.n_failed} dim={result.embed_dim} duration_s={result.duration_s:.1f}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"embedded {result.n_embedded} (dim={result.embed_dim}) / "
        f"skipped {result.n_skipped_existing} / "
        f"failed {result.n_failed} / "
        f"{result.duration_s:.1f}s"
    )
