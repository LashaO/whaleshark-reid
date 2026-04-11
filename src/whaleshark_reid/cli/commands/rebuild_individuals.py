"""`catalog rebuild-individuals` — materialize annotations.name_uuid from pair_decisions."""
from __future__ import annotations

from pathlib import Path

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.feedback.unionfind import rebuild_individuals_cache
from whaleshark_reid.storage.db import Storage


def rebuild_individuals_command(
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Rebuild annotations.name_uuid from confirmed pair_decisions via union-find."""
    storage = Storage(db_path)
    storage.init_schema()
    config: dict = {}
    with run_context(stage="rebuild", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info("rebuilding individuals from pair_decisions")
        result = rebuild_individuals_cache(storage)
        ctx.logger.info(
            f"n_components={result.n_components} n_singletons={result.n_singletons} "
            f"updated={result.n_annotations_updated}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"rebuilt: {result.n_components} individuals / "
        f"{result.n_singletons} singletons / "
        f"{result.n_annotations_updated} annotations updated"
    )
