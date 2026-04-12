"""`catalog ingest` — load an iNat CSV into the annotations table."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.ingest.inat import ingest_inat_csv
from whaleshark_reid.storage.db import Storage


def ingest_command(
    csv: Path = typer.Option(..., "--csv", exists=True, help="Path to minimal or dfx CSV"),
    photos_dir: Path = typer.Option(..., "--photos-dir", exists=True, file_okay=False),
    source: str = typer.Option("inat", "--source"),
    rich_csv: Optional[Path] = typer.Option(
        None, "--rich-csv", exists=True,
        help="Optional provenance CSV — accepts either the dfx schema or the raw iNat export (df_exploded_inat_v1 format)"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Ingest an iNat CSV into the annotations table."""
    storage = Storage(db_path)
    storage.init_schema()
    config = {
        "csv": str(csv),
        "photos_dir": str(photos_dir),
        "source": source,
        "rich_csv": str(rich_csv) if rich_csv else None,
    }
    with run_context(stage="ingest", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(f"reading {csv}")
        result = ingest_inat_csv(
            csv_path=csv,
            photos_dir=photos_dir,
            storage=storage,
            run_id=ctx.run_id,
            rich_csv_path=rich_csv,
        )
        ctx.logger.info(
            f"ingested={result.n_ingested} skipped={result.n_skipped_existing} "
            f"missing_files={result.n_missing_files}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"ingested {result.n_ingested} / "
        f"skipped {result.n_skipped_existing} / "
        f"missing {result.n_missing_files}"
    )
