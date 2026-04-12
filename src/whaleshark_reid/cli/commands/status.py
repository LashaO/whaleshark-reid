"""`catalog status` — show the latest run for each pipeline stage."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from whaleshark_reid.storage.db import Storage

# Pipeline stages in execution order
STAGES = ("ingest", "embed", "cluster", "matching", "project", "rebuild")


def status_command(
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
) -> None:
    """Show the most recent run for each pipeline stage."""
    storage = Storage(db_path)
    storage.init_schema()

    typer.echo(f"{'Stage':<20} {'Status':<10} {'Run ID':<30} {'Duration':<12} {'Key Metrics'}")
    typer.echo("-" * 100)

    for stage in STAGES:
        row = storage.conn.execute(
            """
            SELECT run_id, status, config_json, metrics_json, started_at, finished_at
            FROM runs
            WHERE stage = ?
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (stage,),
        ).fetchone()

        if row is None:
            typer.echo(f"{stage:<20} {'—':<10} {'(no runs)':<30}")
            continue

        status = row["status"]
        run_id = row["run_id"]

        # Compute duration if both timestamps exist
        duration_str = "—"
        if row["started_at"] and row["finished_at"]:
            from datetime import datetime
            try:
                t0 = datetime.fromisoformat(row["started_at"])
                t1 = datetime.fromisoformat(row["finished_at"])
                secs = (t1 - t0).total_seconds()
                duration_str = f"{secs:.1f}s"
            except (ValueError, TypeError):
                pass

        # Extract a few key metrics for display
        metrics_preview = ""
        if row["metrics_json"]:
            try:
                m = json.loads(row["metrics_json"])
                # Pick the most informative metric per stage
                if stage == "ingest":
                    metrics_preview = f"ingested={m.get('n_ingested', '?')}"
                elif stage == "embed":
                    metrics_preview = f"n={m.get('n_embedded', '?')} dim={m.get('embed_dim', '?')}"
                elif stage == "cluster":
                    metrics_preview = f"clusters={m.get('n_clusters', '?')} noise={m.get('n_noise', '?')}"
                elif stage == "matching":
                    metrics_preview = f"pairs={m.get('n_pairs', '?')}"
                elif stage == "project":
                    metrics_preview = f"points={m.get('n_points', '?')}"
                elif stage == "rebuild":
                    metrics_preview = f"individuals={m.get('n_components', '?')}"
            except (json.JSONDecodeError, AttributeError):
                pass

        typer.echo(f"{stage:<20} {status:<10} {run_id:<30} {duration_str:<12} {metrics_preview}")

    # Also show counts
    typer.echo("")
    typer.echo(f"Annotations: {storage.count('annotations')}")
    typer.echo(f"Pair decisions: {storage.count('pair_decisions')}")
    n_queue = storage.conn.execute("SELECT COUNT(*) FROM pair_queue").fetchone()[0]
    typer.echo(f"Pair queue entries: {n_queue}")
