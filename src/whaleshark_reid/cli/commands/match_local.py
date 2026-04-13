"""CLI command: bulk-precompute local feature matches for every pair in a queue.

Feature extraction is deduped per unique annotation UUID; each pair then
reuses cached features.
"""
from __future__ import annotations

from pathlib import Path

import typer

from whaleshark_reid.core.match import lightglue as lg
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.services.local_match import write_cached


def match_local_command(
    run_id: str = typer.Option(..., "--run-id", help="pair_queue run_id to process"),
    db_path: Path = typer.Option(..., "--db-path", help="path to catalog SQLite DB"),
    extractor: str = typer.Option("aliked", "--extractor"),
    limit: int = typer.Option(0, "--limit", help="0 = all pairs"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    storage = Storage(db_path)
    storage.init_schema()

    rows = storage.conn.execute(
        """
        SELECT pq.queue_id, a.file_path AS pa, b.file_path AS pb,
               pq.ann_a_uuid, pq.ann_b_uuid
        FROM pair_queue pq
        JOIN annotations a ON a.annotation_uuid = pq.ann_a_uuid
        JOIN annotations b ON b.annotation_uuid = pq.ann_b_uuid
        WHERE pq.run_id = ?
        ORDER BY pq.position
        """, (run_id,),
    ).fetchall()

    if limit > 0:
        rows = rows[:limit]

    # Skip already-cached pairs unless --overwrite. Single query instead of N
    # per-row lookups — important for large queues.
    if not overwrite:
        cached_ids = {
            r["queue_id"] for r in storage.conn.execute(
                "SELECT queue_id FROM pair_matches WHERE extractor = ?", (extractor,)
            ).fetchall()
        }
        rows = [r for r in rows if r["queue_id"] not in cached_ids]

    if not rows:
        typer.echo("nothing to do")
        return

    # Dedup annotation paths (annotation UUID as identity; same UUID => same file)
    path_by_uuid: dict[str, str] = {}
    for r in rows:
        path_by_uuid[r["ann_a_uuid"]] = r["pa"]
        path_by_uuid[r["ann_b_uuid"]] = r["pb"]

    typer.echo(f"extracting features for {len(path_by_uuid)} unique annotations")
    feats_by_path = lg.extract_features_batch(
        list(path_by_uuid.values()), extractor=extractor,
    )

    typer.echo(f"matching {len(rows)} pairs")
    pair_paths = [(r["pa"], r["pb"]) for r in rows]
    results = lg.match_pairs_batch(pair_paths, feats_by_path, extractor=extractor)

    with storage.transaction():
        for r, res in zip(rows, results):
            write_cached(storage, r["queue_id"], res)
    typer.echo(f"wrote {len(results)} rows to pair_matches")
