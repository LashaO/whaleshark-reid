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
        SELECT
            pq.queue_id, pq.ann_a_uuid, pq.ann_b_uuid,
            a.file_path AS pa, a.bbox_x AS ax, a.bbox_y AS ay,
                a.bbox_w AS aw, a.bbox_h AS ah, a.theta AS ta,
            b.file_path AS pb, b.bbox_x AS bx, b.bbox_y AS by_,
                b.bbox_w AS bw, b.bbox_h AS bh, b.theta AS tb
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

    def _bbox(x, y, w, h):
        return [x, y, w, h] if None not in (x, y, w, h) else None

    # Dedup by annotation UUID. Each UUID gets one feature extraction, reused
    # across every pair that annotation appears in. Features come from the
    # bbox+theta chip, same coordinate frame the UI paints onto.
    specs_by_uuid: dict[str, tuple] = {}
    for r in rows:
        specs_by_uuid.setdefault(r["ann_a_uuid"], (
            r["ann_a_uuid"], r["pa"],
            _bbox(r["ax"], r["ay"], r["aw"], r["ah"]), float(r["ta"] or 0.0),
        ))
        specs_by_uuid.setdefault(r["ann_b_uuid"], (
            r["ann_b_uuid"], r["pb"],
            _bbox(r["bx"], r["by_"], r["bw"], r["bh"]), float(r["tb"] or 0.0),
        ))

    typer.echo(f"extracting features for {len(specs_by_uuid)} unique annotations")
    feats_by_uuid = lg.extract_features_batch(
        list(specs_by_uuid.values()), extractor=extractor,
    )

    typer.echo(f"matching {len(rows)} pairs")
    uuid_pairs = [(r["ann_a_uuid"], r["ann_b_uuid"]) for r in rows]
    results = lg.match_pairs_batch(uuid_pairs, feats_by_uuid, extractor=extractor)

    with storage.transaction():
        for r, res in zip(rows, results):
            write_cached(storage, r["queue_id"], res)
    typer.echo(f"wrote {len(results)} rows to pair_matches")
