"""CLI: bulk-precompute PairX visualizations.

PairX backprops through MiewID — slow per pair (~5-30s on CPU). Renders to
PNG, writes filesystem cache that the web routes serve. Resume-safe: skips
existing files unless --overwrite.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import typer

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.services import pairx as svc


def pairx_precompute_command(
    run_id: str = typer.Option(..., "--run-id"),
    db_path: Path = typer.Option(..., "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
    layer: str = typer.Option("backbone.blocks.3", "--layer"),
    k_lines: int = typer.Option(20, "--k-lines"),
    k_colors: int = typer.Option(5, "--k-colors"),
    limit: int = typer.Option(0, "--limit", help="0 = all pairs"),
    overwrite: bool = typer.Option(False, "--overwrite"),
):
    from PIL import Image
    from whaleshark_reid.core.explain.pairx import explain_pair

    storage = Storage(db_path)
    storage.init_schema()
    rows = storage.conn.execute(
        """
        SELECT
            pq.queue_id,
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

    def _bbox(x, y, w, h):
        return [x, y, w, h] if None not in (x, y, w, h) else None

    todo = []
    for r in rows:
        p = svc.png_path(cache_dir, r["queue_id"], layer)
        if not overwrite and p.exists():
            continue
        todo.append(r)

    typer.echo(f"computing PairX for {len(todo)} of {len(rows)} pairs (skipping {len(rows) - len(todo)} cached)")
    if not todo:
        return

    t0 = time.time()
    for i, r in enumerate(todo, 1):
        try:
            images = explain_pair(
                r["pa"], _bbox(r["ax"], r["ay"], r["aw"], r["ah"]), float(r["ta"] or 0.0),
                r["pb"], _bbox(r["bx"], r["by_"], r["bw"], r["bh"]), float(r["tb"] or 0.0),
                layer_key=layer, k_lines=k_lines, k_colors=k_colors,
            )
        except Exception as e:
            typer.echo(f"  [{i}/{len(todo)}] queue_id={r['queue_id']} FAILED: {type(e).__name__}: {e}")
            continue
        if not images:
            typer.echo(f"  [{i}/{len(todo)}] queue_id={r['queue_id']} produced 0 images, skipping")
            continue
        buf = io.BytesIO()
        Image.fromarray(images[0]).save(buf, format="PNG")
        svc.write_png(cache_dir, r["queue_id"], layer, buf.getvalue())
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else 0
        eta = (len(todo) - i) / rate if rate > 0 else 0
        typer.echo(f"  [{i}/{len(todo)}] queue_id={r['queue_id']} ok ({elapsed:.0f}s elapsed, {rate:.2f}/s, ETA {eta:.0f}s)")
    typer.echo(f"done in {time.time() - t0:.0f}s")
