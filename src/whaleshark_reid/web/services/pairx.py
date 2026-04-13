"""Service layer for PairX visualizations.

Filesystem cache (no DB row): each pair's PNG lives at
`<cache_dir>/pairx/<queue_id>__<safe_layer>.png`. Reads are 404 if missing.
Writes are atomic-ish (write to .tmp then rename).
"""
from __future__ import annotations

import re
from pathlib import Path

from whaleshark_reid.storage.db import Storage


def _safe_layer(layer_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", layer_key)


def png_path(cache_dir: Path, queue_id: int, layer_key: str) -> Path:
    """Deterministic path for a cached PairX render."""
    return Path(cache_dir) / "pairx" / f"{queue_id}__{_safe_layer(layer_key)}.png"


def write_png(cache_dir: Path, queue_id: int, layer_key: str, png_bytes: bytes) -> Path:
    p = png_path(cache_dir, queue_id, layer_key)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".png.tmp")
    tmp.write_bytes(png_bytes)
    tmp.replace(p)
    return p


def lookup_pair_chip_specs(storage: Storage, queue_id: int):
    """Same shape as web.services.local_match.lookup_pair_chip_specs but
    duplicated here to avoid a cross-module import for one query."""
    row = storage.conn.execute(
        """
        SELECT
            a.file_path AS pa,
            a.bbox_x AS ax, a.bbox_y AS ay, a.bbox_w AS aw, a.bbox_h AS ah, a.theta AS ta,
            b.file_path AS pb,
            b.bbox_x AS bx, b.bbox_y AS by_, b.bbox_w AS bw, b.bbox_h AS bh, b.theta AS tb
        FROM pair_queue pq
        JOIN annotations a ON a.annotation_uuid = pq.ann_a_uuid
        JOIN annotations b ON b.annotation_uuid = pq.ann_b_uuid
        WHERE pq.queue_id = ?
        """,
        (queue_id,),
    ).fetchone()
    if row is None:
        raise LookupError(f"no pair_queue row with queue_id={queue_id}")

    def _bbox(x, y, w, h):
        return [x, y, w, h] if None not in (x, y, w, h) else None

    spec_a = (row["pa"], _bbox(row["ax"], row["ay"], row["aw"], row["ah"]),
              float(row["ta"] or 0.0))
    spec_b = (row["pb"], _bbox(row["bx"], row["by_"], row["bw"], row["bh"]),
              float(row["tb"] or 0.0))
    return spec_a, spec_b
