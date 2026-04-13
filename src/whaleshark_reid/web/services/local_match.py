"""Service layer: cache read/write for local feature matches."""
from __future__ import annotations

import json

from whaleshark_reid.core.match.lightglue import MatchResult
from whaleshark_reid.storage.db import Storage


def read_cached(storage: Storage, queue_id: int, extractor: str) -> MatchResult | None:
    row = storage.conn.execute(
        "SELECT * FROM pair_matches WHERE queue_id = ? AND extractor = ?",
        (queue_id, extractor),
    ).fetchone()
    if row is None:
        return None
    blob = json.loads(row["match_data"])
    return MatchResult(
        extractor=row["extractor"],
        n_matches=row["n_matches"],
        mean_score=row["mean_score"],
        median_score=row["median_score"],
        kpts_a=blob["kpts_a"],
        kpts_b=blob["kpts_b"],
        matches=blob["matches"],
        img_a_size=json.loads(row["img_a_size"]),
        img_b_size=json.loads(row["img_b_size"]),
    )


def write_cached(storage: Storage, queue_id: int, result: MatchResult) -> None:
    blob = json.dumps({
        "kpts_a": result.kpts_a,
        "kpts_b": result.kpts_b,
        "matches": result.matches,
    })
    storage.conn.execute(
        """
        INSERT INTO pair_matches(queue_id, extractor, n_matches, mean_score,
                                 median_score, match_data, img_a_size, img_b_size)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(queue_id, extractor) DO UPDATE SET
            n_matches = excluded.n_matches,
            mean_score = excluded.mean_score,
            median_score = excluded.median_score,
            match_data = excluded.match_data,
            img_a_size = excluded.img_a_size,
            img_b_size = excluded.img_b_size,
            computed_at = CURRENT_TIMESTAMP
        """,
        (
            queue_id, result.extractor, result.n_matches,
            result.mean_score, result.median_score, blob,
            json.dumps(result.img_a_size), json.dumps(result.img_b_size),
        ),
    )


def lookup_pair_image_paths(storage: Storage, queue_id: int) -> tuple[str, str]:
    """Returns (path_a, path_b) for the two annotations in a pair.

    Kept for backward compat with callers that don't need the bbox. New code
    should use `lookup_pair_chip_specs` so the matcher can extract features on
    the same chip the reviewer sees.
    """
    spec_a, spec_b = lookup_pair_chip_specs(storage, queue_id)
    return spec_a[1], spec_b[1]


def lookup_pair_chip_specs(
    storage: Storage, queue_id: int,
) -> tuple[tuple[str, str, list | None, float], tuple[str, str, list | None, float]]:
    """Returns chip specs for both annotations in a pair.

    Each spec is (annotation_uuid, file_path, bbox_xywh, theta) — the bbox and
    theta match what `/image/{uuid}?crop=true` uses, so extracted features and
    drawn keypoints share a coordinate frame.
    """
    row = storage.conn.execute(
        """
        SELECT
            pq.ann_a_uuid AS ka, a.file_path AS pa,
            a.bbox_x AS ax, a.bbox_y AS ay, a.bbox_w AS aw, a.bbox_h AS ah, a.theta AS ta,
            pq.ann_b_uuid AS kb, b.file_path AS pb,
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

    spec_a = (
        row["ka"], row["pa"], _bbox(row["ax"], row["ay"], row["aw"], row["ah"]),
        float(row["ta"] or 0.0),
    )
    spec_b = (
        row["kb"], row["pb"], _bbox(row["bx"], row["by_"], row["bw"], row["bh"]),
        float(row["tb"] or 0.0),
    )
    return spec_a, spec_b
