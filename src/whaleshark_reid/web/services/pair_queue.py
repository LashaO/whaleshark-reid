"""Pair queue service — read queue entries, submit decisions, navigate."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.core.schema import Annotation
from whaleshark_reid.storage.db import Storage


class PairView(BaseModel):
    queue_id: int
    run_id: str
    position: int
    total: int
    ann_a: Annotation
    ann_b: Annotation
    distance: float
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    same_cluster: bool = False
    gps_delta_km: Optional[float] = None
    time_delta_days: Optional[int] = None


def _pair_from_row(row, storage: Storage, total: int) -> PairView:
    ann_a = storage.get_annotation(row["ann_a_uuid"])
    ann_b = storage.get_annotation(row["ann_b_uuid"])

    gps_delta = None
    time_delta = None
    if ann_a and ann_b:
        if ann_a.gps_lat_captured and ann_b.gps_lat_captured:
            from math import radians, sin, cos, asin, sqrt
            lat1, lon1 = radians(ann_a.gps_lat_captured), radians(ann_a.gps_lon_captured or 0)
            lat2, lon2 = radians(ann_b.gps_lat_captured), radians(ann_b.gps_lon_captured or 0)
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            gps_delta = round(2 * 6371 * asin(sqrt(a)), 1)

        if ann_a.date_captured and ann_b.date_captured:
            try:
                d1 = datetime.fromisoformat(ann_a.date_captured)
                d2 = datetime.fromisoformat(ann_b.date_captured)
                time_delta = abs((d2 - d1).days)
            except (ValueError, TypeError):
                pass

    return PairView(
        queue_id=row["queue_id"],
        run_id=row["run_id"],
        position=row["position"],
        total=total,
        ann_a=ann_a,
        ann_b=ann_b,
        distance=row["distance"],
        cluster_a=row["cluster_a"],
        cluster_b=row["cluster_b"],
        same_cluster=bool(row["same_cluster"]),
        gps_delta_km=gps_delta,
        time_delta_days=time_delta,
    )


def _distance_filter_sql(min_d: Optional[float], max_d: Optional[float]) -> tuple[str, list]:
    """Build a SQL fragment + params for distance range filtering. Returns ('', [])
    if no filters are active."""
    clauses = []
    params: list = []
    if min_d is not None:
        clauses.append("distance >= ?")
        params.append(min_d)
    if max_d is not None:
        clauses.append("distance <= ?")
        params.append(max_d)
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def get_pair(
    storage: Storage,
    run_id: str,
    position: int,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
) -> Optional[PairView]:
    """Get the pair at position `position` within the filtered subset.

    When min_d/max_d are set, position is an offset into the FILTERED queue, not
    the full queue — so position=0 is the lowest-distance pair within the filter range.
    """
    filter_sql, filter_params = _distance_filter_sql(min_d, max_d)

    # Count pairs in the filtered view
    total = storage.conn.execute(
        f"SELECT COUNT(*) FROM pair_queue WHERE run_id = ?{filter_sql}",
        [run_id, *filter_params],
    ).fetchone()[0]
    if total == 0:
        return None

    row = storage.conn.execute(
        f"""
        SELECT * FROM pair_queue
        WHERE run_id = ?{filter_sql}
        ORDER BY position ASC
        LIMIT 1 OFFSET ?
        """,
        [run_id, *filter_params, position],
    ).fetchone()
    if row is None:
        return None
    return _pair_from_row(row, storage, total)


def get_pair_by_id(storage: Storage, queue_id: int) -> Optional[PairView]:
    row = storage.conn.execute(
        "SELECT * FROM pair_queue WHERE queue_id = ?", (queue_id,)
    ).fetchone()
    if row is None:
        return None
    total = storage.count("pair_queue", run_id=row["run_id"])
    return _pair_from_row(row, storage, total)


def submit_decision(
    storage: Storage, queue_id: int, decision: str, user: str, notes: str = ""
) -> None:
    pair = storage.conn.execute(
        "SELECT ann_a_uuid, ann_b_uuid, distance, run_id FROM pair_queue WHERE queue_id = ?",
        (queue_id,),
    ).fetchone()
    if pair is None:
        return
    storage.conn.execute(
        """
        INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, distance, run_id, user, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pair["ann_a_uuid"], pair["ann_b_uuid"], decision,
            pair["distance"], pair["run_id"], user, notes,
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def get_next_undecided(
    storage: Storage,
    run_id: str,
    from_queue_id: int,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
) -> Optional[PairView]:
    """Return the next pair (by position) within the filter range whose pair has
    no active match/no_match decision. `from_queue_id` is the queue_id we're
    advancing past — the next pair must have position > that queue's position."""
    # Resolve the position of from_queue_id so we can skip past it
    current = storage.conn.execute(
        "SELECT position FROM pair_queue WHERE queue_id = ?", (from_queue_id,)
    ).fetchone()
    from_position = current["position"] if current else -1

    filter_sql, filter_params = _distance_filter_sql(min_d, max_d)

    total = storage.conn.execute(
        f"SELECT COUNT(*) FROM pair_queue WHERE run_id = ?{filter_sql}",
        [run_id, *filter_params],
    ).fetchone()[0]

    row = storage.conn.execute(
        f"""
        SELECT pq.* FROM pair_queue pq
        WHERE pq.run_id = ? AND pq.position > ?{filter_sql}
        AND NOT EXISTS (
            SELECT 1 FROM pair_decisions pd
            WHERE pd.ann_a_uuid = pq.ann_a_uuid AND pd.ann_b_uuid = pq.ann_b_uuid
            AND pd.decision IN ('match', 'no_match')
            AND pd.superseded_by IS NULL
        )
        ORDER BY pq.position ASC
        LIMIT 1
        """,
        [run_id, from_position, *filter_params],
    ).fetchone()
    if row is None:
        return None
    return _pair_from_row(row, storage, total)


def filtered_position_index(
    storage: Storage,
    run_id: str,
    pair_position: int,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
) -> int:
    """Return the 0-indexed offset of the pair at `pair_position` (the raw pair_queue.position)
    within the filtered subset. I.e., how many filtered pairs come before it."""
    filter_sql, filter_params = _distance_filter_sql(min_d, max_d)
    row = storage.conn.execute(
        f"SELECT COUNT(*) FROM pair_queue WHERE run_id = ? AND position < ?{filter_sql}",
        [run_id, pair_position, *filter_params],
    ).fetchone()
    return row[0]


def get_distance_histogram(
    storage: Storage, run_id: str, n_bins: int = 20
) -> dict:
    """Return histogram of pair distances for the given matching run.

    Returns {
        "bins": [edge_0, edge_1, ..., edge_n],  # n+1 edges for n bins
        "counts": [count_0, ..., count_n-1],
        "min": float,
        "max": float,
        "n_total": int,
    }
    """
    rows = storage.conn.execute(
        "SELECT distance FROM pair_queue WHERE run_id = ? ORDER BY distance",
        (run_id,),
    ).fetchall()
    distances = [r["distance"] for r in rows]
    n_total = len(distances)

    if n_total == 0:
        return {"bins": [0.0, 2.0], "counts": [0], "min": 0.0, "max": 0.0, "n_total": 0}

    dmin = min(distances)
    dmax = max(distances)
    # Pad so first/last bins aren't empty
    span = max(dmax - dmin, 1e-6)
    edges = [dmin + (span * i / n_bins) for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for d in distances:
        # Bin index
        if d >= dmax:
            idx = n_bins - 1
        else:
            idx = int((d - dmin) / span * n_bins)
            idx = max(0, min(idx, n_bins - 1))
        counts[idx] += 1
    return {
        "bins": edges,
        "counts": counts,
        "min": dmin,
        "max": dmax,
        "n_total": n_total,
    }
