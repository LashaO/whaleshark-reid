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


_UNDECIDED_SUBQUERY = (
    "NOT EXISTS ("
    " SELECT 1 FROM pair_decisions pd"
    " WHERE pd.ann_a_uuid = pq.ann_a_uuid AND pd.ann_b_uuid = pq.ann_b_uuid"
    "   AND pd.decision IN ('match', 'no_match')"
    "   AND pd.superseded_by IS NULL"
    ")"
)


def _build_filter_clauses(
    min_d: Optional[float],
    max_d: Optional[float],
    min_td: Optional[int],
    max_td: Optional[int],
    min_km: Optional[float] = None,
    max_km: Optional[float] = None,
    undecided_only: bool = False,
) -> tuple[str, list]:
    """Build WHERE clauses for pair_queue queries over the pq alias.

    Returns (where_sql, params) where where_sql is a " AND ..." fragment (or
    empty). All columns referenced are on pair_queue directly — km_delta and
    time_delta_days are pre-computed at insert / migration time, so no JOINs.

    When a km/td filter is active, pairs with a NULL in that column (missing
    date or GPS) are excluded — they can't satisfy the range.

    When `undecided_only` is True, pairs with an active match/no_match decision
    are excluded via a correlated NOT EXISTS on pair_decisions.
    """
    clauses: list[str] = []
    params: list = []

    if min_d is not None:
        clauses.append("pq.distance >= ?")
        params.append(min_d)
    if max_d is not None:
        clauses.append("pq.distance <= ?")
        params.append(max_d)
    if min_td is not None:
        clauses.append("pq.time_delta_days >= ?")
        params.append(min_td)
    if max_td is not None:
        clauses.append("pq.time_delta_days <= ?")
        params.append(max_td)
    if min_km is not None:
        clauses.append("pq.km_delta >= ?")
        params.append(min_km)
    if max_km is not None:
        clauses.append("pq.km_delta <= ?")
        params.append(max_km)
    if undecided_only:
        clauses.append(_UNDECIDED_SUBQUERY)

    where_sql = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def _order_key_expr(order_by: str, seed: Optional[int], alias: str = "pq") -> str:
    """Return a SQL expression for the sort key of a pair_queue row.

    'distance' → pq.position (already sorted by ascending distance).
    'random'   → deterministic pseudo-random from queue_id × seed so
                 pagination is stable across swaps within a session.

    `alias` prefixes the queue_id column (use '' for subqueries against
    pair_queue directly with no alias).
    """
    prefix = f"{alias}." if alias else ""
    if order_by == "random" and seed is not None:
        return f"(({prefix}queue_id * {int(seed)}) % 1000003)"
    return f"{prefix}position"


def get_pair(
    storage: Storage,
    run_id: str,
    position: int,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
    min_td: Optional[int] = None,
    max_td: Optional[int] = None,
    min_km: Optional[float] = None,
    max_km: Optional[float] = None,
    order_by: str = "distance",
    seed: Optional[int] = None,
    undecided_only: bool = True,
) -> Optional[PairView]:
    """Get the pair at position `position` within the filtered+sorted subset.

    `position` is an offset into the FILTERED+SORTED queue — position=0 is
    the first pair under the current sort order (lowest distance by default,
    or pseudo-random seeded by `seed` if order_by='random').

    `undecided_only=True` hides pairs with an active match/no_match decision
    so the active review loop shrinks as you decide. Set to False to revisit
    decided pairs.
    """
    where_sql, filter_params = _build_filter_clauses(
        min_d, max_d, min_td, max_td, min_km, max_km, undecided_only
    )
    order_key = _order_key_expr(order_by, seed)

    total = storage.conn.execute(
        f"SELECT COUNT(*) FROM pair_queue pq WHERE pq.run_id = ?{where_sql}",
        [run_id, *filter_params],
    ).fetchone()[0]
    if total == 0:
        return None

    row = storage.conn.execute(
        f"""
        SELECT pq.* FROM pair_queue pq
        WHERE pq.run_id = ?{where_sql}
        ORDER BY {order_key} ASC
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

    # Materialize annotations.name_uuid from the new decision graph. Match
    # decisions add edges (union-find connects a component); no_match/unsure
    # don't move the identity graph, but rebuild is idempotent + O(n) so we
    # just run it unconditionally on every decision. Without this, pair_decisions
    # accumulates without anything ever being surfaced on individual/annotation
    # views — "confirmed match but no ID assigned" bug.
    if decision == "match":
        from whaleshark_reid.core.feedback.unionfind import rebuild_individuals_cache
        rebuild_individuals_cache(storage)


def get_next_undecided(
    storage: Storage,
    run_id: str,
    from_queue_id: int,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
    min_td: Optional[int] = None,
    max_td: Optional[int] = None,
    min_km: Optional[float] = None,
    max_km: Optional[float] = None,
    order_by: str = "distance",
    seed: Optional[int] = None,
) -> Optional[PairView]:
    """Return the next pair (by current sort order) within the filter range
    whose pair has no active match/no_match decision. `from_queue_id` is the
    queue_id we're advancing past — the next pair must sort after it."""
    where_sql, filter_params = _build_filter_clauses(
        min_d, max_d, min_td, max_td, min_km, max_km
    )
    order_key = _order_key_expr(order_by, seed)
    # Subquery reads the sort key for from_queue_id from a plain pair_queue scan
    order_key_sub = _order_key_expr(order_by, seed, alias="")

    total = storage.conn.execute(
        f"SELECT COUNT(*) FROM pair_queue pq WHERE pq.run_id = ?{where_sql}",
        [run_id, *filter_params],
    ).fetchone()[0]

    row = storage.conn.execute(
        f"""
        SELECT pq.* FROM pair_queue pq
        WHERE pq.run_id = ? AND {order_key} > COALESCE(
            (SELECT {order_key_sub} FROM pair_queue WHERE queue_id = ?),
            -1
        ){where_sql}
        AND NOT EXISTS (
            SELECT 1 FROM pair_decisions pd
            WHERE pd.ann_a_uuid = pq.ann_a_uuid AND pd.ann_b_uuid = pq.ann_b_uuid
            AND pd.decision IN ('match', 'no_match')
            AND pd.superseded_by IS NULL
        )
        ORDER BY {order_key} ASC
        LIMIT 1
        """,
        [run_id, from_queue_id, *filter_params],
    ).fetchone()
    if row is None:
        return None
    return _pair_from_row(row, storage, total)


def filtered_position_index(
    storage: Storage,
    run_id: str,
    target_queue_id: int,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
    min_td: Optional[int] = None,
    max_td: Optional[int] = None,
    min_km: Optional[float] = None,
    max_km: Optional[float] = None,
    order_by: str = "distance",
    seed: Optional[int] = None,
    undecided_only: bool = True,
) -> int:
    """Return the 0-indexed offset of the pair with queue_id=target_queue_id
    within the filtered+sorted subset — i.e., how many filtered pairs come
    before it under the current sort order."""
    where_sql, filter_params = _build_filter_clauses(
        min_d, max_d, min_td, max_td, min_km, max_km, undecided_only
    )
    order_key = _order_key_expr(order_by, seed)
    order_key_sub = _order_key_expr(order_by, seed, alias="")
    row = storage.conn.execute(
        f"""
        SELECT COUNT(*) FROM pair_queue pq
        WHERE pq.run_id = ? AND {order_key} < (
            SELECT {order_key_sub} FROM pair_queue WHERE queue_id = ?
        ){where_sql}
        """,
        [run_id, target_queue_id, *filter_params],
    ).fetchone()
    return row[0]


def _histogram_from_values(values: list[float], n_bins: int = 20) -> dict:
    n_total = len(values)
    if n_total == 0:
        return {"bins": [0.0, 1.0], "counts": [0], "min": 0.0, "max": 0.0, "n_total": 0}
    vmin = min(values)
    vmax = max(values)
    span = max(vmax - vmin, 1e-6)
    edges = [vmin + (span * i / n_bins) for i in range(n_bins + 1)]
    counts = [0] * n_bins
    for v in values:
        if v >= vmax:
            idx = n_bins - 1
        else:
            idx = int((v - vmin) / span * n_bins)
            idx = max(0, min(idx, n_bins - 1))
        counts[idx] += 1
    return {
        "bins": edges,
        "counts": counts,
        "min": vmin,
        "max": vmax,
        "n_total": n_total,
    }


def get_distance_histogram(storage: Storage, run_id: str, n_bins: int = 20) -> dict:
    """Histogram of pair embedding distances for the given matching run."""
    rows = storage.conn.execute(
        "SELECT distance FROM pair_queue WHERE run_id = ?", (run_id,)
    ).fetchall()
    return _histogram_from_values([r["distance"] for r in rows], n_bins)


def get_time_delta_histogram(storage: Storage, run_id: str, n_bins: int = 20) -> dict:
    """Histogram of |Δdate| in days across pairs in the run. Pairs with a NULL
    time_delta_days (missing date on either side) are excluded from the histogram
    but still counted in n_missing so the UI can disclose them."""
    rows = storage.conn.execute(
        "SELECT time_delta_days FROM pair_queue WHERE run_id = ? AND time_delta_days IS NOT NULL",
        (run_id,),
    ).fetchall()
    hist = _histogram_from_values([r["time_delta_days"] for r in rows], n_bins)
    total_rows = storage.conn.execute(
        "SELECT COUNT(*) FROM pair_queue WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    hist["n_missing"] = total_rows - hist["n_total"]
    return hist


def get_km_delta_histogram(storage: Storage, run_id: str, n_bins: int = 20) -> dict:
    """Histogram of Δkm between pair GPS coordinates across the run. Pairs with
    a NULL km_delta (missing GPS on either side) are excluded but counted as
    n_missing."""
    rows = storage.conn.execute(
        "SELECT km_delta FROM pair_queue WHERE run_id = ? AND km_delta IS NOT NULL",
        (run_id,),
    ).fetchall()
    hist = _histogram_from_values([r["km_delta"] for r in rows], n_bins)
    total_rows = storage.conn.execute(
        "SELECT COUNT(*) FROM pair_queue WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    hist["n_missing"] = total_rows - hist["n_total"]
    return hist
