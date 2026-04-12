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


def get_pair(storage: Storage, run_id: str, position: int) -> Optional[PairView]:
    total = storage.count("pair_queue", run_id=run_id)
    if total == 0:
        return None
    row = storage.conn.execute(
        "SELECT * FROM pair_queue WHERE run_id = ? AND position = ?",
        (run_id, position),
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
    storage: Storage, run_id: str, from_position: int
) -> Optional[PairView]:
    total = storage.count("pair_queue", run_id=run_id)
    row = storage.conn.execute(
        """
        SELECT pq.* FROM pair_queue pq
        WHERE pq.run_id = ? AND pq.position >= ?
        AND NOT EXISTS (
            SELECT 1 FROM pair_decisions pd
            WHERE pd.ann_a_uuid = pq.ann_a_uuid AND pd.ann_b_uuid = pq.ann_b_uuid
            AND pd.decision IN ('match', 'no_match')
            AND pd.superseded_by IS NULL
        )
        ORDER BY pq.position ASC
        LIMIT 1
        """,
        (run_id, from_position),
    ).fetchone()
    if row is None:
        return None
    return _pair_from_row(row, storage, total)
