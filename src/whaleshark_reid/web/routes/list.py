"""List views: annotations, decisions, individuals."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_storage
from whaleshark_reid.web.services import annotations as ann_service

router = APIRouter()


@router.get("/list/annotations", response_class=HTMLResponse)
def annotations_list(
    request: Request,
    page: int = 0,
    storage: Storage = Depends(get_storage),
):
    result = ann_service.list_annotations(storage, page=page)
    return templates.TemplateResponse(
        "list/annotations.html",
        {"request": request, "result": result},
    )


@router.get("/annotation/{annotation_uuid}", response_class=HTMLResponse)
def annotation_detail(
    request: Request,
    annotation_uuid: str,
    storage: Storage = Depends(get_storage),
):
    ann = ann_service.get_annotation_detail(storage, annotation_uuid)
    if ann is None:
        raise HTTPException(status_code=404, detail="Annotation not found")
    return templates.TemplateResponse(
        "list/annotation_detail.html",
        {"request": request, "ann": ann},
    )


@router.get("/list/decisions", response_class=HTMLResponse)
def decisions_list(
    request: Request,
    page: int = 0,
    decision: str = "all",
    storage: Storage = Depends(get_storage),
):
    page_size = 50
    where_parts = ["superseded_by IS NULL"]
    params: list = []
    if decision in ("match", "no_match", "unsure", "skip"):
        where_parts.append("decision = ?")
        params.append(decision)
    where_sql = " AND ".join(where_parts)

    total = storage.conn.execute(
        f"SELECT COUNT(*) FROM pair_decisions WHERE {where_sql}", params
    ).fetchone()[0]

    rows = storage.conn.execute(
        f"""
        SELECT * FROM pair_decisions
        WHERE {where_sql}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        [*params, page_size, page * page_size],
    ).fetchall()

    counts = {
        r["decision"]: r["n"]
        for r in storage.conn.execute(
            """
            SELECT decision, COUNT(*) as n FROM pair_decisions
            WHERE superseded_by IS NULL
            GROUP BY decision
            """
        ).fetchall()
    }

    return templates.TemplateResponse(
        "list/decisions.html",
        {
            "request": request,
            "decisions": [dict(r) for r in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "n_pages": max(1, (total + page_size - 1) // page_size),
            "filter_decision": decision,
            "counts_by_decision": counts,
        },
    )


@router.get("/list/individuals", response_class=HTMLResponse)
def individuals_list(request: Request, storage: Storage = Depends(get_storage)):
    # Per-individual stats. Max time spread is cheap (max-min of dates). Max km
    # spread requires a pairwise self-join — but only within each group, so
    # cost is O(Σ k_i²) across individuals, tiny for real-world counts.
    base = storage.conn.execute(
        """
        SELECT
            name_uuid,
            MAX(name)                                                   AS name,
            COUNT(*)                                                    AS member_count,
            CAST(MAX(julianday(date_captured)) - MIN(julianday(date_captured)) AS INTEGER) AS max_td_days
        FROM annotations
        WHERE name_uuid IS NOT NULL
        GROUP BY name_uuid
        """
    ).fetchall()

    km_spread = {
        r["name_uuid"]: r["max_km"]
        for r in storage.conn.execute(
            """
            SELECT
                a.name_uuid,
                MAX(haversine_km(a.gps_lat_captured, a.gps_lon_captured,
                                 b.gps_lat_captured, b.gps_lon_captured)) AS max_km
            FROM annotations a
            JOIN annotations b
              ON a.name_uuid = b.name_uuid
             AND a.annotation_uuid < b.annotation_uuid
            WHERE a.name_uuid IS NOT NULL
            GROUP BY a.name_uuid
            """
        ).fetchall()
    }

    individuals = []
    for r in base:
        d = dict(r)
        d["max_km"] = km_spread.get(d["name_uuid"])
        individuals.append(d)
    # Primary sort: member count desc. Secondary: max_td_days desc (interesting resightings first).
    individuals.sort(key=lambda d: (d["member_count"], d["max_td_days"] or 0), reverse=True)

    return templates.TemplateResponse(
        "list/individuals.html",
        {"request": request, "individuals": individuals},
    )


@router.get("/individual/{name_uuid}", response_class=HTMLResponse)
def individual_detail(
    request: Request,
    name_uuid: str,
    storage: Storage = Depends(get_storage),
):
    """Show all annotations that make up a single derived individual — with
    images, metadata, and a multi-point map."""
    rows = storage.conn.execute(
        """
        SELECT * FROM annotations
        WHERE name_uuid = ?
        ORDER BY date_captured NULLS LAST, annotation_uuid
        """,
        (name_uuid,),
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Individual not found")

    members = [dict(r) for r in rows]
    # GPS points for the map. Deduplicate identical coordinates so overlapping
    # observations don't stack identical markers.
    gps_points = [
        {
            "lat": m["gps_lat_captured"],
            "lon": m["gps_lon_captured"],
            "uuid": m["annotation_uuid"],
            "date": m["date_captured"],
        }
        for m in members
        if m["gps_lat_captured"] is not None and m["gps_lon_captured"] is not None
    ]
    return templates.TemplateResponse(
        "list/individual_detail.html",
        {
            "request": request,
            "name_uuid": name_uuid,
            "name": members[0]["name"],
            "members": members,
            "gps_points": gps_points,
        },
    )
