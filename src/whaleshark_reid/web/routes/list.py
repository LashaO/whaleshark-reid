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
def decisions_list(request: Request, storage: Storage = Depends(get_storage)):
    rows = storage.conn.execute(
        "SELECT * FROM pair_decisions ORDER BY created_at DESC LIMIT 100"
    ).fetchall()
    return templates.TemplateResponse(
        "list/decisions.html",
        {"request": request, "decisions": [dict(r) for r in rows]},
    )


@router.get("/list/individuals", response_class=HTMLResponse)
def individuals_list(request: Request, storage: Storage = Depends(get_storage)):
    rows = storage.conn.execute(
        """
        SELECT name_uuid, name, COUNT(*) as member_count
        FROM annotations
        WHERE name_uuid IS NOT NULL
        GROUP BY name_uuid
        ORDER BY member_count DESC
        """
    ).fetchall()
    return templates.TemplateResponse(
        "list/individuals.html",
        {"request": request, "individuals": [dict(r) for r in rows]},
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
