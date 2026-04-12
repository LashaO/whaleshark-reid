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
