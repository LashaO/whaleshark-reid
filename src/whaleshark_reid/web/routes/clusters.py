"""Cluster scatter view routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import cluster_view as cv_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


@router.get("/clusters")
@router.get("/clusters/")
def clusters_index(storage: Storage = Depends(get_storage)):
    """Redirect to the latest cluster run's scatter page, or to /experiments if none exists."""
    latest = storage.get_latest_run_id("cluster")
    if latest:
        return RedirectResponse(url=f"/clusters/{latest}", status_code=307)
    return RedirectResponse(url="/experiments", status_code=307)


@router.get("/clusters/{run_id}", response_class=HTMLResponse)
def cluster_scatter(
    request: Request,
    run_id: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    proj_run = storage.get_latest_run_id("project")
    return templates.TemplateResponse(
        "clusters/scatter.html",
        {"request": request, "cluster_run_id": run_id, "proj_run_id": proj_run or run_id},
    )


@router.get("/api/projections/{run_id}")
def projection_json(
    run_id: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    cluster_run = storage.get_latest_run_id("cluster")
    proj = cv_service.get_projection(
        settings.cache_dir, run_id, cluster_run or run_id
    )
    if proj is None:
        return JSONResponse({"points": []})
    return proj.model_dump()
