"""Experiments view routes: runs list, detail, diff."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import experiments as exp_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


@router.get("/experiments", response_class=HTMLResponse)
def experiments_list(
    request: Request,
    storage: Storage = Depends(get_storage),
):
    runs = exp_service.list_runs(storage)
    return templates.TemplateResponse(
        "experiments/index.html",
        {"request": request, "runs": runs},
    )


@router.get("/run/{run_id}", response_class=HTMLResponse)
def run_detail(
    request: Request,
    run_id: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    detail = exp_service.get_run_detail(storage, settings.cache_dir, run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return templates.TemplateResponse(
        "experiments/detail.html",
        {"request": request, "detail": detail},
    )


@router.get("/run/{run_a}/diff/{run_b}", response_class=HTMLResponse)
def run_diff(
    request: Request,
    run_a: str,
    run_b: str,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    diff = exp_service.diff_runs(storage, settings.cache_dir, run_a, run_b)
    if diff is None:
        raise HTTPException(status_code=404, detail="One or both runs not found")
    return templates.TemplateResponse(
        "experiments/diff.html",
        {"request": request, "diff": diff},
    )
