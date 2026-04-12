"""Pair review carousel routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import pair_queue as pq_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


@router.get("/review/pairs")
@router.get("/review/pairs/")
def review_pairs_index(storage: Storage = Depends(get_storage)):
    """Redirect to the latest matching run's carousel, or to /experiments if none exists."""
    latest = storage.get_latest_run_id("matching")
    if latest:
        return RedirectResponse(url=f"/review/pairs/{latest}", status_code=307)
    return RedirectResponse(url="/experiments", status_code=307)


@router.get("/review/pairs/{run_id}", response_class=HTMLResponse)
def carousel(
    request: Request,
    run_id: str,
    position: int = 0,
    storage: Storage = Depends(get_storage),
):
    pair = pq_service.get_pair(storage, run_id, position)
    if pair is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": run_id},
        )
    return templates.TemplateResponse(
        "pairs/carousel.html",
        {"request": request, "pair": pair},
    )


@router.post("/api/pairs/{queue_id}/decide", response_class=HTMLResponse)
def decide(
    request: Request,
    queue_id: int,
    decision: str = Form(...),
    notes: str = Form(""),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    current = pq_service.get_pair_by_id(storage, queue_id)
    if current is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": "unknown"},
        )

    pq_service.submit_decision(storage, queue_id, decision, settings.user, notes)

    next_pair = pq_service.get_next_undecided(
        storage, run_id=current.run_id, from_position=current.position + 1
    )
    if next_pair is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": current.run_id},
        )
    return templates.TemplateResponse(
        "partials/pair_card.html",
        {"request": request, "pair": next_pair},
    )
