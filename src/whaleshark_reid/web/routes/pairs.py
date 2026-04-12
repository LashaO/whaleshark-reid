"""Pair review carousel routes."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.app import templates
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import pair_queue as pq_service
from whaleshark_reid.web.settings import Settings

router = APIRouter()


def _parse_optional_int(raw: Optional[str]) -> Optional[int]:
    """Coerce empty/whitespace form values to None; parse valid ints."""
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return int(raw)


def _parse_optional_float(raw: Optional[str]) -> Optional[float]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return float(raw)


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
    min_d: Optional[str] = None,
    max_d: Optional[str] = None,
    min_td: Optional[str] = None,
    max_td: Optional[str] = None,
    storage: Storage = Depends(get_storage),
):
    min_d = _parse_optional_float(min_d)
    max_d = _parse_optional_float(max_d)
    min_td = _parse_optional_int(min_td)
    max_td = _parse_optional_int(max_td)
    pair = pq_service.get_pair(
        storage, run_id, position,
        min_d=min_d, max_d=max_d, min_td=min_td, max_td=max_td,
    )
    histogram = pq_service.get_distance_histogram(storage, run_id)
    filter_active = any(v is not None for v in (min_d, max_d, min_td, max_td))
    if pair is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {
                "request": request,
                "run_id": run_id,
                "histogram": histogram,
                "min_d": min_d,
                "max_d": max_d,
                "min_td": min_td,
                "max_td": max_td,
                "filter_active": filter_active,
            },
        )
    return templates.TemplateResponse(
        "pairs/carousel.html",
        {
            "request": request,
            "pair": pair,
            "histogram": histogram,
            "min_d": min_d if min_d is not None else histogram["min"],
            "max_d": max_d if max_d is not None else histogram["max"],
            "min_td": min_td,
            "max_td": max_td,
            "filter_active": filter_active,
            "view_position": position,  # offset within the filtered subset
        },
    )


@router.post("/api/pairs/{queue_id}/decide", response_class=HTMLResponse)
def decide(
    request: Request,
    queue_id: int,
    decision: str = Form(...),
    notes: str = Form(""),
    min_d: Optional[str] = Form(None),
    max_d: Optional[str] = Form(None),
    min_td: Optional[str] = Form(None),
    max_td: Optional[str] = Form(None),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    min_d = _parse_optional_float(min_d)
    max_d = _parse_optional_float(max_d)
    min_td = _parse_optional_int(min_td)
    max_td = _parse_optional_int(max_td)
    current = pq_service.get_pair_by_id(storage, queue_id)
    if current is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {"request": request, "run_id": "unknown"},
        )

    pq_service.submit_decision(storage, queue_id, decision, settings.user, notes)

    next_pair = pq_service.get_next_undecided(
        storage,
        run_id=current.run_id,
        from_queue_id=queue_id,
        min_d=min_d,
        max_d=max_d,
        min_td=min_td,
        max_td=max_td,
    )
    histogram = pq_service.get_distance_histogram(storage, current.run_id)
    filter_active = any(v is not None for v in (min_d, max_d, min_td, max_td))
    if next_pair is None:
        return templates.TemplateResponse(
            "partials/empty_queue.html",
            {
                "request": request,
                "run_id": current.run_id,
                "histogram": histogram,
                "min_d": min_d,
                "max_d": max_d,
                "min_td": min_td,
                "max_td": max_td,
                "filter_active": filter_active,
            },
        )
    # Compute the filtered-subset offset for the next pair so the UI counter stays accurate
    view_position = pq_service.filtered_position_index(
        storage, current.run_id, next_pair.position,
        min_d=min_d, max_d=max_d, min_td=min_td, max_td=max_td,
    )
    return templates.TemplateResponse(
        "partials/pair_card.html",
        {
            "request": request,
            "pair": next_pair,
            "histogram": histogram,
            "min_d": min_d if min_d is not None else histogram["min"],
            "max_d": max_d if max_d is not None else histogram["max"],
            "min_td": min_td,
            "max_td": max_td,
            "filter_active": filter_active,
            "view_position": view_position,
        },
    )
