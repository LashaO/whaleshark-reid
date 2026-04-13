"""Pair review carousel routes."""
from __future__ import annotations

import random
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


def _resolve_order(order_by: Optional[str], seed: Optional[int]) -> tuple[str, Optional[int]]:
    """Normalize sort params. 'random' without a seed gets a fresh one
    (reseeding per request — user asked for reseed-on-apply behavior)."""
    order_by = (order_by or "distance").strip().lower()
    if order_by not in ("distance", "random"):
        order_by = "distance"
    if order_by == "random":
        if seed is None:
            seed = random.randint(1, 1_000_000)
    else:
        seed = None  # irrelevant for distance ordering
    return order_by, seed


def _filter_active(min_d, max_d, min_td, max_td, min_km, max_km) -> bool:
    return any(v is not None for v in (min_d, max_d, min_td, max_td, min_km, max_km))


def _build_histograms(storage: Storage, run_id: str) -> dict:
    return {
        "distance": pq_service.get_distance_histogram(storage, run_id),
        "time_delta": pq_service.get_time_delta_histogram(storage, run_id),
        "km_delta": pq_service.get_km_delta_histogram(storage, run_id),
    }


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
    min_km: Optional[str] = None,
    max_km: Optional[str] = None,
    order_by: Optional[str] = None,
    seed: Optional[str] = None,
    storage: Storage = Depends(get_storage),
):
    min_d = _parse_optional_float(min_d)
    max_d = _parse_optional_float(max_d)
    min_td = _parse_optional_int(min_td)
    max_td = _parse_optional_int(max_td)
    min_km = _parse_optional_float(min_km)
    max_km = _parse_optional_float(max_km)
    order_by, seed = _resolve_order(order_by, _parse_optional_int(seed))

    pair = pq_service.get_pair(
        storage, run_id, position,
        min_d=min_d, max_d=max_d,
        min_td=min_td, max_td=max_td,
        min_km=min_km, max_km=max_km,
        order_by=order_by, seed=seed,
    )
    histograms = _build_histograms(storage, run_id)
    filter_active = _filter_active(min_d, max_d, min_td, max_td, min_km, max_km)
    ctx = {
        "request": request,
        "run_id": run_id,
        "histograms": histograms,
        "min_d": min_d if min_d is not None else histograms["distance"]["min"],
        "max_d": max_d if max_d is not None else histograms["distance"]["max"],
        "min_td": min_td,
        "max_td": max_td,
        "min_km": min_km,
        "max_km": max_km,
        "order_by": order_by,
        "seed": seed,
        "filter_active": filter_active,
    }
    if pair is None:
        return templates.TemplateResponse("partials/empty_queue.html", ctx)
    return templates.TemplateResponse(
        "pairs/carousel.html",
        {**ctx, "pair": pair, "view_position": position},
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
    min_km: Optional[str] = Form(None),
    max_km: Optional[str] = Form(None),
    order_by: Optional[str] = Form(None),
    seed: Optional[str] = Form(None),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    min_d = _parse_optional_float(min_d)
    max_d = _parse_optional_float(max_d)
    min_td = _parse_optional_int(min_td)
    max_td = _parse_optional_int(max_td)
    min_km = _parse_optional_float(min_km)
    max_km = _parse_optional_float(max_km)
    order_by, seed = _resolve_order(order_by, _parse_optional_int(seed))

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
        min_d=min_d, max_d=max_d,
        min_td=min_td, max_td=max_td,
        min_km=min_km, max_km=max_km,
        order_by=order_by, seed=seed,
    )
    histograms = _build_histograms(storage, current.run_id)
    filter_active = _filter_active(min_d, max_d, min_td, max_td, min_km, max_km)
    ctx = {
        "request": request,
        "run_id": current.run_id,
        "histograms": histograms,
        "min_d": min_d if min_d is not None else histograms["distance"]["min"],
        "max_d": max_d if max_d is not None else histograms["distance"]["max"],
        "min_td": min_td,
        "max_td": max_td,
        "min_km": min_km,
        "max_km": max_km,
        "order_by": order_by,
        "seed": seed,
        "filter_active": filter_active,
    }
    if next_pair is None:
        return templates.TemplateResponse("partials/empty_queue.html", ctx)

    view_position = pq_service.filtered_position_index(
        storage, current.run_id, next_pair.queue_id,
        min_d=min_d, max_d=max_d,
        min_td=min_td, max_td=max_td,
        min_km=min_km, max_km=max_km,
        order_by=order_by, seed=seed,
    )
    return templates.TemplateResponse(
        "partials/pair_card.html",
        {**ctx, "pair": next_pair, "view_position": view_position},
    )
