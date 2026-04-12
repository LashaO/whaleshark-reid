"""Map stub route."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from whaleshark_reid.web.app import templates

router = APIRouter()


@router.get("/map", response_class=HTMLResponse)
def map_stub(request: Request):
    return templates.TemplateResponse("map/stub.html", {"request": request})
