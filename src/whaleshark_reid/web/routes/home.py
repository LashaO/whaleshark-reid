"""Home route — redirects to the latest pair review queue."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_storage

router = APIRouter()


@router.get("/")
def home(storage: Storage = Depends(get_storage)):
    latest = storage.get_latest_run_id("matching")
    if latest:
        return RedirectResponse(url=f"/review/pairs/{latest}")
    return RedirectResponse(url="/experiments")
