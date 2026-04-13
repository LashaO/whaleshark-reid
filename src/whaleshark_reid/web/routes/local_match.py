"""GET + POST /api/pairs/{queue_id}/local-match."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from whaleshark_reid.core.match import lightglue as lg_module
from whaleshark_reid.core.match.lightglue import LightGlueUnavailable
from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_storage
from whaleshark_reid.web.services import local_match as svc

router = APIRouter()


@router.get("/api/pairs/{queue_id}/local-match")
def get_local_match(
    queue_id: int,
    extractor: str = "aliked",
    storage: Storage = Depends(get_storage),
):
    # Verify the pair exists before answering cache miss vs unknown
    try:
        svc.lookup_pair_image_paths(storage, queue_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="pair not found")

    cached = svc.read_cached(storage, queue_id, extractor)
    if cached is None:
        raise HTTPException(status_code=404, detail="not cached")
    return cached.to_json_dict()


@router.post("/api/pairs/{queue_id}/local-match")
def post_local_match(
    queue_id: int,
    extractor: str = "aliked",
    overwrite: int = 0,
    storage: Storage = Depends(get_storage),
):
    try:
        path_a, path_b = svc.lookup_pair_image_paths(storage, queue_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="pair not found")

    if not overwrite:
        cached = svc.read_cached(storage, queue_id, extractor)
        if cached is not None:
            return cached.to_json_dict()

    try:
        matcher = lg_module.get_matcher(extractor)
    except LightGlueUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))

    result = matcher.match_pair(path_a, path_b)
    svc.write_cached(storage, queue_id, result)
    return result.to_json_dict()
