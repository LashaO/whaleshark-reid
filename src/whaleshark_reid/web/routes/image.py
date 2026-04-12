"""Image serving route: /image/<annotation_uuid>."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_storage
from whaleshark_reid.web.services import images as images_service

router = APIRouter()


@router.get("/image/{annotation_uuid}")
def serve_image(
    annotation_uuid: str,
    crop: bool = False,
    storage: Storage = Depends(get_storage),
):
    try:
        body, content_type = images_service.serve_annotation_image(
            storage, annotation_uuid, crop=crop
        )
        return Response(content=body, media_type=content_type)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found")
