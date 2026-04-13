"""GET + POST /api/pairs/{queue_id}/pairx — A/B PairX visualization.

Filesystem-cached PNG. GET serves the file (404 if absent). POST runs PairX
on demand, writes the PNG, returns it. Layer key is configurable via query.
"""
from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.web.dependencies import get_settings, get_storage
from whaleshark_reid.web.services import pairx as svc
from whaleshark_reid.web.settings import Settings

router = APIRouter()

DEFAULT_LAYER = "backbone.blocks.3"


@router.get("/api/pairs/{queue_id}/pairx.png")
def get_pairx_png(
    queue_id: int,
    layer: str = DEFAULT_LAYER,
    settings: Settings = Depends(get_settings),
):
    p = svc.png_path(settings.cache_dir, queue_id, layer)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not cached")
    return Response(p.read_bytes(), media_type="image/png")


@router.post("/api/pairs/{queue_id}/pairx.png")
def post_pairx(
    queue_id: int,
    layer: str = DEFAULT_LAYER,
    k_lines: int = 20,
    k_colors: int = 5,
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
):
    try:
        spec_a, spec_b = svc.lookup_pair_chip_specs(storage, queue_id)
    except LookupError:
        raise HTTPException(status_code=404, detail="pair not found")

    try:
        from whaleshark_reid.core.explain.pairx import PairXUnavailable, explain_pair
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"PairX import failed: {e}")

    pa, ba, tha = spec_a
    pb, bb, thb = spec_b
    try:
        images = explain_pair(
            pa, ba, tha, pb, bb, thb,
            layer_key=layer, k_lines=k_lines, k_colors=k_colors,
        )
    except PairXUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    if not images:
        raise HTTPException(status_code=500, detail="PairX returned no images")

    # Encode the first (and usually only) layer's image to PNG.
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(images[0]).save(buf, format="PNG")
    png_bytes = buf.getvalue()
    svc.write_png(settings.cache_dir, queue_id, layer, png_bytes)
    return Response(png_bytes, media_type="image/png")
