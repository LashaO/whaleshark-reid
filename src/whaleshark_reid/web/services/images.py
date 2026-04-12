"""Image serving: load annotation image, optionally crop to bbox+theta."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

from whaleshark_reid.storage.db import Storage


@lru_cache(maxsize=256)
def _load_and_crop(file_path: str, bbox_tuple: tuple, theta: float, crop: bool) -> bytes:
    img = Image.open(file_path).convert("RGB")

    if crop and bbox_tuple:
        try:
            from wbia_miew_id.datasets.helpers import get_chip_from_img
            img_arr = np.array(img)
            bbox = list(bbox_tuple)
            chip = get_chip_from_img(img_arr, bbox, theta)
            img = Image.fromarray(chip)
        except Exception:
            pass  # fallback to full image on any crop error

    # Resize to max 800px on longest side
    max_dim = max(img.size)
    if max_dim > 800:
        scale = 800 / max_dim
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def serve_annotation_image(
    storage: Storage,
    annotation_uuid: str,
    crop: bool = False,
) -> tuple[bytes, str]:
    """Returns (jpeg_bytes, content_type). Raises FileNotFoundError if annotation or image missing."""
    ann = storage.get_annotation(annotation_uuid)
    if ann is None:
        raise FileNotFoundError(f"No annotation with uuid {annotation_uuid}")

    file_path = ann.file_path
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    bbox_tuple = tuple(ann.bbox) if ann.bbox else ()
    jpeg_bytes = _load_and_crop(file_path, bbox_tuple, ann.theta, crop)
    return jpeg_bytes, "image/jpeg"
