"""Annotation listing and detail service."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.core.schema import Annotation
from whaleshark_reid.storage.db import Storage


class AnnotationPage(BaseModel):
    items: list[Annotation]
    total: int
    page: int
    page_size: int


def list_annotations(
    storage: Storage, page: int = 0, page_size: int = 50
) -> AnnotationPage:
    total = storage.count("annotations")
    rows = storage.conn.execute(
        "SELECT * FROM annotations ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (page_size, page * page_size),
    ).fetchall()
    items = [storage._row_to_annotation(r) for r in rows]
    return AnnotationPage(items=items, total=total, page=page, page_size=page_size)


def get_annotation_detail(
    storage: Storage, annotation_uuid: str
) -> Optional[Annotation]:
    return storage.get_annotation(annotation_uuid)
