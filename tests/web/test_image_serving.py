"""Tests for /image/<uuid> endpoint."""
from __future__ import annotations

from starlette.testclient import TestClient

from whaleshark_reid.core.schema import inat_annotation_uuid


def test_serve_full_image(seeded_web_client: TestClient):
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/image/{uuid}")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_serve_cropped_image(seeded_web_client: TestClient):
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/image/{uuid}?crop=true")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")


def test_missing_uuid_returns_404(seeded_web_client: TestClient):
    r = seeded_web_client.get("/image/does-not-exist")
    assert r.status_code == 404
