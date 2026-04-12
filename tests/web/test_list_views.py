"""Tests for list views — annotations, decisions, individuals."""
from __future__ import annotations

from starlette.testclient import TestClient

from whaleshark_reid.core.schema import inat_annotation_uuid


def test_annotations_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/list/annotations")
    assert r.status_code == 200
    assert "annotation_uuid" in r.text.lower() or "Annotations" in r.text


def test_annotation_detail_renders(seeded_web_client: TestClient):
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/annotation/{uuid}")
    assert r.status_code == 200
    assert uuid[:12] in r.text


def test_annotation_detail_404_for_missing(seeded_web_client: TestClient):
    r = seeded_web_client.get("/annotation/does-not-exist")
    assert r.status_code == 404
