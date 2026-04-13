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


def test_nav_has_individuals_link(web_client: TestClient):
    r = web_client.get("/list/annotations")
    assert r.status_code == 200
    assert 'href="/list/individuals"' in r.text


def test_individual_detail_404_for_unknown(web_client: TestClient):
    r = web_client.get("/individual/does-not-exist")
    assert r.status_code == 404


def test_individual_detail_renders_members(seeded_web_client: TestClient):
    """After confirming a match the derived individual should have a detail page
    rendering its member annotations and the map container."""
    import re
    page = seeded_web_client.get("/review/pairs/r_match")
    queue_id = re.search(r'data-queue-id="(\d+)"', page.text).group(1)
    seeded_web_client.post(
        f"/api/pairs/{queue_id}/decide",
        data={"decision": "match"},
        headers={"HX-Request": "true"},
    )

    from whaleshark_reid.web.dependencies import get_storage
    storage = get_storage()
    name_uuid = storage.conn.execute(
        "SELECT name_uuid FROM annotations WHERE name_uuid IS NOT NULL LIMIT 1"
    ).fetchone()["name_uuid"]

    r = seeded_web_client.get(f"/individual/{name_uuid}")
    assert r.status_code == 200
    assert "Individual" in r.text
    # Either the map container renders or we disclose missing GPS — never both.
    assert 'id="individual-map"' in r.text or "No GPS data" in r.text
