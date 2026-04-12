"""Cross-cutting tests: HTMX fragments, keyboard shortcuts, full page structure."""
from __future__ import annotations

import re

from starlette.testclient import TestClient


def test_pair_card_has_keyboard_shortcut_attrs(seeded_web_client: TestClient):
    r = seeded_web_client.get("/review/pairs/r_match")
    assert r.status_code == 200
    # All four decision buttons should have data-shortcut
    for key in ["y", "n", "u", " "]:
        assert f'data-shortcut="{key}"' in r.text, f"Missing data-shortcut for key '{key}'"


def test_htmx_decide_returns_fragment_not_full_page(seeded_web_client: TestClient):
    # Get the first pair's queue_id
    page = seeded_web_client.get("/review/pairs/r_match")
    match = re.search(r'data-queue-id="(\d+)"', page.text)
    assert match
    queue_id = match.group(1)

    # HTMX POST should return a fragment (no <html> tag)
    r = seeded_web_client.post(
        f"/api/pairs/{queue_id}/decide",
        data={"decision": "skip"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    assert "<html" not in r.text


def test_all_nav_links_return_200(seeded_web_client: TestClient):
    """Verify every link in the nav bar returns a non-500 response."""
    pages = [
        "/review/pairs/r_match",
        "/list/annotations",
        "/list/decisions",
        "/list/individuals",
        "/clusters/r_cluster",
        "/experiments",
        "/map",
        "/health",
    ]
    for url in pages:
        r = seeded_web_client.get(url)
        assert r.status_code < 500, f"{url} returned {r.status_code}"


def test_image_crop_in_carousel_returns_jpeg(seeded_web_client: TestClient):
    from whaleshark_reid.core.schema import inat_annotation_uuid
    uuid = inat_annotation_uuid(100, 0)
    r = seeded_web_client.get(f"/image/{uuid}?crop=true")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
