"""Tests for the pair carousel page and HTMX decision flow."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_carousel_renders_first_pair(seeded_web_client: TestClient):
    r = seeded_web_client.get("/review/pairs/r_match")
    assert r.status_code == 200
    assert "data-pair-position" in r.text
    assert "data-distance" in r.text
    # Decision buttons present
    assert "Match" in r.text
    assert "No match" in r.text


def test_submit_decision_returns_next_pair(seeded_web_client: TestClient):
    # Get the first pair to find the queue_id
    page = seeded_web_client.get("/review/pairs/r_match")
    assert page.status_code == 200

    # Extract queue_id from HTML — look for data-queue-id attribute
    import re
    match = re.search(r'data-queue-id="(\d+)"', page.text)
    assert match, "No data-queue-id found in carousel HTML"
    queue_id = match.group(1)

    # Submit a decision via HTMX POST
    r = seeded_web_client.post(
        f"/api/pairs/{queue_id}/decide",
        data={"decision": "match"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200
    # HTMX fragment — no full <html> tag
    assert "<html" not in r.text
    # Should be either a new pair_card or empty_queue
    assert "data-pair-position" in r.text or "No pairs" in r.text


def test_empty_queue_renders_gracefully(web_client: TestClient):
    # No matching run → should handle gracefully
    r = web_client.get("/review/pairs/nonexistent")
    assert r.status_code == 200
    assert "No pairs" in r.text or "no pairs" in r.text.lower()
