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


def test_carousel_accepts_time_delta_filter(seeded_web_client: TestClient):
    """The pair carousel should accept min_td/max_td query params and render
    without error. Since fixture pairs may not have dates, filtering is allowed
    to yield zero pairs — in which case the filtered-empty-state renders."""
    r = seeded_web_client.get("/review/pairs/r_match?min_td=30")
    assert r.status_code == 200
    # Either a pair card (data-pair-position) or the filtered empty-state
    assert "data-pair-position" in r.text or "No pairs match the current filter" in r.text


def test_carousel_empty_form_values_do_not_break(seeded_web_client: TestClient):
    """Empty filter fields should coerce to None, not raise validation errors."""
    r = seeded_web_client.get("/review/pairs/r_match?min_td=&max_td=&min_d=&max_d=")
    assert r.status_code == 200


def test_time_delta_filter_sql(tmp_db_path):
    """Verify _build_filter_clauses generates correct SQL for time-delta filter."""
    from whaleshark_reid.web.services.pair_queue import _build_filter_clauses

    # No filters → empty join and where
    j, w, p = _build_filter_clauses(None, None, None, None)
    assert j == "" and w == "" and p == []

    # Only distance → no join
    j, w, p = _build_filter_clauses(0.1, 0.5, None, None)
    assert j == ""
    assert "pq.distance >= ?" in w and "pq.distance <= ?" in w
    assert p == [0.1, 0.5]

    # Time delta → joins annotations twice, filters on julianday delta
    j, w, p = _build_filter_clauses(None, None, 7, 30)
    assert "JOIN annotations a" in j and "JOIN annotations b" in j
    assert "IS NOT NULL" in w  # null-date exclusion
    assert "julianday" in w
    assert 7 in p and 30 in p

    # Combined
    j, w, p = _build_filter_clauses(0.2, 0.6, 7, None)
    assert "JOIN annotations a" in j
    assert p == [0.2, 0.6, 7]
