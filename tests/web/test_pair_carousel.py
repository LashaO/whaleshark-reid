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


def test_decided_pairs_are_hidden_by_default(seeded_web_client: TestClient):
    """After deciding the first pair, it should not reappear in the default
    (undecided-only) carousel — the total should shrink by one."""
    import re
    first = seeded_web_client.get("/review/pairs/r_match")
    before = int(re.search(r"pair \d+/(\d+)", first.text).group(1))
    qid = re.search(r'data-queue-id="(\d+)"', first.text).group(1)

    seeded_web_client.post(
        f"/api/pairs/{qid}/decide",
        data={"decision": "match"},
        headers={"HX-Request": "true"},
    )

    after_default = seeded_web_client.get("/review/pairs/r_match")
    # Either we're on the next pair and total shrank, or we hit the empty state.
    if "pair" in after_default.text and "/" in after_default.text:
        m = re.search(r"pair \d+/(\d+)", after_default.text)
        if m:
            assert int(m.group(1)) == before - 1

    # With undecided_only=0, the original total is visible again.
    show_all = seeded_web_client.get("/review/pairs/r_match?undecided_only=0")
    m = re.search(r"pair \d+/(\d+)", show_all.text)
    if m:
        assert int(m.group(1)) == before


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


def test_filter_clauses_sql(tmp_db_path):
    """Verify _build_filter_clauses generates correct SQL for all filter types."""
    from whaleshark_reid.web.services.pair_queue import _build_filter_clauses

    # No filters → empty where
    w, p = _build_filter_clauses(None, None, None, None)
    assert w == "" and p == []

    # Distance only
    w, p = _build_filter_clauses(0.1, 0.5, None, None)
    assert "pq.distance >= ?" in w and "pq.distance <= ?" in w
    assert p == [0.1, 0.5]

    # Time-delta reads pre-computed pq.time_delta_days (no JOIN needed now)
    w, p = _build_filter_clauses(None, None, 7, 30)
    assert "pq.time_delta_days >= ?" in w and "pq.time_delta_days <= ?" in w
    assert 7 in p and 30 in p

    # Km-delta reads pq.km_delta directly
    w, p = _build_filter_clauses(None, None, None, None, min_km=10.0, max_km=500.0)
    assert "pq.km_delta >= ?" in w and "pq.km_delta <= ?" in w
    assert p == [10.0, 500.0]

    # Combined
    w, p = _build_filter_clauses(0.2, 0.6, 7, None, min_km=5.0)
    assert p == [0.2, 0.6, 7, 5.0]


def test_match_decision_assigns_name_uuid(seeded_web_client: TestClient):
    """Confirming a match should immediately materialize annotations.name_uuid —
    not wait for a manual CLI rebuild. Regression for the 'confirmed match but
    no ID assigned' bug."""
    import re

    page = seeded_web_client.get("/review/pairs/r_match")
    match = re.search(r'data-queue-id="(\d+)"', page.text)
    assert match, "no queue_id in carousel"
    queue_id = int(match.group(1))

    # Find this pair's annotation uuids
    ann_a = re.search(r'data-annotation-uuid-a="([^"]+)"', page.text).group(1)
    ann_b = re.search(r'data-annotation-uuid-b="([^"]+)"', page.text).group(1)

    # Before: neither should have a name_uuid
    from whaleshark_reid.web.dependencies import get_storage
    storage = get_storage()
    a_name = storage.conn.execute(
        "SELECT name_uuid FROM annotations WHERE annotation_uuid = ?", (ann_a,)
    ).fetchone()["name_uuid"]
    assert a_name is None

    # Submit match
    r = seeded_web_client.post(
        f"/api/pairs/{queue_id}/decide",
        data={"decision": "match"},
        headers={"HX-Request": "true"},
    )
    assert r.status_code == 200

    # After: both annotations should share a newly-minted name_uuid
    a_after = storage.conn.execute(
        "SELECT name_uuid FROM annotations WHERE annotation_uuid = ?", (ann_a,)
    ).fetchone()["name_uuid"]
    b_after = storage.conn.execute(
        "SELECT name_uuid FROM annotations WHERE annotation_uuid = ?", (ann_b,)
    ).fetchone()["name_uuid"]
    assert a_after is not None, "ann_a should have name_uuid after match"
    assert a_after == b_after, "both annotations in the pair should share the same name_uuid"


def test_carousel_accepts_km_and_random_params(seeded_web_client: TestClient):
    """All new params — min_km, max_km, order_by, seed — should be accepted."""
    r = seeded_web_client.get("/review/pairs/r_match?max_km=10000")
    assert r.status_code == 200
    r = seeded_web_client.get("/review/pairs/r_match?order_by=random")
    assert r.status_code == 200
    # A random-sort response should advertise a generated seed
    assert "random sort" in r.text
    r = seeded_web_client.get("/review/pairs/r_match?order_by=random&seed=42")
    assert r.status_code == 200
    assert "seed 42" in r.text


def test_haversine_udf_registered(tmp_db_path):
    """Verify the storage layer registers the haversine_km SQL function."""
    from whaleshark_reid.storage.db import Storage
    s = Storage(tmp_db_path)
    # Same point → 0
    row = s.conn.execute("SELECT haversine_km(0, 0, 0, 0)").fetchone()
    assert row[0] == 0.0
    # Approximate NY → LA ~= 3940 km (order-of-magnitude check)
    row = s.conn.execute(
        "SELECT haversine_km(40.7128, -74.0060, 34.0522, -118.2437)"
    ).fetchone()
    assert 3900 < row[0] < 4000
    # NULL propagation
    row = s.conn.execute("SELECT haversine_km(NULL, 0, 0, 0)").fetchone()
    assert row[0] is None


def test_pair_queue_migration_adds_columns(tmp_db_path):
    """Fresh init_schema on an empty DB should produce a pair_queue with
    km_delta and time_delta_days columns."""
    from whaleshark_reid.storage.db import Storage
    s = Storage(tmp_db_path)
    s.init_schema()
    cols = {r["name"] for r in s.conn.execute("PRAGMA table_info(pair_queue)").fetchall()}
    assert "km_delta" in cols
    assert "time_delta_days" in cols


def test_histograms_include_km_and_td(tmp_db_path):
    """The service exposes all three histogram types with expected shape."""
    from whaleshark_reid.storage.db import Storage
    from whaleshark_reid.web.services.pair_queue import (
        get_distance_histogram, get_time_delta_histogram, get_km_delta_histogram,
    )
    s = Storage(tmp_db_path)
    s.init_schema()
    # Empty run → histogram with n_total=0 but consistent shape
    for fn in (get_distance_histogram, get_time_delta_histogram, get_km_delta_histogram):
        h = fn(s, "nonexistent_run")
        assert set(["bins", "counts", "min", "max", "n_total"]).issubset(h.keys())
        assert h["n_total"] == 0
    # km/td histograms also carry n_missing
    assert "n_missing" in get_time_delta_histogram(s, "nonexistent_run")
    assert "n_missing" in get_km_delta_histogram(s, "nonexistent_run")


def test_order_key_random_vs_distance():
    from whaleshark_reid.web.services.pair_queue import _order_key_expr

    assert _order_key_expr("distance", None) == "pq.position"
    assert _order_key_expr("distance", 42) == "pq.position"  # seed ignored for distance
    assert _order_key_expr("random", 42) == "((pq.queue_id * 42) % 1000003)"
    assert _order_key_expr("random", 42, alias="") == "((queue_id * 42) % 1000003)"
    # random without a seed falls back to distance ordering (seed required)
    assert _order_key_expr("random", None) == "pq.position"
