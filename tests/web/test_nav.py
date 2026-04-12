"""Tests that all nav links work (no 500s, no dead links)."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_home_redirects(seeded_web_client: TestClient):
    r = seeded_web_client.get("/", follow_redirects=False)
    assert r.status_code in (301, 302, 307, 308)


def test_map_stub_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/map")
    assert r.status_code == 200
    assert "Phase" in r.text or "coming" in r.text.lower()


def test_decisions_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/list/decisions")
    assert r.status_code == 200


def test_individuals_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/list/individuals")
    assert r.status_code == 200
