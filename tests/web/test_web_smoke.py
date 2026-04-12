"""Smoke tests for the web app factory."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_health_endpoint(web_client: TestClient):
    r = web_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_static_css_accessible(web_client: TestClient):
    r = web_client.get("/static/css/app.css")
    assert r.status_code == 200
    assert "var(--bg)" in r.text


def test_static_htmx_accessible(web_client: TestClient):
    r = web_client.get("/static/vendor/htmx.min.js")
    assert r.status_code == 200
