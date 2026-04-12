"""Tests for the experiments view — list, detail, diff."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_experiments_list_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/experiments")
    assert r.status_code == 200
    assert "r_embed" in r.text or "embed" in r.text


def test_run_detail_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/run/r_embed")
    assert r.status_code == 200
    assert "embed" in r.text


def test_run_diff_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/run/r_embed/diff/r_cluster")
    assert r.status_code == 200
