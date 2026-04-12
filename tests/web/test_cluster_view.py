"""Tests for the cluster scatter view."""
from __future__ import annotations

from starlette.testclient import TestClient


def test_cluster_scatter_renders(seeded_web_client: TestClient):
    r = seeded_web_client.get("/clusters/r_cluster")
    assert r.status_code == 200
    assert "plotly" in r.text.lower() or "scatter" in r.text.lower()


def test_projection_json_endpoint(seeded_web_client: TestClient):
    r = seeded_web_client.get("/api/projections/r_project")
    assert r.status_code == 200
    data = r.json()
    assert "points" in data
    assert len(data["points"]) == 10
    for p in data["points"]:
        assert "x" in p and "y" in p and "annotation_uuid" in p
