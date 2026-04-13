"""Confirm the pair card exposes the DOM hooks the JS module relies on."""
from starlette.testclient import TestClient


def test_pair_card_has_chip_wrap_hooks(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    assert page.status_code == 200
    html = page.text
    # Both chips wrapped for overlay positioning
    assert html.count('class="chip-wrap"') >= 2 or html.count("chip-wrap") >= 2
    # Controls container
    assert 'class="local-match-controls"' in html
    # Shared SVG host for lines across the gap
    assert 'id="local-match-overlay"' in html
    # JS included
    assert 'local_match.js' in html


def test_pair_card_loads_local_match_css_with_cache_bust(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    # existing app.css version bumped OR new file, either way: must appear
    assert "app.css?v=" in page.text
