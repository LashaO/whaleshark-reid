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
    # Shared SVG host for lines across the gap (class, not id, so multiple
    # pair cards can coexist on a page without collision)
    assert 'class="local-match-overlay"' in html
    # JS included
    assert 'local_match.js' in html


def test_pair_card_loads_local_match_css_with_cache_bust(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    # existing app.css version bumped OR new file, either way: must appear
    assert "app.css?v=" in page.text


def test_card_is_positioned_ancestor_for_overlay():
    """The overlay is `position: absolute; top:0; left:0; width:100%; height:100%`
    relative to the nearest positioned ancestor. If `.card` is not positioned,
    coordinates collapse to the viewport and lines render in the wrong place
    — silently, since route/template tests still pass.
    """
    from pathlib import Path
    css = (Path(__file__).resolve().parents[2]
           / "src/whaleshark_reid/web/static/css/app.css").read_text()
    # Find the .card rule body and confirm a positioning property is present.
    import re
    m = re.search(r"\.card\s*\{([^}]*)\}", css)
    assert m, ".card rule not found in app.css"
    body = m.group(1)
    assert re.search(r"position\s*:\s*(relative|absolute|fixed|sticky)", body), (
        ".card must establish a positioned containing block so .local-match-overlay "
        "with position:absolute resolves coordinates against the card, not the viewport."
    )
