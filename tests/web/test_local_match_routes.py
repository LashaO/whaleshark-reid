"""Route tests using a fake matcher (no LightGlue required)."""
import re

from starlette.testclient import TestClient

from whaleshark_reid.core.match import lightglue as lg
from whaleshark_reid.core.match.lightglue import MatchResult


def _install_fake(monkeypatch):
    """Replace get_matcher with a deterministic stub."""
    class FakeMatcher:
        def match_pair(self, a, b, **kw):
            return MatchResult(
                extractor="aliked", n_matches=3, mean_score=0.8, median_score=0.8,
                kpts_a=[[1, 2], [3, 4]], kpts_b=[[5, 6], [7, 8]],
                matches=[[0, 0, 0.9], [1, 1, 0.7], [0, 1, 0.3]],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )
    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": FakeMatcher())


def _find_queue_id(html: str) -> int:
    m = re.search(r'data-queue-id="(\d+)"', html)
    assert m, "no data-queue-id in carousel"
    return int(m.group(1))


def test_get_returns_404_when_uncached(seeded_web_client: TestClient):
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)
    r = seeded_web_client.get(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 404


def test_post_runs_matcher_and_get_returns_cached(seeded_web_client: TestClient, monkeypatch):
    _install_fake(monkeypatch)
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 200
    body = r.json()
    assert body["extractor"] == "aliked"
    assert body["n_matches"] == 3
    assert body["matches"][0] == [0, 0, 0.9]

    # Second call hits cache and returns same payload
    r2 = seeded_web_client.get(f"/api/pairs/{qid}/local-match")
    assert r2.status_code == 200
    assert r2.json()["n_matches"] == 3


def test_post_without_overwrite_is_idempotent(seeded_web_client: TestClient, monkeypatch):
    _install_fake(monkeypatch)
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    seeded_web_client.post(f"/api/pairs/{qid}/local-match")

    call_count = {"n": 0}
    class CountingMatcher:
        def match_pair(self, a, b, **kw):
            call_count["n"] += 1
            return MatchResult(
                extractor="aliked", n_matches=99, mean_score=0.1, median_score=0.1,
                kpts_a=[], kpts_b=[], matches=[],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )
    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": CountingMatcher())

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 200
    # Cached, so matcher was NOT called again
    assert call_count["n"] == 0
    assert r.json()["n_matches"] == 3


def test_post_with_overwrite_recomputes(seeded_web_client: TestClient, monkeypatch):
    _install_fake(monkeypatch)
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    seeded_web_client.post(f"/api/pairs/{qid}/local-match")

    class NewerMatcher:
        def match_pair(self, a, b, **kw):
            return MatchResult(
                extractor="aliked", n_matches=42, mean_score=0.2, median_score=0.2,
                kpts_a=[], kpts_b=[], matches=[],
                img_a_size=[440, 440], img_b_size=[440, 440],
            )
    monkeypatch.setattr(lg, "get_matcher", lambda extractor="aliked": NewerMatcher())

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match?overwrite=1")
    assert r.status_code == 200
    assert r.json()["n_matches"] == 42


def test_get_404_for_unknown_queue_id(seeded_web_client: TestClient):
    r = seeded_web_client.get("/api/pairs/999999/local-match")
    assert r.status_code == 404


def test_lightglue_unavailable_returns_503(seeded_web_client: TestClient, monkeypatch):
    from whaleshark_reid.core.match.lightglue import LightGlueUnavailable
    page = seeded_web_client.get("/review/pairs/r_match")
    qid = _find_queue_id(page.text)

    def raiser(extractor="aliked"):
        raise LightGlueUnavailable("not installed")
    monkeypatch.setattr(lg, "get_matcher", raiser)

    r = seeded_web_client.post(f"/api/pairs/{qid}/local-match")
    assert r.status_code == 503
