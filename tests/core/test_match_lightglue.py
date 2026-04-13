"""Unit tests for core.match.lightglue that do NOT require the real LightGlue install.

Anything that calls the real extractor/matcher is marked `@pytest.mark.lightglue`
and skipped by default; CI/GPU runs can opt in with `-m lightglue`.
"""
import json

import pytest

from whaleshark_reid.core.match.lightglue import MatchResult


def test_match_result_roundtrip_json():
    mr = MatchResult(
        extractor="aliked",
        n_matches=2,
        mean_score=0.6,
        median_score=0.6,
        kpts_a=[[1.0, 2.0], [3.0, 4.0]],
        kpts_b=[[5.0, 6.0], [7.0, 8.0]],
        matches=[[0, 0, 0.8], [1, 1, 0.4]],
        img_a_size=[440, 440],
        img_b_size=[440, 440],
    )
    blob = mr.to_json_dict()
    assert set(blob.keys()) == {
        "extractor", "n_matches", "mean_score", "median_score",
        "kpts_a", "kpts_b", "matches", "img_a_size", "img_b_size",
    }
    # Round trip through real JSON
    roundtripped = MatchResult.from_json_dict(json.loads(json.dumps(blob)))
    assert roundtripped == mr


def test_n_matches_counts_confidence_threshold():
    """n_matches counts score >= threshold."""
    from whaleshark_reid.core.match.lightglue import count_confident_matches
    assert count_confident_matches([[0, 0, 0.8], [1, 1, 0.4], [2, 2, 0.5]], thr=0.5) == 2
    assert count_confident_matches([], thr=0.5) == 0
