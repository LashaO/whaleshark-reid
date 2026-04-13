"""ALIKED + LightGlue local feature matching for the pair review UI.

This module isolates the only code path that imports `lightglue`. Everything
else in the project can be unaware of it. Import failure (the optional extra
isn't installed) raises LightGlueUnavailable only when the matcher is actually
requested — not at module import.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable


class LightGlueUnavailable(RuntimeError):
    """Raised when LightGlue is not installed and the matcher is requested."""


@dataclass
class MatchResult:
    extractor: str
    n_matches: int
    mean_score: float | None
    median_score: float | None
    kpts_a: list[list[float]]
    kpts_b: list[list[float]]
    matches: list[list[float]]  # [i, j, score]
    img_a_size: list[int]       # [w, h] — coord space of kpts_a
    img_b_size: list[int]

    def to_json_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, d: dict) -> "MatchResult":
        return cls(**d)


def count_confident_matches(matches: Iterable[list[float]], thr: float = 0.5) -> int:
    return sum(1 for m in matches if m[2] >= thr)


# ---- Matcher singleton / seam ------------------------------------------------
#
# Tests inject a fake via monkeypatch on `_build_matcher` or `get_matcher`.
# Real matcher construction lives in _build_matcher, which imports lightglue
# lazily so the module stays importable without the optional dep.

_MATCHER_CACHE: dict[str, "LocalMatcher"] = {}


class LocalMatcher:
    """ALIKED+LightGlue (or SuperPoint+LightGlue, etc.) bound to a device."""

    def __init__(self, extractor: str):
        import torch

        try:
            from lightglue import ALIKED, SuperPoint, DISK, LightGlue
            from lightglue.utils import load_image, rbd
        except ImportError as e:
            raise LightGlueUnavailable(
                "LightGlue is not installed. Install the optional extra: "
                "`pip install 'whaleshark-reid[local-match]'`"
            ) from e

        self.extractor_name = extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._load_image = load_image
        self._rbd = rbd

        if extractor == "aliked":
            self.extractor = ALIKED(max_num_keypoints=2048).eval().to(self.device)
            features = "aliked"
        elif extractor == "superpoint":
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            features = "superpoint"
        elif extractor == "disk":
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
            features = "disk"
        else:
            raise ValueError(f"unknown extractor: {extractor!r}")
        self.matcher = LightGlue(features=features).eval().to(self.device)

    @staticmethod
    def _statsf(scores: list[float]) -> tuple[float | None, float | None]:
        if not scores:
            return None, None
        xs = sorted(scores)
        mean = sum(xs) / len(xs)
        median = xs[len(xs) // 2] if len(xs) % 2 else 0.5 * (xs[len(xs) // 2 - 1] + xs[len(xs) // 2])
        return float(mean), float(median)

    def _extract(self, img_path: str):
        img = self._load_image(img_path, resize=440).to(self.device)
        with self._torch.inference_mode():
            feats = self.extractor.extract(img)
        return feats, img.shape[-1], img.shape[-2]  # width, height

    def _match_prebuilt(self, feats_a, feats_b, size_a, size_b) -> MatchResult:
        with self._torch.inference_mode():
            out = self.matcher({"image0": feats_a, "image1": feats_b})
        fa, fb, out = [self._rbd(x) for x in (feats_a, feats_b, out)]

        kpts_a = fa["keypoints"].cpu().tolist()
        kpts_b = fb["keypoints"].cpu().tolist()
        pairs_ij = out["matches"].cpu().tolist()
        scores = out["scores"].cpu().tolist()
        matches = [[int(i), int(j), float(s)] for (i, j), s in zip(pairs_ij, scores)]
        mean, median = self._statsf(scores)
        return MatchResult(
            extractor=self.extractor_name, n_matches=count_confident_matches(matches, 0.5),
            mean_score=mean, median_score=median,
            kpts_a=kpts_a, kpts_b=kpts_b, matches=matches,
            img_a_size=[int(size_a[0]), int(size_a[1])],
            img_b_size=[int(size_b[0]), int(size_b[1])],
        )

    def match_pair(self, img_a_path: str, img_b_path: str) -> MatchResult:
        feats_a, wa, ha = self._extract(img_a_path)
        feats_b, wb, hb = self._extract(img_b_path)
        return self._match_prebuilt(feats_a, feats_b, (wa, ha), (wb, hb))


def _build_matcher(extractor: str) -> LocalMatcher:
    return LocalMatcher(extractor)


def get_matcher(extractor: str = "aliked") -> LocalMatcher:
    if extractor not in _MATCHER_CACHE:
        _MATCHER_CACHE[extractor] = _build_matcher(extractor)
    return _MATCHER_CACHE[extractor]


def match_pair(img_a_path: str, img_b_path: str, extractor: str = "aliked") -> MatchResult:
    return get_matcher(extractor).match_pair(img_a_path, img_b_path)


def extract_features_batch(
    image_paths: list[str], extractor: str = "aliked",
) -> dict[str, tuple[object, tuple[int, int]]]:
    """Extract features once per unique path.

    Returns mapping path -> (feats, (w, h)).
    """
    m = get_matcher(extractor)
    out: dict[str, tuple[object, tuple[int, int]]] = {}
    for path in set(image_paths):
        feats, w, h = m._extract(path)
        out[path] = (feats, (w, h))
    return out


def match_pairs_batch(
    pairs: list[tuple[str, str]],
    feats_by_path: dict[str, tuple[object, tuple[int, int]]],
    extractor: str = "aliked",
) -> list[MatchResult]:
    m = get_matcher(extractor)
    results: list[MatchResult] = []
    for a, b in pairs:
        fa, sa = feats_by_path[a]
        fb, sb = feats_by_path[b]
        results.append(m._match_prebuilt(fa, fb, sa, sb))
    return results
