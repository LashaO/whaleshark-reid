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

    def _load_chip_tensor(self, file_path: str, bbox, theta: float):
        """Load the same bbox+theta chip the UI shows, as a LightGlue-ready tensor.

        Without this, we'd run the extractor on the full source image (often
        multi-subject, 2k+ px) but plot keypoints over the displayed chip —
        coordinates and features would both be wrong.
        """
        import numpy as np
        from PIL import Image
        from wbia_miew_id.datasets.helpers import get_chip_from_img

        img = Image.open(file_path).convert("RGB")
        arr = np.array(img)
        if bbox:
            chip = get_chip_from_img(arr, list(bbox), float(theta or 0.0))
        else:
            chip = arr
        # HWC uint8 → CHW float [0,1]
        t = self._torch.from_numpy(chip.astype(np.float32) / 255.0).permute(2, 0, 1)
        # Match `lightglue.load_image(resize=440)`: longer side → 440, keep aspect.
        h, w = t.shape[-2:]
        scale = 440 / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        t = self._torch.nn.functional.interpolate(
            t.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
        )[0]
        return t.to(self.device)

    def _extract(self, img_path: str, bbox=None, theta: float = 0.0):
        if bbox is None:
            img = self._load_image(img_path, resize=440).to(self.device)
        else:
            img = self._load_chip_tensor(img_path, bbox, theta)
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

    def match_pair(
        self,
        img_a_path: str,
        img_b_path: str,
        bbox_a=None,
        theta_a: float = 0.0,
        bbox_b=None,
        theta_b: float = 0.0,
    ) -> MatchResult:
        feats_a, wa, ha = self._extract(img_a_path, bbox_a, theta_a)
        feats_b, wb, hb = self._extract(img_b_path, bbox_b, theta_b)
        return self._match_prebuilt(feats_a, feats_b, (wa, ha), (wb, hb))


def _build_matcher(extractor: str) -> LocalMatcher:
    return LocalMatcher(extractor)


def get_matcher(extractor: str = "aliked") -> LocalMatcher:
    if extractor not in _MATCHER_CACHE:
        _MATCHER_CACHE[extractor] = _build_matcher(extractor)
    return _MATCHER_CACHE[extractor]


def match_pair(
    img_a_path: str,
    img_b_path: str,
    extractor: str = "aliked",
    bbox_a=None,
    theta_a: float = 0.0,
    bbox_b=None,
    theta_b: float = 0.0,
) -> MatchResult:
    return get_matcher(extractor).match_pair(
        img_a_path, img_b_path,
        bbox_a=bbox_a, theta_a=theta_a, bbox_b=bbox_b, theta_b=theta_b,
    )


# A chip spec is identified by a caller-chosen key (typically an annotation UUID)
# and produces a distinct extraction from the same source file — two annotations
# on one image get their own features.
#   ChipSpec = (key, file_path, bbox_xywh_list_or_None, theta_float)


def extract_features_batch(
    specs: list[tuple[str, str, object, float]] | list[str],
    extractor: str = "aliked",
) -> dict[str, tuple[object, tuple[int, int]]]:
    """Extract features once per unique key.

    Accepts either a list of `(key, path, bbox, theta)` specs (preferred — the
    key is whatever identity the caller wants, e.g. annotation UUID) or a bare
    list of paths (backward compat; key == path, no cropping).

    Returns mapping key -> (feats, (w, h)).
    """
    m = get_matcher(extractor)
    out: dict[str, tuple[object, tuple[int, int]]] = {}
    seen: set[str] = set()
    for spec in specs:
        if isinstance(spec, str):
            key, path, bbox, theta = spec, spec, None, 0.0
        else:
            key, path, bbox, theta = spec
        if key in seen:
            continue
        seen.add(key)
        feats, w, h = m._extract(path, bbox, theta)
        out[key] = (feats, (w, h))
    return out


def match_pairs_batch(
    pairs: list[tuple[str, str]],
    feats_by_key: dict[str, tuple[object, tuple[int, int]]],
    extractor: str = "aliked",
) -> list[MatchResult]:
    m = get_matcher(extractor)
    results: list[MatchResult] = []
    for a, b in pairs:
        fa, sa = feats_by_key[a]
        fb, sb = feats_by_key[b]
        results.append(m._match_prebuilt(fa, fb, sa, sb))
    return results
