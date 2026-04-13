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
