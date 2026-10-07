"""Risk engine: probability → category, percentile and relative risk.

Why not fixed 20/40/70 % cut points?
-----------------------------------
Weekly time-loss event probabilities in football are low (≈5–10 % per
athlete-week), so with absolute cut points such as 20/40/70 % virtually every
athlete would be "LOW" and the categories would carry no information.

The default (``risk.mode: relative``) therefore defines categories as
*multiples of the reference base rate* estimated on the validation period
(e.g. HIGH = ≥ 2.5 × the typical weekly risk). Absolute thresholds remain
available (``risk.mode: absolute``) for sports/contexts where they make sense.
Both are configured in ``configs/config.yaml``; the resolved thresholds are
stored in the model metadata so they are versioned with the model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CATEGORIES = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]


@dataclass
class RiskAssessment:
    risk_probability: float
    risk_category: str
    risk_percentile: float
    relative_risk: float
    base_rate: float
    thresholds: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_thresholds(risk_cfg: dict, base_rate: float) -> dict[str, float]:
    """Lower bounds (probabilities) of MODERATE, HIGH, VERY_HIGH."""
    if risk_cfg.get("mode", "relative") == "absolute":
        th = {k: float(v) for k, v in risk_cfg["absolute_thresholds"].items()}
    else:
        th = {k: float(min(0.99, v * base_rate)) for k, v in risk_cfg["relative_multipliers"].items()}
    ordered = [th[c] for c in CATEGORIES[1:]]
    if any(b <= a for a, b in zip(ordered, ordered[1:])) or not all(0 < t < 1 for t in ordered):
        raise ValueError(f"risk thresholds must be increasing and inside (0, 1): {th}")
    return {c: th[c] for c in CATEGORIES[1:]}


class RiskEngine:
    def __init__(self, thresholds: dict[str, float], base_rate: float, reference_quantiles: list[float]):
        self.thresholds = dict(thresholds)
        self.base_rate = float(base_rate)
        self.reference = np.sort(np.asarray(reference_quantiles, dtype=float))

    @classmethod
    def from_metadata(cls, meta: dict) -> RiskEngine:
        r = meta["risk_engine"]
        return cls(r["thresholds"], r["base_rate"], r["reference_quantiles"])

    def categorize(self, p: float) -> str:
        cat = "LOW"
        for c in CATEGORIES[1:]:
            if p >= self.thresholds[c]:
                cat = c
        return cat

    def percentile(self, p: float) -> float:
        if len(self.reference) == 0:
            return float("nan")
        return float(100.0 * np.searchsorted(self.reference, p, side="right") / len(self.reference))

    def assess(self, p: float) -> RiskAssessment:
        p = float(np.clip(p, 0.0, 1.0))
        return RiskAssessment(
            risk_probability=round(p, 4),
            risk_category=self.categorize(p),
            risk_percentile=round(self.percentile(p), 1),
            relative_risk=round(p / self.base_rate, 2) if self.base_rate > 0 else float("nan"),
            base_rate=round(self.base_rate, 4),
            thresholds={k: round(v, 4) for k, v in self.thresholds.items()},
        )

    def categorize_many(self, p: np.ndarray) -> np.ndarray:
        bounds = [self.thresholds[c] for c in CATEGORIES[1:]]
        return np.asarray(CATEGORIES, dtype=object)[np.searchsorted(bounds, np.asarray(p), side="right")]
