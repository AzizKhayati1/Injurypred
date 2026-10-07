"""Turn SHAP contributions into careful, human-readable explanations.

Language policy (enforced by ``tests/test_explainability.py``):
never claim an athlete *will* get injured; describe *associations* with
historical risk estimated from workload patterns.
"""

from __future__ import annotations

import math

import pandas as pd

from src.features.build import BASELINE_METRICS
from src.features.catalog import FEATURE_SPECS

BASELINE_LABELS = {
    "player_load": ("Player load", "AU"),
    "total_distance": ("Total distance", "m"),
    "high_speed_distance": ("High-speed running", "m"),
    "sprint_distance": ("Sprint distance", "m"),
    "duration": ("Session minutes", "min"),
    "accelerations": ("Accelerations", ""),
}

FORBIDDEN_PHRASES = ("will get injured", "will be injured", "is going to get injured", "diagnos")

DISCLAIMER = ("Statistical estimate of short-term availability risk derived from workload patterns. "
              "It is not a medical diagnosis and does not predict any individual outcome with certainty.")


def _fmt(v: float, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if unit in ("m", "AU", "min", "count", "", "per week") and abs(v) >= 100:
        s = f"{v:,.0f}"
    elif abs(v) >= 10:
        s = f"{v:.1f}"
    else:
        s = f"{v:.2f}"
    return f"{s} {unit}".strip() if unit and unit not in ("count", "flag", "ratio", "fraction") else s


def baseline_comparison(row: pd.Series) -> dict:
    """Current 7-day exposure vs the athlete's *own* normal week."""
    out = {}
    for m in BASELINE_METRICS:
        current = row.get(f"{m}_7d")
        pct = row.get(f"{m}_vs_baseline_pct")
        z = row.get(f"{m}_vs_baseline_z")
        base = row.get(f"{m}_baseline_weekly")
        if (base is None or pd.isna(base)) and pct is not None and not pd.isna(pct) and pct > -100:
            base = current / (1 + pct / 100.0)
        status = "insufficient history"
        if z is not None and not pd.isna(z):
            status = ("well above normal" if z >= 2 else "above normal" if z >= 1 else
                      "well below normal" if z <= -2 else "below normal" if z <= -1 else "within normal range")
        out[m] = {
            "current_7d": None if current is None or pd.isna(current) else round(float(current), 1),
            "baseline_weekly": None if base is None or pd.isna(base) else round(float(base), 1),
            "deviation_pct": None if pct is None or pd.isna(pct) else round(float(pct), 1),
            "z_score": None if z is None or pd.isna(z) else round(float(z), 2),
            "status": status,
        }
    return out


def describe_feature(name: str, row: pd.Series, previous: pd.Series | None, direction: str) -> str:
    """One sentence describing a feature value in context."""
    spec = FEATURE_SPECS.get(name)
    v = row.get(name)
    prev = previous.get(name) if previous is not None else None
    assoc = "associated with higher estimated risk" if direction == "increases_risk" \
        else "associated with lower estimated risk"

    if name.endswith("_vs_baseline_pct"):
        m = name.replace("_vs_baseline_pct", "")
        label, unit = BASELINE_LABELS.get(m, (m, ""))
        cur = row.get(f"{m}_7d")
        base = row.get(f"{m}_baseline_weekly")
        if (base is None or pd.isna(base)) and v is not None and not pd.isna(v) and v > -100:
            base = cur / (1 + v / 100.0)
        if v is None or pd.isna(v):
            return f"{label}: no personal baseline yet ({assoc})"
        word = "above" if v >= 0 else "below"
        return (f"{label} over the last 7 days is {abs(v):.0f}% {word} the athlete's own baseline "
                f"({_fmt(cur, unit)} vs {_fmt(base, unit)} per week) – {assoc}")
    if name.endswith("_vs_baseline_z"):
        m = name.replace("_vs_baseline_z", "")
        label, _ = BASELINE_LABELS.get(m, (m, ""))
        return f"{label} is {v:+.1f} SD from the athlete's usual weekly range – {assoc}"
    if "acwr" in name:
        lab = spec.label if spec else name
        if prev is not None and not pd.isna(prev) and not pd.isna(v):
            return f"{lab} moved from {prev:.2f} → {v:.2f} over the last 7 days – {assoc}"
        return f"{lab} is {_fmt(v)} – {assoc}"
    if name == "consecutive_high_load_days":
        return f"{int(v)} consecutive high-load session(s) relative to the athlete's own distribution – {assoc}"
    if name == "rest_days_7d":
        return f"{int(v)} recovery day(s) in the last 7 days – {assoc}"
    if name == "consecutive_training_days":
        return f"{int(v)} consecutive days with a session – {assoc}"
    if name == "days_since_return":
        if v >= 365:
            return f"No recent return from absence – {assoc}"
        return f"Returned from an absence {int(v)} day(s) ago – {assoc}"
    if name == "days_since_previous_injury":
        if v >= 730:
            return f"No injury recorded in the available history – {assoc}"
        return f"Most recent injury began {int(v)} day(s) ago – {assoc}"
    if name in ("previous_injury_count", "injuries_365d"):
        return f"{spec.label.capitalize()}: {int(v)} – {assoc}"
    if name.endswith("_change"):
        if v is None or pd.isna(v):
            return f"{spec.label.capitalize()}: n/a – {assoc}"
        return f"{spec.label.capitalize()}: {v * 100:+.0f}% – {assoc}"
    if name == "max_speed_pct_7d":
        return f"Top speed in the last 7 days reached {v * 100:.0f}% of the athlete's recent peak – {assoc}"
    if name == "scheduled_matches_next_7d":
        return f"{int(v)} team fixture(s) scheduled in the next 7 days – {assoc}"
    if name.startswith("position_"):
        return f"Playing position ({name.split('_', 1)[1]}) – {assoc}"
    unit = spec.unit if spec else ""
    lab = spec.label if spec else name
    text = f"{lab[0].upper() + lab[1:]}: {_fmt(v, unit)}"
    if prev is not None and not pd.isna(prev) and not pd.isna(v) and name.endswith(("_7d", "_14d", "_28d")):
        text += f" (7 days ago: {_fmt(prev, unit)})"
    return f"{text} – {assoc}"


def build_factors(row: pd.Series, contrib: pd.Series, previous: pd.Series | None = None,
                  top_k: int = 5, min_abs_impact: float = 1e-3) -> tuple[list[dict], list[dict]]:
    """Split contributions into top risk-increasing and top protective factors."""
    def factor(name, impact):
        direction = "increases_risk" if impact > 0 else "decreases_risk"
        spec = FEATURE_SPECS.get(name)
        val = row.get(name)
        return {
            "feature": name,
            "label": spec.label if spec else name,
            "group": spec.group if spec else None,
            "value": None if val is None or pd.isna(val) else round(float(val), 4),
            "impact": round(float(impact), 4),
            "direction": direction,
            "explanation": describe_feature(name, row, previous, direction),
        }

    pos = contrib[contrib > min_abs_impact].sort_values(ascending=False).head(top_k)
    neg = contrib[contrib < -min_abs_impact].sort_values().head(top_k)
    return [factor(n, i) for n, i in pos.items()], [factor(n, i) for n, i in neg.items()]


def summary_sentence(category: str, relative_risk: float, percentile: float) -> str:
    if category in ("HIGH", "VERY_HIGH"):
        lead = "Current workload profile is associated with elevated historical risk"
    elif category == "MODERATE":
        lead = "Current workload profile is associated with moderately increased historical risk"
    else:
        lead = "Current workload profile is associated with low-to-typical historical risk"
    return (f"{lead} ({category.replace('_', ' ')}; {relative_risk:.1f}× the reference base rate, "
            f"{percentile:.0f}th percentile of reference predictions). This is a statistical estimate, "
            f"not a medical assessment.")
