"""What-if simulation: "what happens to the estimated risk if tomorrow's session changes?"

Mechanics
---------
1. Take the athlete's data **as known at the end of day t** (``Dataset.truncate``).
2. Build a *planned session* for t+1: by default the athlete's typical recent
   training session (or typical match exposure if the team has a fixture), or
   a session supplied by the user.
3. Apply each scenario's modifications (e.g. +20 % sprint distance, rest day).
4. Append the planned session, recompute the **same** feature pipeline for
   t+1 and score it with the served model.

IMPORTANT: this is a *model simulation*. It shows how the model's estimate
responds to a hypothetical input; it is not causal evidence that changing
the load would change the athlete's actual injury risk, and it is not
medical advice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.data.repository import Dataset
from src.features.build import build_features

SIMULATION_DISCLAIMER = ("Model simulation only: shows how the model's statistical estimate responds to a hypothetical "
                         "workload change. It is not causal evidence and not medical advice.")

VOLUME_COLUMNS = ["total_distance", "high_speed_running_distance", "sprint_distance", "sprint_count", "accelerations",
                  "decelerations", "player_load"]
KEY_FEATURES = ["player_load_7d", "sprint_distance_7d", "high_speed_distance_7d", "acwr_total_load",
                "acwr_sprint_load", "acwr_high_speed_load", "ewma_acwr_total_load", "sprint_distance_vs_baseline_pct",
                "player_load_vs_baseline_pct", "rest_days_7d", "consecutive_training_days",
                "consecutive_high_load_days", "monotony_7d"]


@dataclass
class Scenario:
    name: str
    total_load_pct: float = 0.0
    sprint_load_pct: float = 0.0
    hsr_pct: float = 0.0
    duration_pct: float = 0.0
    rpe_delta: float = 0.0
    rest_day: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        fields = {k: v for k, v in d.items() if k in cls.__dataclass_fields__ and v is not None}
        if "name" not in fields:
            fields["name"] = describe(fields)
        return cls(**fields)


def describe(d: dict) -> str:
    if d.get("rest_day"):
        return "Additional recovery day"
    parts = [f"{d[k]:+g}% {label}" for k, label in (("total_load_pct", "total load"), ("sprint_load_pct", "sprint"),
                                                    ("hsr_pct", "HSR"), ("duration_pct", "duration")) if d.get(k)]
    return ", ".join(parts) or "Planned session (no change)"


def planned_session(ds: Dataset, athlete_id: str, as_of: pd.Timestamp) -> dict:
    """Typical next-day session for this athlete, inferred from data up to ``as_of``."""
    s = ds.sessions[(ds.sessions["athlete_id"] == athlete_id) & (ds.sessions["date"] <= as_of)]
    team = ds.athletes.set_index("athlete_id").loc[athlete_id, "team_id"]
    tomorrow = as_of + pd.Timedelta(days=1)
    fixture = ((ds.fixtures["team_id"] == team) & (ds.fixtures["date"] == tomorrow)).any() if len(ds.fixtures) else False
    recent = s[s["date"] > as_of - pd.Timedelta(days=42)]
    if fixture and (recent["match_or_training"] == "match").any():
        pool, kind, stype = recent[recent["match_or_training"] == "match"], "match", "match"
    else:
        pool = recent[(recent["match_or_training"] == "training") & (recent["session_type"] != "rehab_modified")]
        kind, stype = "training", "typical_training"
        if pool.empty:
            pool = s[s["match_or_training"] == "training"].tail(10)
    if pool.empty:
        raise ValueError(f"no recent sessions to derive a planned session for {athlete_id}")
    med = pool[VOLUME_COLUMNS + ["duration_minutes", "minutes_played", "max_speed", "perceived_exertion"]].median()
    return {
        "session_type": stype, "match_or_training": kind,
        "duration_minutes": float(med["duration_minutes"]), "minutes_played": float(med["minutes_played"]),
        **{c: float(med[c]) for c in VOLUME_COLUMNS},
        "max_speed": float(med["max_speed"]), "perceived_exertion": float(round(med["perceived_exertion"])),
        "source": f"median of the athlete's last {len(pool)} {kind} session(s)" + (" (fixture scheduled)" if fixture
                                                                                     else ""),
    }


def apply_scenario(session: dict, sc: Scenario) -> dict | None:
    """Return the modified session (``None`` = rest day). Keeps sprint ≤ HSR ≤ total distance."""
    if sc.rest_day:
        return None
    s = dict(session)
    f_total = 1 + sc.total_load_pct / 100
    f_dur = 1 + sc.duration_pct / 100
    for c in VOLUME_COLUMNS:
        s[c] = s[c] * f_total * f_dur
    s["duration_minutes"] = max(1.0, s["duration_minutes"] * f_dur)
    s["minutes_played"] = min(s["minutes_played"] * f_dur, s["duration_minutes"])
    if sc.hsr_pct:
        delta = s["high_speed_running_distance"] * sc.hsr_pct / 100
        s["high_speed_running_distance"] += delta
        s["total_distance"] += delta
    if sc.sprint_load_pct:
        f = 1 + sc.sprint_load_pct / 100
        delta = s["sprint_distance"] * (f - 1)
        s["sprint_distance"] *= f
        s["sprint_count"] *= f
        s["high_speed_running_distance"] += delta
        s["total_distance"] += delta
        s["player_load"] += delta / 1000 * 88
    s["perceived_exertion"] = float(np.clip(s["perceived_exertion"] + sc.rpe_delta, 0, 10))
    for c in VOLUME_COLUMNS:
        s[c] = max(0.0, s[c])
    s["sprint_count"] = round(s["sprint_count"])
    s["accelerations"] = round(s["accelerations"])
    s["decelerations"] = round(s["decelerations"])
    s["high_speed_running_distance"] = min(s["high_speed_running_distance"], s["total_distance"])
    s["sprint_distance"] = min(s["sprint_distance"], s["high_speed_running_distance"])
    return s


def _features_with_plan(ds: Dataset, cfg: dict, athlete_id: str, as_of: pd.Timestamp, session: dict | None
                        ) -> pd.Series:
    day = as_of + pd.Timedelta(days=1)
    sessions = ds.sessions
    if session is not None:
        row = {k: v for k, v in session.items() if k != "source"}
        row.update(session_id=f"SIM-{athlete_id}-{day.date()}", athlete_id=athlete_id, date=day,
                   training_load=session["perceived_exertion"] * session["duration_minutes"])
        sessions = pd.concat([sessions, pd.DataFrame([row])], ignore_index=True)
    plan_ds = Dataset(ds.athletes, sessions, ds.absences, ds.fixtures)
    feats = build_features(plan_ds, cfg, end=day)
    return feats.loc[(athlete_id, day)]


@dataclass
class SimulationResult:
    athlete_id: str
    as_of_date: str
    simulated_day: str
    planned_session: dict
    baseline: dict
    scenarios: list[dict] = field(default_factory=list)
    disclaimer: str = SIMULATION_DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)


def run_simulation(ds: Dataset, cfg: dict, model, engine, athlete_id: str, as_of, scenarios: list[Scenario],
                   planned: dict | None = None) -> SimulationResult:
    as_of = pd.Timestamp(as_of)
    known = ds.subset([athlete_id]).truncate(as_of)          # nothing after t is visible
    plan = dict(planned) if planned else planned_session(known, athlete_id, as_of)
    plan.setdefault("source", "user supplied")
    plan.setdefault("match_or_training", "training")
    plan.setdefault("session_type", "planned")
    plan.setdefault("minutes_played", 0.0)

    def score(session):
        row = _features_with_plan(known, cfg, athlete_id, as_of, session)
        p = float(model.predict_proba(row.to_frame().T)[0])
        return p, row

    p0, row0 = score(plan)
    a0 = engine.assess(p0)
    baseline = {"name": "Planned session (no change)", **a0.to_dict(),
                "key_features": {k: _r(row0.get(k)) for k in KEY_FEATURES}}
    out = []
    for sc in scenarios:
        p, row = score(apply_scenario(plan, sc))
        a = engine.assess(p)
        changes = {k: {"baseline": _r(row0.get(k)), "scenario": _r(row.get(k))}
                   for k in KEY_FEATURES if _r(row0.get(k)) != _r(row.get(k))}
        out.append({"name": sc.name, "scenario": asdict(sc), **a.to_dict(),
                    "delta_probability": round(p - p0, 4),
                    "delta_relative_pct": round(100 * (p - p0) / p0, 1) if p0 > 0 else None,
                    "feature_changes": changes})
    plan_out = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in plan.items()}
    return SimulationResult(athlete_id, str(as_of.date()), str((as_of + pd.Timedelta(days=1)).date()), plan_out,
                            baseline, out)


def _r(v):
    return None if v is None or pd.isna(v) else round(float(v), 3)
