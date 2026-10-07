"""Feature catalogue: one source of truth for names, groups, units and descriptions.

Used by the feature pipeline (column order / schema), the explainability layer
(human-readable explanations) and the ``GET /features`` endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

POSITIONS = ["GK", "CB", "FB", "CM", "W", "ST"]


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    unit: str
    description: str
    label: str  # short human label used in explanations


_SPECS: list[FeatureSpec] = []


def _add(name, group, unit, description, label=None):
    _SPECS.append(FeatureSpec(name, group, unit, description, label or description))


# ---- LOAD (acute, 7 d) -------------------------------------------------------
_add("total_distance_7d", "load", "m", "Total distance covered in the last 7 days", "7-day total distance")
_add("high_speed_distance_7d", "load", "m", "High-speed running distance (>19.8 km/h) in the last 7 days",
     "7-day high-speed running")
_add("sprint_distance_7d", "load", "m", "Sprint distance (>25.2 km/h) in the last 7 days", "7-day sprint distance")
_add("sprint_count_7d", "load", "count", "Number of sprints in the last 7 days", "7-day sprint count")
_add("accelerations_7d", "load", "count", "Accelerations (>3 m/s²) in the last 7 days", "7-day accelerations")
_add("decelerations_7d", "load", "count", "Decelerations (<-3 m/s²) in the last 7 days", "7-day decelerations")
_add("player_load_7d", "load", "AU", "Accelerometer player load in the last 7 days", "7-day player load")
_add("srpe_load_7d", "load", "AU", "Session-RPE load (RPE × minutes) in the last 7 days", "7-day sRPE load")
_add("duration_7d", "load", "min", "Total session minutes in the last 7 days", "7-day training/match minutes")
_add("match_minutes_7d", "load", "min", "Match minutes played in the last 7 days", "7-day match minutes")
_add("matches_7d", "load", "count", "Matches played in the last 7 days", "matches in the last 7 days")
_add("sessions_7d", "load", "count", "Tracked sessions in the last 7 days", "sessions in the last 7 days")
# ---- 14 d / CHRONIC (28 d) ---------------------------------------------------
_add("player_load_14d", "load_14d", "AU", "Player load in the last 14 days", "14-day player load")
_add("sprint_distance_14d", "load_14d", "m", "Sprint distance in the last 14 days", "14-day sprint distance")
_add("high_speed_distance_14d", "load_14d", "m", "High-speed running in the last 14 days", "14-day HSR")
_add("total_distance_28d", "chronic", "m", "Total distance in the last 28 days", "28-day total distance")
_add("high_speed_distance_28d", "chronic", "m", "High-speed running in the last 28 days", "28-day HSR")
_add("sprint_distance_28d", "chronic", "m", "Sprint distance in the last 28 days", "28-day sprint distance")
_add("player_load_28d", "chronic", "AU", "Player load in the last 28 days", "28-day (chronic) player load")
_add("srpe_load_28d", "chronic", "AU", "Session-RPE load in the last 28 days", "28-day sRPE load")
# ---- RATIOS ------------------------------------------------------------------
_add("acwr_total_load", "ratio", "ratio", "Coupled rolling ACWR of player load (7 d / 28 d weekly average)",
     "ACWR (player load)")
_add("acwr_total_load_uncoupled", "ratio", "ratio", "Uncoupled rolling ACWR of player load (7 d / days 8–28)",
     "uncoupled ACWR (player load)")
_add("acwr_sprint_load", "ratio", "ratio", "Coupled rolling ACWR of sprint distance", "ACWR (sprint distance)")
_add("acwr_high_speed_load", "ratio", "ratio", "Coupled rolling ACWR of high-speed running", "ACWR (high-speed running)")
_add("acwr_srpe_load", "ratio", "ratio", "Coupled rolling ACWR of session-RPE load", "ACWR (sRPE load)")
_add("ewma_acwr_total_load", "ewma", "ratio", "EWMA acute (λ=2/8) / chronic (λ=2/29) player load",
     "EWMA ACWR (player load)")
_add("ewma_acwr_sprint_load", "ewma", "ratio", "EWMA acute/chronic sprint distance", "EWMA ACWR (sprint)")
_add("ewma_acwr_high_speed_load", "ewma", "ratio", "EWMA acute/chronic high-speed running", "EWMA ACWR (HSR)")
_add("ewma_acute_player_load", "ewma", "AU/day", "EWMA (7-day span) of daily player load", "EWMA acute player load")
_add("ewma_chronic_player_load", "ewma", "AU/day", "EWMA (28-day span) of daily player load", "EWMA chronic player load")
# ---- CHANGE ------------------------------------------------------------------
_add("weekly_load_change", "change", "fraction", "Week-over-week change in player load", "week-over-week load change")
_add("sprint_load_change", "change", "fraction", "Week-over-week change in sprint distance",
     "week-over-week sprint change")
_add("high_speed_load_change", "change", "fraction", "Week-over-week change in high-speed running",
     "week-over-week HSR change")
_add("srpe_load_change", "change", "fraction", "Week-over-week change in sRPE load", "week-over-week sRPE change")
_add("monotony_7d", "variability", "ratio", "Training monotony (mean/sd of daily sRPE load, 7 d)", "training monotony")
_add("strain_7d", "variability", "AU", "Training strain (weekly sRPE load × monotony)", "training strain")
# ---- RECOVERY ----------------------------------------------------------------
_add("days_since_last_session", "recovery", "days", "Days since the last tracked session", "days since last session")
_add("rest_days_7d", "recovery", "days", "Days without a session in the last 7 days", "rest days in the last 7 days")
_add("consecutive_training_days", "recovery", "days", "Consecutive days with a session (ending today)",
     "consecutive training days")
_add("consecutive_high_load_days", "recovery", "days",
     "Consecutive session days above the athlete's own 75th-percentile load", "consecutive high-load days")
_add("days_since_last_match", "recovery", "days", "Days since the last match", "days since last match")
# ---- RPE / SPEED -------------------------------------------------------------
_add("rpe_mean_7d", "wellness", "CR-10", "Mean session RPE over the last 7 days", "7-day mean RPE")
_add("rpe_trend", "wellness", "CR-10", "7-day mean RPE minus 28-day mean RPE", "RPE trend")
_add("baseline_speed", "athlete", "km/h", "Athlete's peak speed over the previous 90 days", "baseline top speed")
_add("max_speed_pct_7d", "speed", "fraction", "Highest speed in the last 7 days as a fraction of baseline speed",
     "top-speed exposure (7 d)")
_add("days_since_near_max_speed", "speed", "days", "Days since the athlete last reached ≥90% of baseline speed",
     "days since near-maximal speed")
# ---- HISTORY -----------------------------------------------------------------
_add("previous_injury_count", "history", "count", "Recorded injuries before today (incl. pre-dataset history)",
     "previous injury count")
_add("days_since_previous_injury", "history", "days", "Days since the onset of the most recent injury",
     "days since previous injury")
_add("days_since_return", "history", "days", "Days since returning from the most recent absence",
     "days since return to play")
_add("previous_30d_load", "history", "AU", "Player load over the last 30 days", "30-day player load")
_add("previous_90d_injury_burden", "history", "days", "Days unavailable due to injury in the last 90 days",
     "injury burden (90 d)")
_add("injuries_365d", "history", "count", "Injuries with onset in the last 365 days", "injuries in the last year")
# ---- ATHLETE -----------------------------------------------------------------
_add("age", "athlete", "years", "Age on the prediction date", "age")
_add("typical_training_load", "athlete", "AU", "Athlete's median session sRPE load over the previous 12 weeks",
     "typical session load")
for _p in POSITIONS:
    _add(f"position_{_p}", "athlete", "flag", f"Position is {_p}", f"position {_p}")
# ---- BASELINE (athlete-specific) ----------------------------------------------
for _m, _lab in [("player_load", "player load"), ("total_distance", "total distance"),
                 ("high_speed_distance", "high-speed running"), ("sprint_distance", "sprint distance"),
                 ("duration", "session minutes"), ("accelerations", "accelerations")]:
    _add(f"{_m}_vs_baseline_pct", "baseline", "%", f"7-day {_lab} vs. the athlete's own weekly baseline",
         f"{_lab} vs. personal baseline")
    _add(f"{_m}_vs_baseline_z", "baseline", "z", f"7-day {_lab} as a z-score of the athlete's weekly history",
         f"{_lab} z-score vs. personal history")
for _m, _lab in [("player_load", "player load"), ("high_speed_distance", "high-speed running"),
                 ("sprint_distance", "sprint distance")]:
    _add(f"{_m}_baseline_weekly", "baseline", "per week", f"Athlete's normal weekly {_lab} (trailing 12 weeks)",
         f"normal weekly {_lab}")
# ---- CONTEXT -----------------------------------------------------------------
_add("scheduled_matches_next_7d", "context", "count",
     "Team fixtures scheduled in the next 7 days (published calendar, known in advance)", "upcoming fixtures")

FEATURE_SPECS: dict[str, FeatureSpec] = {s.name: s for s in _SPECS}
FEATURE_NAMES: list[str] = [s.name for s in _SPECS]
FEATURE_GROUPS: dict[str, list[str]] = {}
for _s in _SPECS:
    FEATURE_GROUPS.setdefault(_s.group, []).append(_s.name)

# Workload-representation families used by the representation experiment.
REPRESENTATION_FAMILIES: dict[str, list[str]] = {
    "absolute_rolling_loads": FEATURE_GROUPS["load"] + FEATURE_GROUPS["load_14d"] + FEATURE_GROUPS["chronic"],
    "acwr_rolling": ["acwr_total_load", "acwr_sprint_load", "acwr_high_speed_load", "acwr_srpe_load"],
    "acwr_uncoupled": ["acwr_total_load_uncoupled"],
    "ewma": FEATURE_GROUPS["ewma"],
    "week_change_monotony": FEATURE_GROUPS["change"] + FEATURE_GROUPS["variability"],
    "athlete_baseline_deviation": FEATURE_GROUPS["baseline"],
}


def catalog_records() -> list[dict]:
    return [asdict(s) for s in _SPECS]
