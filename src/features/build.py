"""Temporal feature-engineering pipeline.

``GPS sessions + absences + athlete table`` → daily athlete × date panel →
leakage-free features (+ labels for training).

Leakage rules enforced here
---------------------------
1. Every feature at prediction date ``t`` is a function of data dated ``≤ t``
   only (trailing windows; baselines are additionally lagged by 7 days).
   This is verified by ``tests/test_leakage.py``, which truncates the raw data
   at ``t`` and checks that features at ``t`` are bit-identical.
2. The only forward-looking input is the *published* team fixture calendar
   (``scheduled_matches_next_7d``), which is known in advance in practice.
   It can be disabled with ``features.use_fixture_calendar: false``.
3. Labels (``src.features.labels``) are computed from the absence log over
   ``(t, t+h]`` and are never joined back into the feature columns.
4. Latent synthetic traits (tolerance, susceptibility, …) are never read.

The same function serves training, batch scoring, single predictions and the
what-if simulator, so there is no train/serve skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.repository import Dataset
from src.features import workload as W
from src.features.catalog import FEATURE_NAMES, POSITIONS

# raw session column → daily panel name
DAILY_SUM_COLUMNS = {
    "total_distance": "total_distance",
    "high_speed_running_distance": "high_speed_distance",
    "sprint_distance": "sprint_distance",
    "sprint_count": "sprint_count",
    "accelerations": "accelerations",
    "decelerations": "decelerations",
    "player_load": "player_load",
    "training_load": "srpe_load",
    "duration_minutes": "duration",
    "minutes_played": "match_minutes",
}
BASELINE_METRICS = ["player_load", "total_distance", "high_speed_distance", "sprint_distance", "duration",
                    "accelerations"]


class DailyPanel:
    """Wide daily matrices (dates × athletes) for every raw metric and status flag."""

    def __init__(self, ds: Dataset, end: pd.Timestamp | None = None, start: pd.Timestamp | None = None,
                 fixture_lookahead_days: int = 14):
        sessions = ds.sessions
        if "training_load" not in sessions.columns or sessions["training_load"].isna().any():
            sessions = sessions.copy()
            tl = sessions["perceived_exertion"] * sessions["duration_minutes"]
            sessions["training_load"] = sessions.get("training_load", tl).fillna(tl)
        start = pd.Timestamp(start) if start is not None else sessions["date"].min()
        end = pd.Timestamp(end) if end is not None else sessions["date"].max()
        self.dates = pd.date_range(start, end, freq="D")
        self.athletes = sorted(ds.athletes["athlete_id"].unique())
        self.end = end
        s = sessions[(sessions["date"] >= start) & (sessions["date"] <= end)]
        g = s.groupby(["date", "athlete_id"])

        def wide(series: pd.Series, fill=0.0) -> pd.DataFrame:
            out = series.unstack("athlete_id").reindex(index=self.dates, columns=self.athletes)
            return out.fillna(fill) if fill is not None else out

        self.m: dict[str, pd.DataFrame] = {}
        for raw, name in DAILY_SUM_COLUMNS.items():
            self.m[name] = wide(g[raw].sum())
        self.m["max_speed"] = wide(g["max_speed"].max(), fill=None)
        rpe_dur = (s["perceived_exertion"] * s["duration_minutes"]).groupby([s["date"], s["athlete_id"]]).sum()
        self.m["rpe_x_duration"] = wide(rpe_dur)
        self.is_session = wide(g.size(), fill=0) > 0
        self.is_match = wide((s["match_or_training"] == "match").groupby([s["date"], s["athlete_id"]]).sum()) > 0

        # status flags from the absence log
        zeros = pd.DataFrame(False, index=self.dates, columns=self.athletes)
        self.unavailable = zeros.copy()
        self.injury_unavailable = zeros.copy()
        self.onset_any = zeros.copy()
        self.onset_injury = zeros.copy()
        for row in ds.absences.itertuples(index=False):
            if row.athlete_id not in self.unavailable.columns:
                continue
            last_day = row.return_date - pd.Timedelta(days=1)
            self.unavailable.loc[row.onset_date:last_day, row.athlete_id] = True
            if row.absence_type == "injury":
                self.injury_unavailable.loc[row.onset_date:last_day, row.athlete_id] = True
            if start <= row.onset_date <= end:
                self.onset_any.loc[row.onset_date, row.athlete_id] = True
                if row.absence_type == "injury":
                    self.onset_injury.loc[row.onset_date, row.athlete_id] = True

        # athlete static information
        ath = ds.athletes.set_index("athlete_id").reindex(self.athletes)
        self.athlete_info = ath
        first = sessions.groupby("athlete_id")["date"].min().reindex(self.athletes)
        self.first_session = first

        # team fixtures (public calendar) → per-athlete upcoming fixtures
        ext_dates = pd.date_range(start, end + pd.Timedelta(days=fixture_lookahead_days), freq="D")
        fx = ds.fixtures
        team_fx = pd.DataFrame(0.0, index=ext_dates, columns=sorted(ath["team_id"].unique()))
        if fx is not None and len(fx):
            counts = fx.groupby(["date", "team_id"]).size().unstack("team_id")
            counts = counts.reindex(index=ext_dates, columns=team_fx.columns).fillna(0.0)
            team_fx = counts
        next7 = W.forward_window_sum(team_fx, 7).reindex(self.dates)
        self.scheduled_next_7d = pd.DataFrame({a: next7[t] for a, t in ath["team_id"].items()},
                                              index=self.dates)[self.athletes]


def _feature_frames(p: DailyPanel, fcfg: dict) -> dict[str, pd.DataFrame]:
    a, mid, c = fcfg["acute_window"], fcfg["mid_window"], fcfg["chronic_window"]
    sa, sc = fcfg["ewma_acute_span"], fcfg["ewma_chronic_span"]
    bw, lag, bmin = fcfg["baseline_window"], fcfg["baseline_lag"], fcfg["baseline_min_days"]
    floor = fcfg.get("acwr_floor", 1e-6)
    m = p.m
    f: dict[str, pd.DataFrame] = {}

    # ---- acute / mid / chronic loads
    for name in ["total_distance", "high_speed_distance", "sprint_distance", "sprint_count", "accelerations",
                 "decelerations", "player_load", "srpe_load", "duration", "match_minutes"]:
        f[f"{name}_7d"] = W.rolling_sum(m[name], a)
    f["matches_7d"] = W.rolling_sum(p.is_match.astype(float), a)
    f["sessions_7d"] = W.rolling_sum(p.is_session.astype(float), a)
    for name in ["player_load", "sprint_distance", "high_speed_distance"]:
        f[f"{name}_{mid}d"] = W.rolling_sum(m[name], mid)
    for name in ["total_distance", "high_speed_distance", "sprint_distance", "player_load", "srpe_load"]:
        f[f"{name}_{c}d"] = W.rolling_sum(m[name], c)

    # ---- ratios: several definitions, the model decides which (if any) is useful
    f["acwr_total_load"] = W.acwr_rolling(m["player_load"], a, c, floor)
    f["acwr_total_load_uncoupled"] = W.acwr_uncoupled(m["player_load"], a, c, floor)
    f["acwr_sprint_load"] = W.acwr_rolling(m["sprint_distance"], a, c, floor)
    f["acwr_high_speed_load"] = W.acwr_rolling(m["high_speed_distance"], a, c, floor)
    f["acwr_srpe_load"] = W.acwr_rolling(m["srpe_load"], a, c, floor)
    f["ewma_acwr_total_load"] = W.acwr_ewma(m["player_load"], sa, sc, floor)
    f["ewma_acwr_sprint_load"] = W.acwr_ewma(m["sprint_distance"], sa, sc, floor)
    f["ewma_acwr_high_speed_load"] = W.acwr_ewma(m["high_speed_distance"], sa, sc, floor)
    f["ewma_acute_player_load"] = W.ewma(m["player_load"], sa)
    f["ewma_chronic_player_load"] = W.ewma(m["player_load"], sc)
    for k in [k for k in f if "acwr" in k]:
        f[k] = f[k].clip(upper=5.0)

    # ---- change / variability
    f["weekly_load_change"] = W.week_over_week_change(m["player_load"], a)
    f["sprint_load_change"] = W.week_over_week_change(m["sprint_distance"], a)
    f["high_speed_load_change"] = W.week_over_week_change(m["high_speed_distance"], a)
    f["srpe_load_change"] = W.week_over_week_change(m["srpe_load"], a)
    f["monotony_7d"] = W.monotony(m["srpe_load"], a)
    f["strain_7d"] = W.strain(m["srpe_load"], a)

    # ---- recovery
    sess = p.is_session
    f["days_since_last_session"] = W.days_since(sess, cap=60).fillna(60.0)
    f["rest_days_7d"] = a - W.rolling_sum(sess.astype(float), a)
    f["consecutive_training_days"] = W.consecutive_true(sess).astype(float)
    pl_sessions = m["player_load"].where(sess)
    high_thr = pl_sessions.rolling(bw, min_periods=10).quantile(fcfg["high_load_quantile"]).shift(1)
    f["consecutive_high_load_days"] = W.consecutive_true(sess & (m["player_load"] > high_thr)).astype(float)
    f["days_since_last_match"] = W.days_since(p.is_match, cap=60).fillna(60.0)

    # ---- RPE (duration-weighted)
    rpe7 = W.safe_ratio(W.rolling_sum(m["rpe_x_duration"], a), W.rolling_sum(m["duration"], a))
    rpe28 = W.safe_ratio(W.rolling_sum(m["rpe_x_duration"], c), W.rolling_sum(m["duration"], c))
    f["rpe_mean_7d"] = rpe7
    f["rpe_trend"] = rpe7 - rpe28

    # ---- speed
    base_speed = m["max_speed"].rolling(90, min_periods=1).max().shift(1)
    f["baseline_speed"] = base_speed
    f["max_speed_pct_7d"] = W.safe_ratio(m["max_speed"].rolling(a, min_periods=1).max(), base_speed)
    f["days_since_near_max_speed"] = W.days_since(m["max_speed"] >= 0.9 * base_speed, cap=60).fillna(60.0)

    # ---- injury / availability history (past only)
    prior = p.athlete_info["prior_injury_count"].astype(float)
    f["previous_injury_count"] = p.onset_injury.astype(float).cumsum() + prior.values
    # "no recorded event" is information, not a missing value: encode it as the cap
    f["days_since_previous_injury"] = W.days_since(p.onset_injury, cap=730).fillna(730.0)
    returned = (~p.unavailable) & p.unavailable.shift(1, fill_value=False)
    f["days_since_return"] = W.days_since(returned, cap=365).fillna(365.0)
    f["previous_30d_load"] = W.rolling_sum(m["player_load"], 30)
    f["previous_90d_injury_burden"] = W.rolling_sum(p.injury_unavailable.astype(float), 90)
    f["injuries_365d"] = W.rolling_sum(p.onset_injury.astype(float), 365)

    # ---- athlete
    dob = pd.to_datetime(p.athlete_info["date_of_birth"])
    age_days = np.subtract.outer(p.dates.values, dob.values).astype("timedelta64[D]").astype(float)
    f["age"] = pd.DataFrame(age_days / 365.25, index=p.dates, columns=p.athletes)
    f["typical_training_load"] = (m["srpe_load"].where(sess).rolling(bw, min_periods=5).median().shift(1))
    for pos in POSITIONS:
        flag = (p.athlete_info["position"] == pos).astype(float).values
        f[f"position_{pos}"] = pd.DataFrame(np.repeat(flag[None, :], len(p.dates), axis=0),
                                            index=p.dates, columns=p.athletes)

    # ---- athlete-specific baselines (trailing 12 weeks, lagged 7 days, available days only)
    unavailable_in_week = W.rolling_sum(p.unavailable.astype(float), a) > 0
    for name in BASELINE_METRICS:
        daily = m[name]
        current = W.rolling_sum(daily, a)
        base_daily = daily.where(~p.unavailable).rolling(bw, min_periods=bmin).mean().shift(lag)
        base_week = base_daily * a
        weekly_hist = current.where(~unavailable_in_week)
        base_sd = weekly_hist.rolling(bw, min_periods=bmin).std().shift(lag)
        base_sd = base_sd.where(base_sd > 0.05 * base_week.abs() + 1e-6, 0.05 * base_week.abs() + 1e-6)
        f[f"{name}_vs_baseline_pct"] = (100 * W.safe_ratio(current - base_week, base_week)).clip(-100, 500)
        f[f"{name}_vs_baseline_z"] = W.safe_ratio(current - base_week, base_sd).clip(-6, 6)
        if name in ("player_load", "high_speed_distance", "sprint_distance"):
            f[f"{name}_baseline_weekly"] = base_week

    # ---- context
    f["scheduled_matches_next_7d"] = p.scheduled_next_7d if fcfg.get("use_fixture_calendar", True) \
        else p.scheduled_next_7d * 0.0
    return f


def _season_info(dates: pd.DatetimeIndex, seasons: list[dict]) -> pd.DataFrame:
    info = pd.DataFrame(index=dates, data={"season": pd.Series(None, index=dates, dtype=object),
                                           "season_start": pd.NaT, "season_end": pd.NaT})
    for s in seasons:
        st, en = pd.Timestamp(s["start"]), pd.Timestamp(s["end"])
        mask = (dates >= st) & (dates <= en)
        info.loc[mask, "season"] = s["name"]
        info.loc[mask, "season_start"] = st
        info.loc[mask, "season_end"] = en
    return info


def _stack(frames: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    cols = {}
    for n in names:
        fr = frames[n]
        fr.index.name, fr.columns.name = "date", "athlete_id"
        cols[n] = fr.stack(future_stack=True)
    out = pd.DataFrame(cols)
    return out.swaplevel().sort_index()


def build_features(ds: Dataset, cfg: dict, end=None, start=None, athletes: list[str] | None = None,
                   panel: DailyPanel | None = None) -> pd.DataFrame:
    """Compute features for every athlete-day in [start, end].

    Returns a long frame indexed by (athlete_id, date) with all ``FEATURE_NAMES``
    plus metadata columns (team_id, position, season, available, eligible).
    """
    if athletes is not None:
        ds = ds.subset(athletes)
    p = panel or DailyPanel(ds, end=end, start=start)
    fcfg = cfg["features"]
    frames = _feature_frames(p, fcfg)
    missing = set(FEATURE_NAMES) - set(frames)
    if missing:  # pragma: no cover - guards catalogue/pipeline drift
        raise RuntimeError(f"catalogue features not produced by pipeline: {sorted(missing)}")
    frames["_available"] = ~p.unavailable
    frames["_is_session"] = p.is_session
    out = _stack(frames, FEATURE_NAMES + ["_available", "_is_session"])
    out = out.rename(columns={"_available": "available", "_is_session": "had_session"})

    meta = p.athlete_info[["team_id", "position"]]
    out = out.join(meta, on="athlete_id")
    seasons = _season_info(p.dates, cfg["calendar"]["seasons"])
    out = out.join(seasons, on="date")
    first = p.first_session.rename("first_session")
    out = out.join(first, on="athlete_id")
    burn_in = pd.Timedelta(days=fcfg["burn_in_days"])
    dates = out.index.get_level_values("date")
    out["in_season"] = out["season"].notna()
    out["eligible"] = (out["available"] & out["in_season"]
                       & (dates >= out["first_session"] + burn_in)).astype(bool)
    out[FEATURE_NAMES] = out[FEATURE_NAMES].astype("float64")
    return out


def daily_workload(ds: Dataset, athlete_id: str, cfg: dict, start=None, end=None) -> pd.DataFrame:
    """Daily raw metrics + key rolling features for one athlete (dashboard / API)."""
    sub = ds.subset([athlete_id])
    p = DailyPanel(sub, end=end)
    feats = build_features(sub, cfg, panel=p)
    feats = feats.xs(athlete_id, level="athlete_id")
    daily = pd.DataFrame({k: v[athlete_id] for k, v in p.m.items() if k != "rpe_x_duration"})
    daily["rpe"] = W.safe_ratio(p.m["rpe_x_duration"][athlete_id], p.m["duration"][athlete_id])
    daily["is_match"] = p.is_match[athlete_id]
    daily["had_session"] = p.is_session[athlete_id]
    daily["available"] = ~p.unavailable[athlete_id]
    cols = ["total_distance_7d", "player_load_7d", "player_load_28d", "sprint_distance_7d",
            "high_speed_distance_7d", "accelerations_7d", "decelerations_7d", "acwr_total_load",
            "acwr_sprint_load", "acwr_high_speed_load", "ewma_acwr_total_load", "monotony_7d", "strain_7d",
            "rest_days_7d", "sprint_distance_baseline_weekly", "player_load_baseline_weekly",
            "high_speed_distance_baseline_weekly", "sprint_distance_vs_baseline_pct"]
    out = daily.join(feats[cols])
    out.index.name = "date"
    if start is not None:
        out = out[out.index >= pd.Timestamp(start)]
    return out
