"""Application service: the single entry point used by the API and the dashboard.

Business logic lives here (and in ``src/*``); the FastAPI routes and the
Streamlit app are thin presentation layers on top of :class:`RiskService`.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from src.clinical import roster
from src.clinical.injuries import catalogue_records
from src.clinical.prevention import DISCLAIMER as PREVENTION_DISCLAIMER
from src.clinical.prevention import recommend
from src.config import load_config
from src.data.repository import Dataset, load_dataset, log_predictions
from src.explainability.narrative import DISCLAIMER, baseline_comparison, build_factors, summary_sentence
from src.explainability.shap_explainer import ModelExplainer
from src.features.build import DailyPanel, build_features, daily_workload
from src.features.catalog import catalog_records
from src.models.registry import load_bundle
from src.risk.engine import CATEGORIES, RiskEngine
from src.simulation.whatif import Scenario, run_simulation
from src.utils import json_safe

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    status_code = 400


class UnknownAthleteError(ServiceError):
    status_code = 404


class DateOutOfRangeError(ServiceError):
    status_code = 422


class AthleteUnavailableError(ServiceError):
    status_code = 409


def _iso(d) -> str:
    return str(pd.Timestamp(d).date())


class RiskService:
    def __init__(self, cfg: dict | None = None, dataset: Dataset | None = None, bundle: dict | None = None):
        t0 = time.time()
        self.cfg = cfg or load_config()
        self.ds = dataset or load_dataset(self.cfg)
        self.bundle = bundle or load_bundle(self.cfg["paths"]["model_dir"])
        self.model = self.bundle["model"]
        self.meta = self.bundle["metadata"]
        self.engine = RiskEngine.from_metadata(self.meta)
        self._explainer: ModelExplainer | None = None
        self._lock = threading.Lock()
        self.panel = DailyPanel(self.ds)
        self.features = build_features(self.ds, self.cfg, panel=self.panel)
        # only in-season days of available athletes are scored (the model's training domain)
        scored = (self.features["available"] & self.features["in_season"]).to_numpy()
        p = np.full(len(self.features), np.nan)
        p[scored] = self.model.predict_proba(self.features[scored])
        self.features["risk_probability"] = p
        cats = np.full(len(p), None, dtype=object)
        cats[scored] = self.engine.categorize_many(p[scored])
        self.features["risk_category"] = cats
        self.athletes = self.ds.athletes.set_index("athlete_id")
        self.latest_date = self.ds.data_end
        self.loaded_at = datetime.now(UTC).replace(microsecond=0).isoformat()
        logger.info("RiskService ready: %d athlete-days scored in %.1fs", int(scored.sum()), time.time() - t0)

    # ------------------------------------------------------------------ helpers
    @property
    def explainer(self) -> ModelExplainer:
        with self._lock:
            if self._explainer is None:
                self._explainer = ModelExplainer(self.model, background=self.bundle.get("background"))
            return self._explainer

    def _check_athlete(self, athlete_id: str) -> None:
        if athlete_id not in self.athletes.index:
            raise UnknownAthleteError(f"unknown athlete_id '{athlete_id}'")

    def _resolve_date(self, d) -> pd.Timestamp:
        if d is None:
            return self.latest_date
        ts = pd.Timestamp(d)
        if ts > self.latest_date:
            raise DateOutOfRangeError(f"no workload data after {_iso(self.latest_date)}; predictions are made at the "
                                      f"end of a day using data up to that day")
        if ts < self.ds.data_start:
            raise DateOutOfRangeError(f"no workload data before {_iso(self.ds.data_start)}")
        return ts

    def _row(self, athlete_id: str, date: pd.Timestamp) -> pd.Series:
        return self.features.loc[(athlete_id, date)]

    def _warnings(self, row: pd.Series, date: pd.Timestamp) -> list[str]:
        w = []
        if not row["eligible"] and row["available"] and row["in_season"]:
            w.append("Less than the burn-in period of tracked history: chronic-load and baseline features are "
                     "incomplete.")
        if date + pd.Timedelta(days=7) > row["season_end"] if pd.notna(row.get("season_end")) else False:
            w.append("The 7-day horizon extends beyond the end of the season.")
        return w

    def _calibration_info(self, p: float) -> dict:
        table = self.meta.get("calibration_table_test", [])
        tm = self.meta["metrics"]["test"]
        info = {"method": self.meta.get("calibration"), "test_brier": round(tm["brier"], 4),
                "test_ece": round(tm["ece"], 4), "test_roc_auc": round(tm["roc_auc"], 3)}
        if table:
            best = min(table, key=lambda r: 0 if r["p_min"] <= p <= r["p_max"] else
                       min(abs(p - r["p_min"]), abs(p - r["p_max"])))
            info.update({"bin_range": [round(best["p_min"], 4), round(best["p_max"], 4)],
                         "bin_mean_predicted": round(best["mean_predicted"], 4),
                         "bin_observed_rate": round(best["observed_rate"], 4), "bin_n": int(best["n"]),
                         "interpretation": f"Among held-out test athlete-days with similar predicted risk, "
                                           f"{100 * best['observed_rate']:.1f}% had a new absence within 7 days "
                                           f"(n={int(best['n'])})."})
        return info

    # ------------------------------------------------------------------ predictions
    @json_safe
    def predict(self, athlete_id: str, prediction_date=None, explain: bool = True, top_k: int | None = None,
                log: bool = True) -> dict:
        self._check_athlete(athlete_id)
        date = self._resolve_date(prediction_date)
        row = self._row(athlete_id, date)
        if not row["available"]:
            ab = self.ds.absences
            cur = ab[(ab["athlete_id"] == athlete_id) & (ab["onset_date"] <= date) & (ab["return_date"] > date)]
            since = _iso(cur["onset_date"].iloc[0]) if len(cur) else "unknown"
            raise AthleteUnavailableError(f"athlete {athlete_id} is unavailable on {_iso(date)} (absence since {since}); "
                                          f"availability risk is only estimated for available athletes")
        if not row["in_season"]:
            raise DateOutOfRangeError(f"{_iso(date)} is outside the configured season calendar (off-season); the "
                                      f"model is only defined for in-season days")
        p = float(row["risk_probability"])
        a = self.engine.assess(p)
        top_k = top_k or self.cfg["risk"]["top_k_factors"]
        out = {
            "athlete_id": athlete_id,
            "prediction_date": _iso(date),
            "prediction_horizon_days": 7,
            "risk_window": {"start": _iso(date + pd.Timedelta(days=1)), "end": _iso(date + pd.Timedelta(days=7))},
            **a.to_dict(),
            "top_risk_factors": [],
            "protective_factors": [],
            "baseline_comparison": baseline_comparison(row),
            "calibration": self._calibration_info(p),
            "summary": summary_sentence(a.risk_category, a.relative_risk, a.risk_percentile),
            "warnings": self._warnings(row, date),
            "model_version": self.meta["model_version"],
            "model_type": self.meta["model_type"],
            "build_id": self.meta["build_id"],
            "data_origin": self.meta.get("data_origin", "unknown"),
            "disclaimer": DISCLAIMER,
        }
        if explain:
            contrib, base = self.explainer.explain(row[self.model.features].to_frame().T)
            prev_date = date - pd.Timedelta(days=7)
            prev = self._row(athlete_id, prev_date) if (athlete_id, prev_date) in self.features.index else None
            pos, neg = build_factors(row, contrib.iloc[0], prev, top_k=top_k)
            out["top_risk_factors"], out["protective_factors"] = pos, neg
            out["explanation_meta"] = {"method": "SHAP", "output_space": self.explainer.output_space,
                                       "base_value": round(base, 4),
                                       "note": "impact = additive contribution to the model's raw output before "
                                               "calibration; positive values increase the estimate."}
        if log:
            log_predictions([{"athlete_id": athlete_id, "prediction_date": _iso(date), "risk_probability": p,
                              "risk_category": a.risk_category, "model_version": self.meta["model_version"],
                              "build_id": self.meta["build_id"], "logged_at": datetime.now(UTC).isoformat()}])
        return out

    @json_safe
    def predict_batch(self, athlete_ids: list[str] | None = None, prediction_date=None, explain: bool = False,
                      top_k: int | None = None) -> dict:
        date = self._resolve_date(prediction_date)
        ids = athlete_ids or list(self.athletes.index)
        results, skipped = [], []
        for aid in ids:
            try:
                results.append(self.predict(aid, date, explain=explain, top_k=top_k, log=False))
            except ServiceError as exc:
                skipped.append({"athlete_id": aid, "reason": str(exc), "status_code": exc.status_code})
        log_predictions([{"athlete_id": r["athlete_id"], "prediction_date": r["prediction_date"],
                          "risk_probability": r["risk_probability"], "risk_category": r["risk_category"],
                          "model_version": self.meta["model_version"], "build_id": self.meta["build_id"],
                          "logged_at": datetime.now(UTC).isoformat()} for r in results])
        cats = pd.Series([r["risk_category"] for r in results])
        return {"prediction_date": _iso(date), "n_requested": len(ids), "n_predicted": len(results),
                "summary": {"mean_risk": round(float(np.mean([r["risk_probability"] for r in results])), 4)
                            if results else None,
                            "categories": {c: int((cats == c).sum()) for c in CATEGORIES}},
                "results": results, "skipped": skipped}

    # ------------------------------------------------------------------ players (bios, injuries, prevention)
    def _latest_row(self, athlete_id: str) -> pd.Series:
        return self._row(athlete_id, self.latest_date)

    @json_safe
    def player_card(self, athlete_id: str) -> dict:
        """Compact player card: bio, club, availability status, latest risk."""
        self._check_athlete(athlete_id)
        bio = roster.player_bio(self.ds, athlete_id, self.latest_date)
        inj = roster.player_injuries(self.ds, athlete_id, self.latest_date)
        row = self._latest_row(athlete_id)
        p = row["risk_probability"]
        return {**bio, "club": roster.club_records(self.ds)[bio["team_id"]],
                **roster.availability_status(inj, self.latest_date),
                "risk_probability": None if pd.isna(p) else round(float(p), 4),
                "risk_category": None if pd.isna(p) else row["risk_category"],
                "injuries_last_365d": inj["injuries_last_365d"], "days_lost_last_365d": inj["days_lost_last_365d"]}

    @json_safe
    def list_players(self, team_id: str | None = None) -> list[dict]:
        ids = self.ds.athletes["athlete_id"] if not team_id else \
            self.ds.athletes.loc[self.ds.athletes["team_id"] == team_id, "athlete_id"]
        return [self.player_card(a) for a in ids]

    @json_safe
    def player_profile(self, athlete_id: str) -> dict:
        self._check_athlete(athlete_id)
        card = self.player_card(athlete_id)
        return {**card, "stats": roster.player_stats(self.ds, self.cfg, athlete_id, self.latest_date),
                "injury_summary": {k: v for k, v in self.injury_history(athlete_id).items() if k != "records"},
                "data_origin": card.get("data_origin", "synthetic")}

    @json_safe
    def injury_history(self, athlete_id: str) -> dict:
        self._check_athlete(athlete_id)
        out = roster.player_injuries(self.ds, athlete_id, self.latest_date)
        out["note"] = ("Diagnoses are as recorded by medical staff (synthetic in this demo). PlayerLoad-Risk does not "
                       "diagnose injuries.")
        return out

    @json_safe
    def prevention(self, athlete_id: str) -> dict:
        self._check_athlete(athlete_id)
        row = self._latest_row(athlete_id)
        hist = roster.player_injuries(self.ds, athlete_id, self.latest_date)
        avail = bool(row["available"]) and bool(row["in_season"])
        cat = row["risk_category"] if avail else None
        recs = recommend(row if avail else None, hist, self.athletes.loc[athlete_id, "position"], cat,
                         as_of=self.latest_date)
        return {"athlete_id": athlete_id, "as_of": _iso(self.latest_date), "risk_category": cat,
                "recommendations": recs, "disclaimer": PREVENTION_DISCLAIMER}

    @json_safe
    def clubs(self) -> list[dict]:
        out = []
        for t, c in roster.club_records(self.ds).items():
            day = self.features.xs(self.latest_date, level="date")
            day = day[day["team_id"] == t]
            avail = day[day["available"]]
            out.append({**c, "n_players": int(len(day)), "n_unavailable": int((~day["available"]).sum()),
                        "mean_risk": round(float(avail["risk_probability"].mean()), 4) if len(avail) else None,
                        "n_high_risk": int(avail["risk_category"].isin(["HIGH", "VERY_HIGH"]).sum())})
        return out

    @json_safe
    def injury_report(self) -> dict:
        return roster.injury_report(self.ds, self.latest_date)

    @json_safe
    def injury_catalogue(self) -> list[dict]:
        return catalogue_records()

    # ------------------------------------------------------------------ athlete views
    @json_safe
    def list_athletes(self, team_id: str | None = None) -> list[dict]:
        a = self.ds.athletes
        if team_id:
            a = a[a["team_id"] == team_id]
        latest = self.features.xs(self.latest_date, level="date")
        out = []
        for r in a.itertuples(index=False):
            lr = latest.loc[r.athlete_id]
            out.append({"athlete_id": r.athlete_id, "team_id": r.team_id, "position": r.position,
                        "available": bool(lr["available"]),
                        "latest_risk_probability": None if pd.isna(lr["risk_probability"])
                        else round(float(lr["risk_probability"]), 4),
                        "latest_risk_category": lr["risk_category"]})
        return out

    @json_safe
    def athlete_profile(self, athlete_id: str) -> dict:
        self._check_athlete(athlete_id)
        info = self.athletes.loc[athlete_id]
        date = self.latest_date
        row = self._row(athlete_id, date)
        ab = self.ds.absences[self.ds.absences["athlete_id"] == athlete_id].sort_values("onset_date")
        s = self.ds.sessions[self.ds.sessions["athlete_id"] == athlete_id]
        current = ab[(ab["onset_date"] <= date) & (ab["return_date"] > date)]
        return {
            "athlete_id": athlete_id, "team_id": info["team_id"], "position": info["position"],
            "age": round(float(row["age"]), 1), "data_origin": info.get("data_origin", "unknown"),
            "as_of": _iso(date),
            "availability_status": "available" if row["available"] else "unavailable",
            "current_absence": None if current.empty else {
                "onset_date": _iso(current["onset_date"].iloc[0]), "absence_type": current["absence_type"].iloc[0],
                "category": current["category"].iloc[0] if "category" in current else None},
            "latest_risk": None if pd.isna(row["risk_probability"]) else self.engine.assess(
                float(row["risk_probability"])).to_dict(),
            "sessions_tracked": int(len(s)), "matches_tracked": int((s["match_or_training"] == "match").sum()),
            "first_session": _iso(s["date"].min()) if len(s) else None,
            "prior_injury_count_before_dataset": int(info["prior_injury_count"]),
            "injury_history": [{"onset_date": _iso(r.onset_date), "return_date": _iso(r.return_date),
                                "days_lost": int(r.days_lost), "absence_type": r.absence_type,
                                "category": getattr(r, "category", None), "occurred_in": getattr(r, "occurred_in", None)}
                               for r in ab.itertuples(index=False) if r.onset_date <= date],
            "baseline": baseline_comparison(row),
            "baseline_speed_kmh": None if pd.isna(row["baseline_speed"]) else round(float(row["baseline_speed"]), 2),
            "typical_training_load": None if pd.isna(row["typical_training_load"])
            else round(float(row["typical_training_load"]), 1),
        }

    @json_safe
    def risk_history(self, athlete_id: str, start=None, end=None) -> dict:
        self._check_athlete(athlete_id)
        end = self._resolve_date(end)
        start = pd.Timestamp(start) if start is not None else end - pd.Timedelta(days=90)
        f = self.features.xs(athlete_id, level="athlete_id").loc[start:end]
        ab = self.ds.absences[self.ds.absences["athlete_id"] == athlete_id]
        onsets = set(ab["onset_date"])
        rows = [{"date": _iso(d), "available": bool(r["available"]),
                 "risk_probability": None if pd.isna(r["risk_probability"]) else round(float(r["risk_probability"]), 4),
                 "risk_category": r["risk_category"], "absence_onset": d in onsets}
                for d, r in f.iterrows()]
        return {"athlete_id": athlete_id, "start": _iso(start), "end": _iso(end),
                "thresholds": self.engine.thresholds, "base_rate": self.engine.base_rate, "history": rows}

    @json_safe
    def workload(self, athlete_id: str, start=None, end=None) -> dict:
        self._check_athlete(athlete_id)
        end = self._resolve_date(end)
        start = pd.Timestamp(start) if start is not None else end - pd.Timedelta(days=90)
        w = daily_workload(self.ds, athlete_id, self.cfg, end=end).loc[start:end]
        w = w.reset_index()
        w["date"] = w["date"].dt.strftime("%Y-%m-%d")
        w = w.astype(object).where(pd.notna(w), None)
        for c in w.columns:
            if c not in ("date",):
                w[c] = [round(float(v), 3) if isinstance(v, int | float | np.floating) and not isinstance(v, bool)
                        else v for v in w[c]]
        return {"athlete_id": athlete_id, "start": _iso(start), "end": _iso(end), "days": w.to_dict("records")}

    # ------------------------------------------------------------------ simulation
    @json_safe
    def simulate(self, athlete_id: str, prediction_date=None, scenarios: list[dict] | None = None,
                 planned_session: dict | None = None) -> dict:
        self._check_athlete(athlete_id)
        date = self._resolve_date(prediction_date)
        row = self._row(athlete_id, date)
        if not row["available"]:
            raise AthleteUnavailableError(f"athlete {athlete_id} is unavailable on {_iso(date)}")
        if not row["in_season"]:
            raise DateOutOfRangeError(f"{_iso(date)} is outside the configured season calendar")
        presets = scenarios or self.cfg["simulation"]["presets"]
        sc = [Scenario.from_dict(s) for s in presets]
        res = run_simulation(self.ds, self.cfg, self.model, self.engine, athlete_id, date, sc, planned_session)
        out = res.to_dict()
        out["current_risk"] = self.engine.assess(float(self._row(athlete_id, date)["risk_probability"])).to_dict()
        out["model_version"] = self.meta["model_version"]
        return out

    # ------------------------------------------------------------------ squad views
    @json_safe
    def squad_overview(self, date=None, team_id: str | None = None) -> dict:
        date = self._resolve_date(date)
        day = self.features.xs(date, level="date")
        if team_id:
            day = day[day["team_id"] == team_id]
        avail = day[day["available"]]
        cats = avail["risk_category"].value_counts()
        window = self.features[(self.features.index.get_level_values("date") > date - pd.Timedelta(days=56))
                               & (self.features.index.get_level_values("date") <= date)]
        if team_id:
            window = window[window["team_id"] == team_id]
        trend = window.groupby(level="date").agg(
            mean_risk=("risk_probability", "mean"), player_load_7d=("player_load_7d", "mean"),
            sprint_distance_7d=("sprint_distance_7d", "mean"), unavailable=("available", lambda s: int((~s).sum())))
        trend.index = trend.index.strftime("%Y-%m-%d")
        top = avail.sort_values("risk_probability", ascending=False).head(10)
        return {
            "date": _iso(date), "team_id": team_id, "n_athletes": int(len(day)),
            "n_available": int(day["available"].sum()), "n_unavailable": int((~day["available"]).sum()),
            "n_high_risk": int(cats.get("HIGH", 0) + cats.get("VERY_HIGH", 0)),
            "mean_risk": round(float(avail["risk_probability"].mean()), 4) if len(avail) else None,
            "categories": {c: int(cats.get(c, 0)) for c in CATEGORIES},
            "base_rate": self.engine.base_rate, "thresholds": self.engine.thresholds,
            "highest_risk": [{"athlete_id": a, "team_id": r["team_id"], "position": r["position"],
                              "risk_probability": round(float(r["risk_probability"]), 4),
                              "risk_category": r["risk_category"],
                              "acwr_total_load": None if pd.isna(r["acwr_total_load"]) else round(float(r["acwr_total_load"]), 2),
                              "sprint_distance_vs_baseline_pct": None if pd.isna(r["sprint_distance_vs_baseline_pct"])
                              else round(float(r["sprint_distance_vs_baseline_pct"]), 1)}
                             for a, r in top.iterrows()],
            "trend": trend.round(4).reset_index().rename(columns={"index": "date"}).to_dict("records"),
        }

    @json_safe
    def heatmap(self, start=None, end=None, team_id: str | None = None) -> dict:
        end = self._resolve_date(end)
        start = pd.Timestamp(start) if start is not None else end - pd.Timedelta(days=27)
        f = self.features
        dates = f.index.get_level_values("date")
        sub = f[(dates >= start) & (dates <= end)]
        if team_id:
            sub = sub[sub["team_id"] == team_id]
        mat = sub["risk_probability"].unstack("date")
        return {"start": _iso(start), "end": _iso(end), "team_id": team_id,
                "athletes": list(mat.index), "dates": [_iso(d) for d in mat.columns],
                "values": [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in mat.to_numpy()],
                "note": "null = athlete unavailable (absent) on that day"}

    # ------------------------------------------------------------------ model info
    @json_safe
    def model_info(self) -> dict:
        m = self.meta
        keys = ["model_name", "model_version", "build_id", "trained_at", "model_type", "target",
                "prediction_horizon_days", "data_origin", "data_fingerprint", "data_range", "config_hash", "git_commit",
                "split", "split_sizes", "event_rates", "imbalance_handling", "calibration", "selection",
                "mlflow_run_id", "disclaimer"]
        out = {k: m.get(k) for k in keys}
        out["n_features"] = len(m["features"])
        out["risk_engine"] = {k: v for k, v in m["risk_engine"].items() if k != "reference_quantiles"}
        out["service_loaded_at"] = self.loaded_at
        out["latest_data_date"] = _iso(self.latest_date)
        return out

    @json_safe
    def model_metrics(self) -> dict:
        m = self.meta
        return {"model_version": m["model_version"], "build_id": m["build_id"], "model_type": m["model_type"],
                "selection": m["selection"], "validation": m["metrics"]["validation"], "test": m["metrics"]["test"],
                "model_comparison": m["model_comparison"], "calibration_table_test": m["calibration_table_test"],
                "operating_threshold": m["operating_threshold"]}

    @json_safe
    def feature_catalog(self) -> dict:
        schema = {f["name"]: f for f in self.bundle["feature_schema"]["features"]}
        recs = []
        for r in catalog_records():
            if r["name"] in schema:
                s = schema[r["name"]]
                r.update({k: s.get(k) for k in ("train_p01", "train_median", "train_p99", "train_missing_rate")})
                recs.append(r)
        return {"n_features": len(recs), "features": recs,
                "leakage_policy": "All features at date t use only data dated ≤ t (trailing windows; baselines lagged "
                                  "7 days). The only forward-looking input is the published fixture calendar."}

    @json_safe
    def health(self) -> dict:
        return {"status": "ok", "model_loaded": True, "model_version": self.meta["model_version"],
                "build_id": self.meta["build_id"], "n_athletes": int(len(self.athletes)),
                "data_range": [_iso(self.ds.data_start), _iso(self.latest_date)], "loaded_at": self.loaded_at}
