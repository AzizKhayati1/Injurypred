"""Export everything the static web app needs to ``web/data/`` (JSON).

    python -m scripts.export_static            # or: make web

Netlify serves static files (its functions run JavaScript/Go), so the Python
model cannot run there. Instead this script runs the *real* pipeline once —
the same :class:`~src.service.RiskService` behind the API — and writes
pre-computed results:

* ``meta.json``            model card, metrics, thresholds, drift summary, experiments
* ``squad.json``           athlete list + squad overview (all teams and per team)
* ``heatmap.json``         athletes × dates risk matrix (last 56 days)
* ``athletes/<id>.json``   profile, explained prediction, risk history, workload,
                           and what-if simulation sweeps

The web app can optionally call a live API (``web/config.js``) for arbitrary
what-if combinations; everything else works fully offline.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT
from src.service import RiskService, ServiceError
from src.utils import clean_json

warnings.filterwarnings("ignore")
logger = logging.getLogger("export")

OUT = PROJECT_ROOT / "web" / "data"
HISTORY_DAYS = 365
WORKLOAD_DAYS = 180
HEATMAP_DAYS = 56

# One-dimensional sweeps for the static simulator sliders (each value = one simulated scenario).
SWEEPS = {
    "total_load_pct": [-30, -15, 15, 30, 50],
    "sprint_load_pct": [-50, -25, 25, 50, 100],
    "hsr_pct": [-50, -25, 25, 50, 100],
    "duration_pct": [-50, -25, 25],
}
WORKLOAD_COLUMNS = [
    "date", "total_distance", "high_speed_distance", "sprint_distance", "player_load", "duration", "rpe", "is_match",
    "had_session", "available", "total_distance_7d", "player_load_7d", "player_load_28d", "sprint_distance_7d",
    "high_speed_distance_7d", "accelerations_7d", "decelerations_7d", "acwr_total_load", "acwr_sprint_load",
    "ewma_acwr_total_load", "monotony_7d", "rest_days_7d", "sprint_distance_baseline_weekly",
    "player_load_baseline_weekly",
]

_SVC: RiskService | None = None


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(obj), separators=(",", ":")))


def _columnar(rows: list[dict], columns: list[str]) -> dict[str, list]:
    """Rows → {column: [values]} (much smaller JSON than repeating keys per day)."""
    def r(v):
        return round(v, 3) if isinstance(v, float) else v
    return {c: [r(row.get(c)) for row in rows] for c in columns}


def _simulate(aid: str) -> tuple[str, dict | None]:
    svc = _SVC
    scenarios = [dict(p) for p in svc.cfg["simulation"]["presets"]]
    for dim, values in SWEEPS.items():
        scenarios += [{"name": f"sweep:{dim}:{v}", dim: v} for v in values]
    try:
        res = svc.simulate(aid, scenarios=scenarios)
    except ServiceError:
        return aid, None
    presets, sweeps = [], {d: {"0": res["baseline"]["risk_probability"]} for d in SWEEPS}
    for s in res["scenarios"]:
        if s["name"].startswith("sweep:"):
            _, dim, v = s["name"].split(":")
            sweeps[dim][v] = s["risk_probability"]
        else:
            presets.append({k: s[k] for k in ("name", "risk_probability", "risk_category", "delta_probability")})
    return aid, {"as_of_date": res["as_of_date"], "simulated_day": res["simulated_day"],
                 "planned_session": res["planned_session"], "current_risk": res["current_risk"],
                 "baseline": {k: res["baseline"][k] for k in ("risk_probability", "risk_category", "key_features")},
                 "presets": presets, "sweeps": sweeps, "disclaimer": res["disclaimer"]}


def _init_worker():
    # Each worker builds its own service: forking a process that has already initialised
    # OpenMP (XGBoost/LightGBM) can deadlock, so we use the "spawn" start method.
    global _SVC
    warnings.filterwarnings("ignore")
    logging.getLogger().setLevel(logging.WARNING)
    _SVC = RiskService()


def export(workers: int = 4) -> None:
    global _SVC
    t0 = time.time()
    svc = RiskService()
    _SVC = svc
    latest = svc.latest_date
    reports = Path(svc.cfg["paths"]["reports_dir"])

    def read(name):
        p = reports / name
        return json.loads(p.read_text()) if p.exists() else None

    evaluation = read("evaluation.json") or {}
    drift = read("monitoring/drift_report.json")
    meta = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "latest_date": str(latest.date()),
        "model": svc.model_info(),
        "metrics": svc.model_metrics(),
        "evaluation": {k: evaluation.get(k) for k in ("forward_cv", "athlete_holdout", "latent_trait_recovery",
                                                      "acwr_bands", "shap_importance_top20",
                                                      "shap_importance_by_family")}
        | {"test_bootstrap": (evaluation.get("test") or {}).get("bootstrap"),
           "test_top_k": (evaluation.get("test") or {}).get("top_k"),
           "test_category_thresholds": (evaluation.get("test") or {}).get("category_thresholds")},
        "experiments": {"workload_representations": read("workload_representations.json"),
                        "imbalance_strategies": read("imbalance_strategies.json")},
        "drift": None if drift is None else {k: drift.get(k) for k in ("summary", "current", "reference",
                                                                      "prediction_drift", "feature_drift",
                                                                      "performance", "label_shift")},
        "features": svc.feature_catalog()["features"],
        "teams": sorted(svc.ds.athletes["team_id"].unique()),
        "sweeps": SWEEPS,
    }
    _write(OUT / "meta.json", meta)

    squad = {"athletes": svc.list_athletes(), "overview": {"ALL": svc.squad_overview()}}
    for t in meta["teams"]:
        squad["overview"][t] = svc.squad_overview(team_id=t)
    _write(OUT / "squad.json", squad)
    _write(OUT / "heatmap.json", svc.heatmap(start=latest - pd.Timedelta(days=HEATMAP_DAYS - 1)))
    logger.info("meta/squad/heatmap written (%.1fs)", time.time() - t0)

    ids = list(svc.athletes.index)
    if workers > 1:
        with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
            sims = dict(pool.map(_simulate, ids, chunksize=4))
    else:
        sims = dict(map(_simulate, ids))
    logger.info("simulations done (%.1fs)", time.time() - t0)

    hist_start = latest - pd.Timedelta(days=HISTORY_DAYS)
    wl_start = latest - pd.Timedelta(days=WORKLOAD_DAYS)
    for aid in ids:
        try:
            pred = svc.predict(aid, log=False)
        except ServiceError as exc:
            pred = {"unavailable_reason": str(exc)}
        history = svc.risk_history(aid, start=hist_start)
        workload = svc.workload(aid, start=wl_start)
        _write(OUT / "athletes" / f"{aid}.json", {
            "profile": svc.athlete_profile(aid), "prediction": pred,
            "risk_history": _columnar(history["history"], ["date", "risk_probability", "risk_category",
                                                           "absence_onset", "available"]),
            "workload": _columnar(workload["days"], WORKLOAD_COLUMNS), "simulation": sims.get(aid)})
    size = sum(p.stat().st_size for p in OUT.rglob("*.json"))
    logger.info("Exported %d athletes, %.1f MB, in %.0fs → %s", len(ids), size / 1e6, time.time() - t0, OUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static data for the Netlify web app")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("src.service").setLevel(logging.WARNING)
    export(args.workers)


if __name__ == "__main__":
    main()
