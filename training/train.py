"""Training pipeline.

    python -m training.train                 # train, select, evaluate, register
    python -m training.train --no-evaluate   # skip the (slower) full evaluation report

Steps
-----
1. Load data (DB or CSV) and **validate** it (blocking on errors).
2. Build the leakage-free modelling table (features + labels).
3. Chronological split with purging (train → validation → test).
4. Fit every enabled model on *train*; early-stop/calibrate on *validation*.
5. Select the champion on validation PR-AUC (Brier as tie-breaker).
   The test period is only used to *report* performance.
6. Resolve risk-engine thresholds, build SHAP background, monitoring reference.
7. Save a versioned bundle under ``models/`` and log everything to MLflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import config_hash, load_config
from src.data.repository import load_dataset
from src.data.validation import validate_dataset
from src.explainability.narrative import DISCLAIMER
from src.features import FEATURE_NAMES, build_modeling_table
from src.models.estimators import best_iteration, fit_model
from src.models.metrics import calibration_table, evaluate
from src.models.registry import feature_schema, git_commit, save_bundle, utc_now
from src.models.splitting import temporal_split
from src.models.tracking import ExperimentTracker
from src.risk.engine import resolve_thresholds

logger = logging.getLogger("training")
warnings.filterwarnings("ignore", category=UserWarning)


def prepare_table(cfg: dict, ds=None, cache: bool = True) -> tuple[pd.DataFrame, object]:
    ds = ds or load_dataset(cfg)
    report = validate_dataset(ds.athletes, ds.sessions, ds.absences)
    logger.info(report.summary())
    fp = ds.fingerprint()
    cache_path = Path(cfg["paths"]["processed_dir"]) / f"modeling_table_{fp}_{config_hash(cfg)}.parquet"
    if cache and cache_path.exists():
        table = pd.read_parquet(cache_path)
        logger.info("Loaded cached modelling table %s", cache_path.name)
    else:
        t0 = time.time()
        table = build_modeling_table(ds, cfg)
        logger.info("Built modelling table %s in %.1fs", table.shape, time.time() - t0)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            for old in cache_path.parent.glob("modeling_table_*.parquet"):
                old.unlink()
            table.to_parquet(cache_path)
    return table, ds


def _select(results: dict, metric: str) -> str:
    return max(results, key=lambda n: (round(results[n]["validation"][metric], 3), -results[n]["validation"]["brier"]))


def run_training(cfg: dict | None = None, ds=None, evaluate_after: bool = True, track: bool = True) -> dict:
    cfg = cfg or load_config()
    seed = cfg["project"]["random_seed"]
    np.random.seed(seed)
    target = cfg["targets"]["primary"]
    horizon = 7
    build_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

    table, ds = prepare_table(cfg, ds)
    splits = temporal_split(table, cfg["split"], horizon=horizon)
    train, valid, test = splits["train"], splits["validation"], splits["test"]
    sizes = {k: int(len(v)) for k, v in splits.items()}
    rates = {k: float(splits[k][target].mean()) for k in ("train", "validation", "test")}
    logger.info("Split sizes %s, event rates %s", sizes, {k: round(v, 4) for k, v in rates.items()})

    tracker = ExperimentTracker(cfg["project"]["name"], enabled=track)
    tcfg = cfg["training"]
    results: dict[str, dict] = {}
    models = {}
    base_rate = rates["validation"]
    thresholds = resolve_thresholds(cfg["risk"], base_rate)
    op_threshold = thresholds["HIGH"]

    with tracker.run(f"train-{build_id}", tags={"build_id": build_id, "data_origin": "synthetic"}):
        tracker.log_params({
            "config_hash": config_hash(cfg), "data_fingerprint": ds.fingerprint(), "target": target,
            "horizon_days": horizon, "n_features": len(FEATURE_NAMES), "split": cfg["split"],
            "calibration": tcfg["calibration"], "selection_metric": tcfg["selection_metric"],
        })
        tracker.log_metrics({"rows": sizes, "event_rate": rates})
        for name, params in tcfg["models"].items():
            if not params.get("enabled", True):
                continue
            t0 = time.time()
            with tracker.run(name, nested=True, tags={"model_type": name}):
                model = fit_model(name, params, FEATURE_NAMES, train, valid, target, seed, tcfg["calibration"],
                                  horizon)
                res = {}
                for split_name, frame in (("validation", valid), ("test", test)):
                    p = model.predict_proba(frame)
                    res[split_name] = evaluate(frame[target], p, op_threshold, thresholds)
                res["fit_seconds"] = round(time.time() - t0, 1)
                res["best_iteration"] = best_iteration(model)
                tracker.log_params({k: v for k, v in params.items()} | {"model_type": name})
                tracker.log_metrics({"validation": _headline(res["validation"]), "test": _headline(res["test"])})
                results[name] = res
                models[name] = model
            logger.info("%-22s val ROC-AUC %.3f PR-AUC %.3f Brier %.4f | test ROC-AUC %.3f PR-AUC %.3f (%.1fs)",
                        name, res["validation"]["roc_auc"], res["validation"]["pr_auc"], res["validation"]["brier"],
                        res["test"]["roc_auc"], res["test"]["pr_auc"], res["fit_seconds"])

        champion_name = _select(results, tcfg["selection_metric"])
        champion = models[champion_name]
        logger.info("Champion: %s (selected on validation %s)", champion_name, tcfg["selection_metric"])

        p_val = champion.predict_proba(valid)
        p_test = champion.predict_proba(test)
        ref_quantiles = np.quantile(p_val, np.linspace(0, 1, 201)).tolist()
        background = train[FEATURE_NAMES].sample(min(500, len(train)), random_state=seed)

        mon_feats = [f for f in cfg["monitoring"]["features"] if f in FEATURE_NAMES]
        ref = train.sample(min(8000, len(train)), random_state=seed)
        reference = ref[FEATURE_NAMES].copy()
        reference["prediction"] = champion.predict_proba(ref)
        reference[target] = ref[target].values
        reference = reference.reset_index()

        metadata = {
            "model_name": cfg["project"]["model_name"],
            "model_version": cfg["project"]["model_version"],
            "build_id": build_id,
            "trained_at": utc_now(),
            "model_type": champion_name,
            "target": target,
            "prediction_horizon_days": horizon,
            "features": FEATURE_NAMES,
            "monitoring_features": mon_feats,
            "data_origin": "synthetic",
            "data_fingerprint": ds.fingerprint(),
            "data_range": [str(ds.data_start.date()), str(ds.data_end.date())],
            "config_hash": config_hash(cfg),
            "git_commit": git_commit(),
            "split": cfg["split"],
            "split_sizes": sizes,
            "event_rates": rates,
            "imbalance_handling": {k: v.get("class_weight", v.get("scale_pos_weight")) for k, v in
                                   tcfg["models"].items() if v.get("enabled", True)},
            "calibration": tcfg["calibration"] + " (fitted on validation period only)",
            "selection": f"highest validation {tcfg['selection_metric']} (Brier tie-break)",
            "best_iteration": results[champion_name]["best_iteration"],
            "metrics": {"validation": results[champion_name]["validation"], "test": results[champion_name]["test"]},
            "model_comparison": {n: {"validation": _headline(r["validation"]), "test": _headline(r["test"]),
                                     "fit_seconds": r["fit_seconds"]} for n, r in results.items()},
            "risk_engine": {"mode": cfg["risk"]["mode"], "thresholds": thresholds, "base_rate": base_rate,
                            "reference_quantiles": ref_quantiles,
                            "reference": "calibrated predictions on the validation period"},
            "calibration_table_test": calibration_table(test[target], p_test, 10).to_dict("records"),
            "operating_threshold": {"name": "HIGH", "value": op_threshold},
            "disclaimer": DISCLAIMER,
        }
        metadata["mlflow_run_id"] = tracker.run_id
        model_dir = Path(cfg["paths"]["model_dir"])
        save_bundle(model_dir, champion, background, metadata, feature_schema(FEATURE_NAMES, train), reference)
        reports = Path(cfg["paths"]["reports_dir"])
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "model_comparison.json").write_text(json.dumps(metadata["model_comparison"], indent=2))
        for f in ("metadata.json", "feature_schema.json"):
            tracker.log_artifact(model_dir / f, "bundle")
        tracker.log_artifact(reports / "model_comparison.json", "reports")
        tracker.log_metrics({"champion": {"validation": _headline(metadata["metrics"]["validation"]),
                                          "test": _headline(metadata["metrics"]["test"])}})
        tracker.set_tags({"champion": champion_name, "model_version": metadata["model_version"]})
        tracker.log_model(model_dir / "model.joblib", registered_name=cfg["project"]["model_name"])

        if evaluate_after:
            from training.evaluate import run_evaluation

            ev = run_evaluation(cfg, table=table, ds=ds)
            for f in Path(cfg["paths"]["reports_dir"]).rglob("*"):
                if f.is_file() and f.suffix in (".json", ".md", ".png"):
                    tracker.log_artifact(f, "reports/" + f.parent.name if f.parent.name != "reports" else "reports")
            tracker.log_metrics({"evaluation": {"test_roc_auc_ci_low": ev["test"]["bootstrap"]["roc_auc_ci"][0],
                                                "test_roc_auc_ci_high": ev["test"]["bootstrap"]["roc_auc_ci"][1]}})
    return metadata


def _headline(m: dict) -> dict:
    keys = ("roc_auc", "pr_auc", "brier", "brier_skill", "log_loss", "ece", "precision", "recall", "f1", "base_rate",
            "mean_predicted")
    return {k: round(float(m[k]), 5) for k in keys if k in m}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PlayerLoad-Risk models")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("mlflow", "alembic", "urllib3", "git"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    meta = run_training(evaluate_after=not args.no_evaluate, track=not args.no_mlflow)
    print(json.dumps({"champion": meta["model_type"], "build_id": meta["build_id"],
                      "test": _headline(meta["metrics"]["test"])}, indent=2))


if __name__ == "__main__":
    main()
