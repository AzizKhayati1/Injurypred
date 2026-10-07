"""Experiment: which workload representation carries predictive signal?

    python -m training.experiments.workload_representations

ACWR is popular but contested (see docs/methodology.md). Instead of assuming
it, we fit the same learners on alternative *families* of workload features
and compare validation discrimination. Two views:

1. **Family alone** – only that family's features (+ nothing else).
2. **Context + family** – a shared non-workload context (athlete, injury
   history, fixtures) plus the family, i.e. the family's *added* value.

All runs use the identical chronological split (train → validation); the
test period is not touched. Results: ``reports/workload_representations.{json,md}``
and MLflow (experiment ``playerload-risk-experiments``).
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import numpy as np

from src.config import load_config
from src.features.catalog import FEATURE_GROUPS, FEATURE_NAMES, REPRESENTATION_FAMILIES
from src.models.estimators import fit_model
from src.models.metrics import cluster_bootstrap_ci, evaluate
from src.models.splitting import temporal_split
from src.models.tracking import ExperimentTracker
from training.train import prepare_table

logger = logging.getLogger("experiments")
warnings.filterwarnings("ignore")

CONTEXT = FEATURE_GROUPS["athlete"] + FEATURE_GROUPS["history"] + FEATURE_GROUPS["context"]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    table, _ = prepare_table(cfg)
    sp = temporal_split(table, cfg["split"])
    train, valid = sp["train"], sp["validation"]
    target, seed = cfg["targets"]["primary"], cfg["project"]["random_seed"]
    learners = {"logistic_regression": cfg["training"]["models"]["logistic_regression"],
                "lightgbm": cfg["training"]["models"]["lightgbm"]}
    families = {"context_only": [], **REPRESENTATION_FAMILIES, "all_features": None}
    tracker = ExperimentTracker(cfg["project"]["name"] + "-experiments")
    results = {}
    with tracker.run("workload-representations"):
        for fam, feats in families.items():
            for view in ("alone", "with_context"):
                if fam == "context_only" and view == "alone":
                    continue
                if fam == "all_features":
                    if view == "alone":
                        continue
                    cols = FEATURE_NAMES
                else:
                    cols = list(dict.fromkeys((CONTEXT if view == "with_context" else []) + feats))
                for lname, params in learners.items():
                    m = fit_model(lname, params, cols, train, valid, target, seed, cfg["training"]["calibration"])
                    p = m.predict_proba(valid)
                    ev = evaluate(valid[target], p, float(np.quantile(p, 0.9)))
                    ci = cluster_bootstrap_ci(valid[target], p, valid.index.get_level_values("athlete_id"), 100, seed)
                    key = f"{fam}|{view}|{lname}"
                    results[key] = {"family": fam, "view": view, "learner": lname, "n_features": len(cols),
                                    "roc_auc": ev["roc_auc"], "pr_auc": ev["pr_auc"], "brier": ev["brier"],
                                    "roc_auc_ci": ci["roc_auc_ci"]}
                    with tracker.run(key, nested=True):
                        tracker.log_params({"family": fam, "view": view, "learner": lname, "n_features": len(cols)})
                        tracker.log_metrics({k: results[key][k] for k in ("roc_auc", "pr_auc", "brier")})
                    logger.info("%-55s ROC-AUC %.3f  PR-AUC %.3f", key, ev["roc_auc"], ev["pr_auc"])
    out = Path(cfg["paths"]["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload_representations.json").write_text(json.dumps(results, indent=2))
    (out / "workload_representations.md").write_text(_markdown(results))
    tracker.log_artifact(out / "workload_representations.md")
    return results


def _markdown(results: dict) -> str:
    L = ["# Workload representation experiment (validation period)", "",
         "Same chronological split for every run; calibrated probabilities; 95% CI = athlete cluster bootstrap "
         "(LightGBM ROC-AUC).", "",
         "| family | view | n features | LR ROC-AUC | LGBM ROC-AUC (95% CI) | LR PR-AUC | LGBM PR-AUC |",
         "|---|---|---|---|---|---|---|"]
    keys = sorted({(r["family"], r["view"]) for r in results.values()}, key=lambda k: (k[1], k[0]))
    for fam, view in keys:
        lr = results.get(f"{fam}|{view}|logistic_regression")
        gb = results.get(f"{fam}|{view}|lightgbm")
        L.append(f"| {fam} | {view} | {gb['n_features']} | {lr['roc_auc']:.3f} | {gb['roc_auc']:.3f} "
                 f"({gb['roc_auc_ci'][0]:.3f}–{gb['roc_auc_ci'][1]:.3f}) | {lr['pr_auc']:.3f} | {gb['pr_auc']:.3f} |")
    L += ["", "Read with the CIs: differences smaller than the interval width are not evidence that one "
              "representation is better.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for n in ("mlflow", "alembic"):
        logging.getLogger(n).setLevel(logging.WARNING)
    run()
