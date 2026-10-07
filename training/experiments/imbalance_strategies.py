"""Experiment: how should the ~7 % positive rate be handled?

    python -m training.experiments.imbalance_strategies

Compares re-weighting strategies for LightGBM / logistic regression and shows
*why* the production pipeline re-calibrates: re-weighting improves the
learner's attention to positives but inflates raw probabilities (the mean raw
prediction drifts far above the base rate). Calibration on the validation
period restores honest probabilities without changing the ranking.

Reported on the validation period (raw vs calibrated Brier / mean prediction).
"""

from __future__ import annotations

import copy
import json
import logging
import warnings
from pathlib import Path

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.config import load_config
from src.features import FEATURE_NAMES
from src.models.estimators import fit_model
from src.models.splitting import temporal_split
from training.train import prepare_table

logger = logging.getLogger("experiments")
warnings.filterwarnings("ignore")


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    table, _ = prepare_table(cfg)
    sp = temporal_split(table, cfg["split"])
    tr, va = sp["train"], sp["validation"]
    target, seed = cfg["targets"]["primary"], cfg["project"]["random_seed"]
    y = va[target].to_numpy()
    grid = [("lightgbm", "scale_pos_weight", v) for v in ("none", "sqrt", "auto")] + \
           [("logistic_regression", "class_weight", v) for v in (None, "balanced")]
    res = {}
    for learner, param, value in grid:
        params = copy.deepcopy(cfg["training"]["models"][learner])
        params[param] = value
        m = fit_model(learner, params, FEATURE_NAMES, tr, va, target, seed, "sigmoid")
        raw, cal = m.predict_raw(va), m.predict_proba(va)
        key = f"{learner}:{param}={value}"
        res[key] = {"roc_auc": roc_auc_score(y, cal), "pr_auc": average_precision_score(y, cal),
                    "brier_raw": brier_score_loss(y, raw), "brier_calibrated": brier_score_loss(y, cal),
                    "mean_raw": float(raw.mean()), "mean_calibrated": float(cal.mean()), "base_rate": float(y.mean())}
        logger.info("%-45s %s", key, {k: round(v, 4) for k, v in res[key].items()})
    out = Path(cfg["paths"]["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "imbalance_strategies.json").write_text(json.dumps(res, indent=2))
    L = ["# Class-imbalance strategies (validation period)", "",
         "| strategy | ROC-AUC | PR-AUC | mean raw p | Brier raw | mean calibrated p | Brier calibrated | base rate |",
         "|---|---|---|---|---|---|---|---|"]
    for k, r in res.items():
        L.append(f"| {k} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['mean_raw']:.3f} | {r['brier_raw']:.4f} | "
                 f"{r['mean_calibrated']:.3f} | {r['brier_calibrated']:.4f} | {r['base_rate']:.3f} |")
    (out / "imbalance_strategies.md").write_text("\n".join(L) + "\n")
    return res


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run()
