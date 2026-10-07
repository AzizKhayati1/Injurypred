"""Evaluation metrics for a rare-event probabilistic classifier.

Accuracy is deliberately *not* reported as a headline metric: with a ~7 %
base rate a model that always predicts "no event" is 93 % accurate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Quantile-binned reliability table (predicted vs observed event rate)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), q=min(n_bins, len(df)), labels=False)
    t = df.groupby("bin").agg(p_min=("p", "min"), p_max=("p", "max"), mean_predicted=("p", "mean"),
                              observed_rate=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return t


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float(np.sum(t["n"] / t["n"].sum() * (t["mean_predicted"] - t["observed_rate"]).abs()))


def classification_at(y, p, threshold: float) -> dict:
    yhat = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "flagged_share": float(yhat.mean()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def best_f1_threshold(y, p) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 120)))
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def top_k_sensitivity(y, p, shares=(0.05, 0.10, 0.20)) -> dict:
    """Recall/precision if staff reviewed only the top-k% highest-risk athlete-days."""
    out = {}
    order = np.argsort(-np.asarray(p))
    y = np.asarray(y)[order]
    for s in shares:
        k = max(1, int(round(s * len(y))))
        out[f"top_{int(s * 100)}pct"] = {"recall": float(y[:k].sum() / max(y.sum(), 1)),
                                         "precision": float(y[:k].mean()),
                                         "lift": float(y[:k].mean() / max(y.mean(), 1e-9))}
    return out


def evaluate(y, p, threshold: float, category_thresholds: dict[str, float] | None = None) -> dict:
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "n_events": int(y.sum()),
        "base_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "brier_skill": float(1 - brier_score_loss(y, p) / max(brier_score_loss(y, np.full_like(p, y.mean())), 1e-12)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
        "at_threshold": classification_at(y, p, threshold),
        "top_k": top_k_sensitivity(y, p),
    }
    out.update({k: out["at_threshold"][k] for k in ("precision", "recall", "f1")})
    if category_thresholds:
        out["category_thresholds"] = {name: classification_at(y, p, t) for name, t in category_thresholds.items()}
    return out


def cluster_bootstrap_ci(y, p, groups, n_boot: int = 200, seed: int = 0, alpha: float = 0.05) -> dict:
    """Athlete-level (cluster) bootstrap CIs for ROC-AUC and PR-AUC.

    Rows of the same athlete are strongly correlated, so resampling rows would
    understate uncertainty; we resample *athletes*.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p), "g": np.asarray(groups)})
    by_group = {g: d for g, d in df.groupby("g")}
    keys = list(by_group)
    rocs, prs = [], []
    for _ in range(n_boot):
        sample = pd.concat([by_group[k] for k in rng.choice(keys, size=len(keys), replace=True)])
        if 0 < sample["y"].sum() < len(sample):
            rocs.append(roc_auc_score(sample["y"], sample["p"]))
            prs.append(average_precision_score(sample["y"], sample["p"]))
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    return {"roc_auc_ci": [float(x) for x in np.percentile(rocs, q)],
            "pr_auc_ci": [float(x) for x in np.percentile(prs, q)], "n_boot": n_boot}
