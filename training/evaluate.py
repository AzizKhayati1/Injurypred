"""Evaluation pipeline for the registered (champion) model.

    python -m training.evaluate

Produces ``reports/evaluation.json``, ``reports/evaluation.md`` and figures in
``reports/figures/``:

* discrimination & calibration on validation / test / labelled production rows
* athlete-level cluster-bootstrap confidence intervals
* sensitivity at the risk-engine thresholds and at top-k % review budgets
* forward (rolling-origin) validation, month by month
* athlete hold-out (generalisation to unseen athletes)
* latent-trait recovery (synthetic data only): rank correlation between an
  athlete's mean predicted risk and the hidden generator traits
* descriptive ACWR analysis (event rate by ACWR band, coupled vs EWMA)
* global SHAP importance, per feature and per feature family
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from src.config import load_config  # noqa: E402
from src.explainability.shap_explainer import ModelExplainer  # noqa: E402
from src.features.catalog import FEATURE_SPECS  # noqa: E402
from src.models.estimators import best_iteration, fit_model  # noqa: E402
from src.models.metrics import calibration_table, cluster_bootstrap_ci, evaluate  # noqa: E402
from src.models.registry import load_bundle  # noqa: E402
from src.models.splitting import athlete_holdout, forward_folds, temporal_split  # noqa: E402

logger = logging.getLogger("evaluation")
warnings.filterwarnings("ignore")


def _athletes(frame: pd.DataFrame) -> np.ndarray:
    return frame.index.get_level_values("athlete_id").to_numpy()


def latent_trait_recovery(cfg: dict, frame: pd.DataFrame, p: np.ndarray) -> dict | None:
    """Synthetic-data audit: does mean predicted risk per athlete track the *hidden* traits?

    The latent traits (susceptibility, tolerance, recovery) are never given to
    the model. A positive rank correlation with susceptibility and a negative
    one with tolerance indicate the model recovers real structure from workload
    and history alone.
    """
    path = Path(cfg["paths"]["synthetic_dir"]) / "latent_athlete_traits.csv"
    if not path.exists():
        return None
    traits = pd.read_csv(path).set_index("athlete_id")
    risk = pd.Series(p, index=frame.index.get_level_values("athlete_id")).groupby(level=0).mean()
    joined = traits.join(risk.rename("mean_predicted_risk"), how="inner")
    out = {"n_athletes": int(len(joined))}
    for col in ("injury_susceptibility", "workload_tolerance", "recovery_capacity"):
        out[f"spearman_{col}"] = round(float(joined["mean_predicted_risk"].corr(joined[col], method="spearman")), 3)
    return out


def acwr_bands(frame: pd.DataFrame, target: str) -> dict:
    out = {}
    bins = [0, 0.8, 1.0, 1.3, 1.5, 2.0, np.inf]
    labels = ["<0.8", "0.8-1.0", "1.0-1.3", "1.3-1.5", "1.5-2.0", ">2.0"]
    for col in ("acwr_total_load", "ewma_acwr_total_load", "acwr_sprint_load"):
        band = pd.cut(frame[col], bins=bins, labels=labels, right=False)
        t = frame.groupby(band, observed=False)[target].agg(["mean", "size"])
        out[col] = [{"band": str(b), "event_rate": None if pd.isna(r["mean"]) else round(float(r["mean"]), 4),
                     "n": int(r["size"])} for b, r in t.iterrows()]
    return out


def _plot_curves(y_by_split: dict, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for name, (y, p) in y_by_split.items():
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
        prec, rec, _ = precision_recall_curve(y, p)
        axes[1].plot(rec, prec, label=f"{name} (base rate {np.mean(y):.3f})")
        ct = calibration_table(y, p, 10)
        axes[2].plot(ct["mean_predicted"], ct["observed_rate"], "o-", label=name)
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC", xlabel="False positive rate", ylabel="True positive rate")
    axes[1].set(title="Precision–Recall", xlabel="Recall", ylabel="Precision")
    lim = max(0.3, max(np.max(p) for _, p in y_by_split.values()))
    axes[2].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[2].set(title="Calibration (decile bins)", xlabel="Mean predicted risk", ylabel="Observed event rate")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr_calibration.png", dpi=120)
    plt.close(fig)


def _plot_importance(imp: pd.Series, groups: pd.Series, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    top = imp.head(20)[::-1]
    axes[0].barh([FEATURE_SPECS[f].label if f in FEATURE_SPECS else f for f in top.index], top.values,
                 color="#3b6ea5")
    axes[0].set(title="Global SHAP importance (mean |SHAP|, test period)", xlabel="mean |contribution|")
    g = groups[::-1]
    axes[1].barh(g.index, g.values, color="#d07c2e")
    axes[1].set(title="Importance by feature family", xlabel="sum of mean |SHAP|")
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_importance.png", dpi=120)
    plt.close(fig)


def _plot_acwr(bands: dict, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for col, rows in bands.items():
        ax.plot([r["band"] for r in rows], [r["event_rate"] for r in rows], "o-", label=col)
    ax.set(title="Observed 7-day event rate by ACWR band (train + validation)", xlabel="ACWR band",
           ylabel="Event rate")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "acwr_bands.png", dpi=120)
    plt.close(fig)


def run_evaluation(cfg: dict | None = None, table: pd.DataFrame | None = None, ds=None) -> dict:
    cfg = cfg or load_config()
    if table is None:
        from training.train import prepare_table

        table, ds = prepare_table(cfg, ds)
    bundle = load_bundle(cfg["paths"]["model_dir"])
    model, meta = bundle["model"], bundle["metadata"]
    target, horizon = meta["target"], meta["prediction_horizon_days"]
    thresholds = meta["risk_engine"]["thresholds"]
    op = meta["operating_threshold"]["value"]
    splits = temporal_split(table, cfg["split"], horizon=horizon)
    reports = Path(cfg["paths"]["reports_dir"])
    fig_dir = reports / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["project"]["random_seed"]

    out: dict = {"model_type": meta["model_type"], "build_id": meta["build_id"], "target": target}
    curves = {}
    for name in ("validation", "test"):
        fr = splits[name]
        p = model.predict_proba(fr)
        m = evaluate(fr[target], p, op, thresholds)
        m["bootstrap"] = cluster_bootstrap_ci(fr[target], p, _athletes(fr), n_boot=200, seed=seed)
        m["calibration_table"] = calibration_table(fr[target], p, 10).round(4).to_dict("records")
        out[name] = m
        curves[name] = (fr[target].to_numpy(), p)
    prod = splits["production"]
    prod_lab = prod[prod[f"observable_{horizon}d"]]
    if len(prod_lab) and prod_lab[target].nunique() > 1:
        out["production_labelled"] = evaluate(prod_lab[target], model.predict_proba(prod_lab), op, thresholds)
    _plot_curves(curves, fig_dir)

    # ---- forward (rolling-origin) validation
    params = dict(cfg["training"]["models"][meta["model_type"]])
    if meta.get("best_iteration"):
        params["n_estimators"] = meta["best_iteration"]
    folds = []
    if cfg["split"]["forward_cv"]["enabled"]:
        for month, tr, te in forward_folds(table, cfg["split"]["forward_cv"]["fold_months"], horizon):
            fm = fit_model(meta["model_type"], params, model.features, tr, None, target, seed, horizon=horizon)
            p = fm.predict_proba(te)
            folds.append({"month": month, "n_train": int(len(tr)), "n_test": int(len(te)),
                          "base_rate": float(te[target].mean()), "roc_auc": float(roc_auc_score(te[target], p)),
                          "pr_auc": float(evaluate(te[target], p, op)["pr_auc"])})
            logger.info("forward fold %s: ROC-AUC %.3f", month, folds[-1]["roc_auc"])
        aucs = [f["roc_auc"] for f in folds]
        out["forward_cv"] = {"folds": folds, "roc_auc_mean": float(np.mean(aucs)), "roc_auc_std": float(np.std(aucs)),
                             "pr_auc_mean": float(np.mean([f["pr_auc"] for f in folds]))}

    # ---- athlete hold-out
    tr_seen, va_seen, te_unseen = athlete_holdout(splits, cfg["split"]["athlete_holdout_fraction"])
    hm = fit_model(meta["model_type"], cfg["training"]["models"][meta["model_type"]], model.features, tr_seen,
                   va_seen, target, seed, cfg["training"]["calibration"], horizon)
    te_seen = splits["test"].drop(te_unseen.index)
    out["athlete_holdout"] = {
        "n_unseen_athletes": int(te_unseen.index.get_level_values("athlete_id").nunique()),
        "unseen_test_roc_auc": float(roc_auc_score(te_unseen[target], hm.predict_proba(te_unseen))),
        "seen_test_roc_auc": float(roc_auc_score(te_seen[target], hm.predict_proba(te_seen))),
        "best_iteration": best_iteration(hm),
    }

    # ---- latent-trait recovery (synthetic audit)
    lt = latent_trait_recovery(cfg, splits["test"], curves["test"][1])
    if lt:
        out["latent_trait_recovery"] = lt

    # ---- ACWR descriptive analysis
    lab = pd.concat([splits["train"], splits["validation"]])
    out["acwr_bands"] = acwr_bands(lab, target)
    _plot_acwr(out["acwr_bands"], fig_dir)

    # ---- global SHAP importance
    explainer = ModelExplainer(model, background=bundle["background"])
    sample = splits["test"].sample(min(3000, len(splits["test"])), random_state=seed)
    imp = explainer.global_importance(sample)
    fam = imp.groupby(lambda f: FEATURE_SPECS[f].group if f in FEATURE_SPECS else "other").sum()
    fam = fam.sort_values(ascending=False)
    out["shap_importance_top20"] = {k: round(float(v), 5) for k, v in imp.head(20).items()}
    out["shap_importance_by_family"] = {k: round(float(v), 5) for k, v in fam.items()}
    out["shap_output_space"] = explainer.output_space
    _plot_importance(imp, fam, fig_dir)

    (reports / "evaluation.json").write_text(json.dumps(out, indent=2, default=float))
    (reports / "evaluation.md").write_text(render_markdown(out, meta))
    logger.info("Evaluation written to %s", reports)
    return out


def render_markdown(ev: dict, meta: dict) -> str:
    def f(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    L = [f"# Evaluation report – {meta['model_name']} v{meta['model_version']} (build {meta['build_id']})", "",
         f"Champion model: **{meta['model_type']}** · target `{meta['target']}` · "
         f"horizon {meta['prediction_horizon_days']} days · data: **synthetic**", "",
         "## Model comparison (selection on validation PR-AUC; test shown for reference only)", "",
         "| model | val ROC-AUC | val PR-AUC | val Brier | test ROC-AUC | test PR-AUC | test Brier | test ECE |",
         "|---|---|---|---|---|---|---|---|"]
    for n, r in meta["model_comparison"].items():
        v, t = r["validation"], r["test"]
        star = " ★" if n == meta["model_type"] else ""
        L.append(f"| {n}{star} | {f(v['roc_auc'])} | {f(v['pr_auc'])} | {f(v['brier'], 4)} | {f(t['roc_auc'])} | "
                 f"{f(t['pr_auc'])} | {f(t['brier'], 4)} | {f(t['ece'], 4)} |")
    L += ["", "## Champion performance", "", "| split | n | base rate | ROC-AUC (95% CI) | PR-AUC (95% CI) | Brier | "
          "Brier skill | ECE |", "|---|---|---|---|---|---|---|---|"]
    for s in ("validation", "test"):
        m = ev[s]
        b = m["bootstrap"]
        L.append(f"| {s} | {m['n']} | {f(m['base_rate'])} | {f(m['roc_auc'])} ({f(b['roc_auc_ci'][0])}–"
                 f"{f(b['roc_auc_ci'][1])}) | {f(m['pr_auc'])} ({f(b['pr_auc_ci'][0])}–{f(b['pr_auc_ci'][1])}) | "
                 f"{f(m['brier'], 4)} | {f(m['brier_skill'])} | {f(m['ece'], 4)} |")
    if "production_labelled" in ev:
        m = ev["production_labelled"]
        L.append(f"| production (labelled so far) | {m['n']} | {f(m['base_rate'])} | {f(m['roc_auc'])} | "
                 f"{f(m['pr_auc'])} | {f(m['brier'], 4)} | {f(m['brier_skill'])} | {f(m['ece'], 4)} |")
    t = ev["test"]
    L += ["", "CIs: athlete-level cluster bootstrap (200 resamples).", "",
          "## Sensitivity at risk-engine thresholds (test)", "",
          "| category ≥ | threshold | flagged share | sensitivity (recall) | precision | specificity |",
          "|---|---|---|---|---|---|"]
    for c, r in t["category_thresholds"].items():
        L.append(f"| {c} | {f(r['threshold'])} | {f(r['flagged_share'])} | {f(r['recall'])} | {f(r['precision'])} | "
                 f"{f(r['specificity'])} |")
    cm = t["at_threshold"]["confusion_matrix"]
    L += ["", f"Confusion matrix at the HIGH threshold ({f(t['at_threshold']['threshold'])}): "
              f"TP={cm['tp']} FP={cm['fp']} FN={cm['fn']} TN={cm['tn']} · precision {f(t['precision'])} · "
              f"recall {f(t['recall'])} · F1 {f(t['f1'])}", "",
          "Review-budget view (test): " + ", ".join(
              f"top {k.split('_')[1]} → recall {f(v['recall'])}, lift {f(v['lift'], 2)}×" for k, v in t["top_k"].items()),
          ""]
    if "forward_cv" in ev:
        fc = ev["forward_cv"]
        L += ["## Forward (rolling-origin) validation", "",
              f"Mean ROC-AUC {f(fc['roc_auc_mean'])} ± {f(fc['roc_auc_std'])} · mean PR-AUC {f(fc['pr_auc_mean'])}", "",
              "| month | n train | n test | base rate | ROC-AUC | PR-AUC |", "|---|---|---|---|---|---|"]
        for r in fc["folds"]:
            L.append(f"| {r['month']} | {r['n_train']} | {r['n_test']} | {f(r['base_rate'])} | {f(r['roc_auc'])} | "
                     f"{f(r['pr_auc'])} |")
        L.append("")
    ah = ev["athlete_holdout"]
    L += ["## Athlete hold-out", "",
          f"{ah['n_unseen_athletes']} athletes never seen in training: test ROC-AUC {f(ah['unseen_test_roc_auc'])} "
          f"(seen athletes: {f(ah['seen_test_roc_auc'])}).", ""]
    if "latent_trait_recovery" in ev:
        lt = ev["latent_trait_recovery"]
        L += ["## Latent-trait recovery (synthetic data only)", "",
              "Spearman correlation between each athlete's mean predicted risk (test period) and the hidden "
              "generator traits the model never sees:", "",
              f"* injury susceptibility: **{lt['spearman_injury_susceptibility']:+.3f}**",
              f"* workload tolerance: **{lt['spearman_workload_tolerance']:+.3f}**",
              f"* recovery capacity: **{lt['spearman_recovery_capacity']:+.3f}**", ""]
    L += ["## ACWR: observed event rate by band (train + validation)", "",
          "| band | " + " | ".join(ev["acwr_bands"]) + " |", "|---|" + "---|" * len(ev["acwr_bands"])]
    bands = [r["band"] for r in next(iter(ev["acwr_bands"].values()))]
    for i, b in enumerate(bands):
        cells = []
        for rows in ev["acwr_bands"].values():
            r = rows[i]
            cells.append("–" if r["event_rate"] is None else f"{r['event_rate']:.3f} (n={r['n']})")
        L.append(f"| {b} | " + " | ".join(cells) + " |")
    L += ["", "## Global SHAP importance (test period)", "", "| feature | mean abs SHAP |", "|---|---|"]
    for k, v in ev["shap_importance_top20"].items():
        L.append(f"| {k} | {v:.4f} |")
    L += ["", "| feature family | sum of mean abs SHAP |", "|---|---|"]
    for k, v in ev["shap_importance_by_family"].items():
        L.append(f"| {k} | {v:.4f} |")
    L += ["", "![curves](figures/roc_pr_calibration.png)", "", "![shap](figures/shap_importance.png)", "",
          "![acwr](figures/acwr_bands.png)", "", f"> {meta['disclaimer']}", ""]
    return "\n".join(L)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ev = run_evaluation()
    print(json.dumps({k: ev[k]["roc_auc"] for k in ("validation", "test")}, indent=2))


if __name__ == "__main__":
    main()
