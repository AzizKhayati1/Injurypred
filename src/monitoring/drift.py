"""Model monitoring: feature drift, prediction drift, data quality, label shift, performance.

    python -m src.monitoring.drift              # production period vs training reference
    python -m src.monitoring.drift --start 2026-09-01

Reference = training-period sample saved with the model (``reference_sample.parquet``).
Current   = scored athlete-days of the monitoring window (default: the
``split.production`` period, i.e. the current season).

Statistics
----------
* **PSI** (population stability index) on reference-quantile bins:
  < 0.10 stable · 0.10–0.25 moderate shift · > 0.25 significant shift.
* **Kolmogorov–Smirnov** two-sample test (p-value reported; with tens of
  thousands of rows tiny shifts are "significant", so PSI drives alerts).
* Missing-value rate change, prediction-distribution drift, class-balance
  change (two-proportion z-test) and performance on rows whose 7-day label
  window has already closed.

If `Evidently <https://github.com/evidentlyai/evidently>`_ is installed an
additional Evidently data-drift HTML report is produced.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (near-)constant reference: compare a two-bin split
        edges = np.array([-np.inf, np.median(ref), np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _status(value: float, warn: float, alert: float) -> str:
    if value is None or np.isnan(value):
        return "unknown"
    return "alert" if value >= alert else "warning" if value >= warn else "ok"


def feature_drift(ref: pd.DataFrame, cur: pd.DataFrame, features: list[str], mcfg: dict) -> list[dict]:
    rows = []
    for f in features:
        r, c = ref[f].to_numpy(dtype=float), cur[f].to_numpy(dtype=float)
        p = psi(r, c)
        rn, cn = r[~np.isnan(r)], c[~np.isnan(c)]
        ks = stats.ks_2samp(rn, cn) if len(rn) and len(cn) else None
        rows.append({
            "feature": f, "psi": round(p, 4), "status": _status(p, mcfg["psi_warning"], mcfg["psi_alert"]),
            "ks_statistic": round(float(ks.statistic), 4) if ks else None,
            "ks_pvalue": float(ks.pvalue) if ks else None,
            "reference_mean": round(float(np.nanmean(r)), 3), "current_mean": round(float(np.nanmean(c)), 3),
            "mean_change_pct": round(100 * (np.nanmean(c) - np.nanmean(r)) / abs(np.nanmean(r)), 1)
            if np.nanmean(r) else None,
            "reference_missing_rate": round(float(np.isnan(r).mean()), 4),
            "current_missing_rate": round(float(np.isnan(c).mean()), 4),
        })
    return sorted(rows, key=lambda x: -(x["psi"] if x["psi"] == x["psi"] else -1))


def label_shift(ref_y: np.ndarray, cur_y: np.ndarray) -> dict:
    p1, p2, n1, n2 = ref_y.mean(), cur_y.mean(), len(ref_y), len(cur_y)
    pooled = (ref_y.sum() + cur_y.sum()) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else 0.0
    return {"reference_event_rate": round(float(p1), 4), "current_event_rate": round(float(p2), 4),
            "z": round(float(z), 2), "pvalue": float(2 * (1 - stats.norm.cdf(abs(z)))), "n_current_labelled": int(n2)}


def build_report(reference: pd.DataFrame, current: pd.DataFrame, meta: dict, cfg: dict, target: str,
                 window: tuple[str, str]) -> dict:
    mcfg = cfg["monitoring"]
    features = [f for f in meta.get("monitoring_features", mcfg["features"]) if f in current.columns]
    all_features = [f for f in meta["features"] if f in current.columns]
    fd = feature_drift(reference, current, features, mcfg)
    pred = {"psi": round(psi(reference["prediction"], current["risk_probability"]), 4),
            "reference_mean": round(float(reference["prediction"].mean()), 4),
            "current_mean": round(float(current["risk_probability"].mean()), 4)}
    pred["status"] = _status(pred["psi"], mcfg["psi_warning"], mcfg["psi_alert"])
    miss_ref = reference[all_features].isna().mean()
    miss_cur = current[all_features].isna().mean()
    dq = [{"feature": f, "reference_missing_rate": round(float(miss_ref[f]), 4),
           "current_missing_rate": round(float(miss_cur[f]), 4)}
          for f in all_features if miss_cur[f] - miss_ref[f] > 0.02]
    labelled = current[current.get("observable_7d", pd.Series(False, index=current.index)).astype(bool)]
    report = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "model_version": meta["model_version"], "build_id": meta["build_id"],
        "reference": {"description": "training-period sample (same calendar months as the window)",
                      "n": int(len(reference))},
        "current": {"start": window[0], "end": window[1], "n": int(len(current))},
        "feature_drift": fd,
        "all_feature_psi": {f: round(psi(reference[f], current[f]), 4) for f in all_features},
        "prediction_drift": pred,
        "data_quality": {"missing_value_increase": dq},
    }
    if len(labelled) and target in labelled and labelled[target].nunique() > 1:
        y, p = labelled[target].to_numpy(), labelled["risk_probability"].to_numpy()
        auc, brier = float(roc_auc_score(y, p)), float(brier_score_loss(y, p))
        ref_auc = meta["metrics"]["test"]["roc_auc"]
        drop = ref_auc - auc
        report["label_shift"] = label_shift(reference[target].to_numpy(), y)
        report["performance"] = {"current_roc_auc": round(auc, 4), "reference_test_roc_auc": round(ref_auc, 4),
                                 "roc_auc_drop": round(drop, 4), "current_brier": round(brier, 4),
                                 "reference_test_brier": round(meta["metrics"]["test"]["brier"], 4),
                                 "current_mean_predicted": round(float(p.mean()), 4),
                                 "current_event_rate": round(float(y.mean()), 4),
                                 "status": "alert" if drop > mcfg["performance_drop_alert"] else "ok",
                                 "n_labelled": int(len(y))}
    n_alert = sum(r["status"] == "alert" for r in fd)
    n_warn = sum(r["status"] == "warning" for r in fd)
    perf_alert = report.get("performance", {}).get("status") == "alert"
    report["summary"] = {
        "overall_status": "alert" if (n_alert or pred["status"] == "alert" or perf_alert) else
        "warning" if (n_warn or pred["status"] == "warning") else "ok",
        "features_alert": n_alert, "features_warning": n_warn,
        "drifted_features": [r["feature"] for r in fd if r["status"] != "ok"],
        "recommendation": _recommendation(n_alert, pred["status"], perf_alert),
    }
    return report


def _recommendation(n_alert: int, pred_status: str, perf_alert: bool) -> str:
    if perf_alert:
        return "Performance degradation detected: investigate data changes and retrain/recalibrate the model."
    if n_alert or pred_status == "alert":
        return ("Significant input drift detected. Check whether training methodology or data collection changed; "
                "monitor performance as labels mature and consider recalibration/retraining.")
    return "No action required."


def render_html(report: dict) -> str:
    color = {"ok": "#2e7d32", "warning": "#ef6c00", "alert": "#c62828", "unknown": "#757575"}

    def badge(s):
        return f'<span style="color:#fff;background:{color.get(s, "#757575")};padding:2px 8px;border-radius:8px">{s}</span>'

    rows = "".join(
        f"<tr><td>{html.escape(r['feature'])}</td><td>{r['psi']}</td><td>{badge(r['status'])}</td>"
        f"<td>{r['reference_mean']}</td><td>{r['current_mean']}</td><td>{r['mean_change_pct']}</td>"
        f"<td>{r['ks_statistic']}</td><td>{r['current_missing_rate']}</td></tr>" for r in report["feature_drift"])
    perf = report.get("performance")
    perf_html = "<p>No labelled rows in the monitoring window yet.</p>" if not perf else (
        f"<p>ROC-AUC {perf['current_roc_auc']} (reference test {perf['reference_test_roc_auc']}, drop "
        f"{perf['roc_auc_drop']}) · Brier {perf['current_brier']} · {badge(perf['status'])} · n={perf['n_labelled']}</p>")
    ls = report.get("label_shift")
    ls_html = "" if not ls else (f"<p>Event rate: reference {ls['reference_event_rate']} → current "
                                 f"{ls['current_event_rate']} (z={ls['z']}, p={ls['pvalue']:.3g})</p>")
    s = report["summary"]
    pd_ = report["prediction_drift"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ddd;padding:4px 8px;font-size:14px}}th{{background:#f4f4f4}}</style></head><body>
<h1>PlayerLoad-Risk – monitoring report</h1>
<p>Model {report['model_version']} (build {report['build_id']}) · generated {report['generated_at']}</p>
<p>Reference: {report['reference']['description']} (n={report['reference']['n']}) · Current window:
{report['current']['start']} → {report['current']['end']} (n={report['current']['n']})</p>
<h2>Overall: {badge(s['overall_status'])}</h2><p>{html.escape(s['recommendation'])}</p>
<h2>Prediction drift</h2><p>PSI {pd_['psi']} {badge(pd_['status'])} · mean prediction {pd_['reference_mean']} →
{pd_['current_mean']}</p>
<h2>Performance (labels matured)</h2>{perf_html}{ls_html}
<h2>Feature drift</h2><table><tr><th>feature</th><th>PSI</th><th>status</th><th>ref mean</th><th>current mean</th>
<th>Δ mean %</th><th>KS</th><th>missing (current)</th></tr>{rows}</table>
<p style="color:#666">PSI &lt; 0.10 stable · 0.10–0.25 moderate · &gt; 0.25 significant. Synthetic data.</p>
</body></html>"""


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str], path: Path) -> bool:
    """Optional Evidently data-drift report (only if the package is installed)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except Exception:
        return False
    try:
        snapshot = Report([DataDriftPreset()]).run(current_data=current[features], reference_data=reference[features])
        snapshot.save_html(str(path))
        return True
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Evidently report failed: %s", exc)
        return False


def seasonal_reference(reference: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, mode: str) -> pd.DataFrame:
    """Restrict the reference to the same calendar months as the monitoring window.

    Workload has a strong seasonal structure (pre-season ramp, congested
    winter, run-in). Comparing a pre-season window with a whole-season
    reference flags "drift" that is just the calendar, so by default we
    compare like with like.
    """
    if mode != "seasonal" or "date" not in reference.columns:
        return reference
    months = set(pd.date_range(start, end, freq="D").month)
    sub = reference[pd.to_datetime(reference["date"]).dt.month.isin(months)]
    return sub if len(sub) >= 500 else reference


def run_monitoring(service=None, start=None, end=None, out_dir: str | Path | None = None) -> dict:
    """Compute and persist a drift report. ``service`` is a RiskService (created if omitted)."""
    from src.service import RiskService

    svc = service or RiskService()
    cfg, meta = svc.cfg, svc.meta
    start = pd.Timestamp(start or cfg["split"]["production"]["start"])
    end = pd.Timestamp(end or svc.latest_date)
    f = svc.features
    dates = f.index.get_level_values("date")
    cur = f[(dates >= start) & (dates <= end) & f["eligible"].to_numpy()].copy()
    from src.features.labels import attach_labels, build_labels

    labels = build_labels(svc.ds, panel=svc.panel)
    cur = attach_labels(cur, labels)
    reference = seasonal_reference(svc.bundle["reference"], start, end, cfg["monitoring"].get("reference_mode",
                                                                                                "seasonal"))
    report = build_report(reference, cur, meta, cfg, meta["target"], (str(start.date()), str(end.date())))
    report["reference"]["mode"] = cfg["monitoring"].get("reference_mode", "seasonal")
    out = Path(out_dir or Path(cfg["paths"]["reports_dir"]) / "monitoring")
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "drift_report.html").write_text(render_html(report))
    report["evidently_report"] = evidently_report(reference, cur, report and [r["feature"] for r in
                                                                              report["feature_drift"]],
                                                  out / "evidently_drift_report.html")
    logger.info("Monitoring report → %s (status: %s)", out, report["summary"]["overall_status"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate drift / monitoring report")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    r = run_monitoring(start=args.start, end=args.end)
    print(json.dumps(r["summary"], indent=2))


if __name__ == "__main__":
    main()
