import { getMeta } from "../data.js";
import { chart, series } from "../charts.js";
import { card, css, esc, fmtDate, kpi, num, pct } from "../ui.js";

const STATUS = { ok: "LOW", warning: "MODERATE", alert: "VERY_HIGH", unknown: "UNAVAILABLE" };

export async function modelView(ctx) {
  const meta = await getMeta();
  ctx.title("Performance & drift");
  ctx.crumbs([["Model"], ["Performance & drift"]]);
  const m = meta.metrics, t = m.test, ev = meta.evaluation, mi = meta.model;
  const boot = ev.test_bootstrap || {};
  const label = Object.fromEntries(meta.features.map((f) => [f.name, f.label]));
  const reps = Object.values(meta.experiments.workload_representations || {}).filter((r) => r.learner === "lightgbm" && r.view === "with_context")
    .sort((a, b) => b.roc_auc - a.roc_auc);
  const drift = meta.drift;
  const statusBadge = (s) => `<span class="badge ${STATUS[s] || "UNAVAILABLE"}">${esc(s)}</span>`;

  ctx.main.innerHTML = `
    <div class="page-head"><div><h1>Model performance & monitoring</h1>
      <p>${esc(mi.model_type)} v${esc(mi.model_version)} (build ${esc(mi.build_id)}), trained ${fmtDate(mi.trained_at?.slice(0, 10))} on synthetic data.
      Selected on validation PR-AUC; the test period (Jan–May 2026) was used only for reporting.</p></div></div>
    <div class="kpis">
      ${kpi("Test ROC-AUC", num(t.roc_auc, 3), boot.roc_auc_ci ? `95% CI ${num(boot.roc_auc_ci[0], 2)}–${num(boot.roc_auc_ci[1], 2)}` : "")}
      ${kpi("Test PR-AUC", num(t.pr_auc, 3), `base rate ${pct(t.base_rate)}`)}
      ${kpi("Brier score", num(t.brier, 4), `calibration error ${num(t.ece, 3)}`)}
      ${kpi("Top-10% review", `${num(ev.test_top_k?.top_10pct?.lift, 1)}×`, `captures ${pct(ev.test_top_k?.top_10pct?.recall, 0)} of events`)}
    </div>
    <div class="grid g-2">
      ${card("Model comparison", `<div class="table-wrap"><table><thead><tr><th>Model</th><th class="num">Val ROC-AUC</th><th class="num">Val PR-AUC</th><th class="num">Test ROC-AUC</th><th class="num">Test PR-AUC</th><th class="num">Test Brier</th></tr></thead>
        <tbody>${Object.entries(m.model_comparison).map(([n, r]) => `<tr><td>${n === m.model_type ? `<strong>${esc(n)} ★</strong>` : esc(n)}</td>
          <td class="num">${num(r.validation.roc_auc, 3)}</td><td class="num">${num(r.validation.pr_auc, 3)}</td><td class="num">${num(r.test.roc_auc, 3)}</td>
          <td class="num">${num(r.test.pr_auc, 3)}</td><td class="num">${num(r.test.brier, 4)}</td></tr>`).join("")}</tbody></table></div>`, "★ champion")}
      ${card("Calibration (test period)", `<div class="chart-box"><canvas id="c-cal"></canvas></div>`, "predicted vs observed, deciles")}
      ${card("Sensitivity at risk thresholds", `<div class="table-wrap"><table><thead><tr><th>Flag at ≥</th><th class="num">Threshold</th><th class="num">Flagged</th><th class="num">Sensitivity</th><th class="num">Precision</th></tr></thead>
        <tbody>${Object.entries(ev.test_category_thresholds || {}).map(([c, r]) => `<tr><td><span class="badge ${c}">${c.replace("_", " ")}</span></td>
          <td class="num">${pct(r.threshold)}</td><td class="num">${pct(r.flagged_share)}</td><td class="num">${pct(r.recall)}</td><td class="num">${pct(r.precision)}</td></tr>`).join("")}</tbody></table></div>
        <p class="muted small">Most flags are not followed by an absence and most absences are not flagged: use for prioritising attention, not individual decisions.</p>`, "test period")}
      ${card("Forward validation", `<div class="chart-box"><canvas id="c-fwd"></canvas></div>`,
        ev.forward_cv ? `ROC-AUC ${num(ev.forward_cv.roc_auc_mean, 3)} ± ${num(ev.forward_cv.roc_auc_std, 3)}` : "")}
      ${card("What drives the estimates", `<div class="chart-box tall"><canvas id="c-shap"></canvas></div>`, "mean |SHAP|, top 12 features")}
      ${card("Which workload representation helps?", `<div class="table-wrap"><table><thead><tr><th>Feature family (+ context)</th><th class="num">Val ROC-AUC</th><th class="num">95% CI</th></tr></thead>
        <tbody>${reps.map((r) => `<tr><td>${esc(r.family.replaceAll("_", " "))}</td><td class="num">${num(r.roc_auc, 3)}</td>
          <td class="num">${num(r.roc_auc_ci[0], 3)}–${num(r.roc_auc_ci[1], 3)}</td></tr>`).join("")}</tbody></table></div>
        <p class="muted small">Overlapping intervals: no representation (including ACWR) clearly dominates.</p>`, "LightGBM")}
    </div>
    ${drift ? `<div class="mt">${card(`Drift monitoring · ${statusBadge(drift.summary.overall_status)}`, `
      <p>${esc(drift.summary.recommendation)} Window ${fmtDate(drift.current.start)} – ${fmtDate(drift.current.end)} vs ${esc(drift.reference.description)}.</p>
      <div class="kpis" style="margin-top:12px">
        ${kpi("Prediction drift (PSI)", num(drift.prediction_drift.psi, 2), statusBadge(drift.prediction_drift.status))}
        ${drift.performance ? kpi("ROC-AUC on matured labels", num(drift.performance.current_roc_auc, 3), `test reference ${num(drift.performance.reference_test_roc_auc, 3)}`) : ""}
        ${drift.label_shift ? kpi("Event rate", pct(drift.label_shift.current_event_rate), `reference ${pct(drift.label_shift.reference_event_rate)}`) : ""}
      </div>
      <div class="table-wrap"><table><thead><tr><th>Feature</th><th class="num">PSI</th><th>Status</th><th class="num">Reference mean</th><th class="num">Current mean</th></tr></thead>
      <tbody>${drift.feature_drift.map((f) => `<tr><td>${esc(label[f.feature] || f.feature)}</td><td class="num">${num(f.psi, 3)}</td><td>${statusBadge(f.status)}</td>
        <td class="num">${num(f.reference_mean, 2)}</td><td class="num">${num(f.current_mean, 2)}</td></tr>`).join("")}</tbody></table></div>
      <p class="muted small">PSI &lt; 0.10 stable · 0.10–0.25 moderate · &gt; 0.25 significant. The current season includes an intentional high-speed-running emphasis in two synthetic teams.</p>`)}</div>` : ""}`;

  const [s1, s2] = series();
  const cal = m.calibration_table_test;
  const lim = Math.max(...cal.map((r) => Math.max(r.mean_predicted, r.observed_rate))) * 1.05;
  chart(document.getElementById("c-cal"), { type: "scatter", yMin: 0, yMax: lim, yFormat: (v) => pct(v, 0), tooltipFormat: (v) => pct(v),
    datasets: [
      { type: "line", label: "Model", data: cal.map((r) => ({ x: r.mean_predicted, y: r.observed_rate })), borderColor: s1, backgroundColor: s1, pointRadius: 4, tension: 0 },
      { type: "line", label: "Perfect calibration", data: [{ x: 0, y: 0 }, { x: lim, y: lim }], borderColor: css("--muted"), borderDash: [5, 4], borderWidth: 1, pointRadius: 0, hideTooltip: true },
    ] });
  const calChart = Chart.getChart(document.getElementById("c-cal"));
  if (calChart) { calChart.options.scales.x = { type: "linear", min: 0, max: lim, ticks: { color: css("--muted"), callback: (v) => pct(v, 0) }, grid: { display: false } }; calChart.update(); }
  if (ev.forward_cv) chart(document.getElementById("c-fwd"), { type: "bar", labels: ev.forward_cv.folds.map((f) => f.month), yMin: 0.5, yMax: 0.75,
    yFormat: (v) => num(v, 2), datasets: [{ label: "ROC-AUC", data: ev.forward_cv.folds.map((f) => f.roc_auc), backgroundColor: s1 }] });
  const top = Object.entries(ev.shap_importance_top20 || {}).slice(0, 12);
  chart(document.getElementById("c-shap"), { type: "bar", indexAxis: "y", legend: false, labels: top.map(([k]) => label[k] || k),
    yFormat: (v) => num(v, 3), datasets: [{ label: "mean |SHAP|", data: top.map(([, v]) => v), backgroundColor: s2 }] });
}
