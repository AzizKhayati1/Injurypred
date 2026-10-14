import { getAthlete } from "../data.js";
import { chart, line, series, threshold } from "../charts.js";
import { CAT_VAR, badge, card, css, esc, fmtDate, kpi, num, pct, signed } from "../ui.js";

const POS = { GK: "Goalkeeper", CB: "Centre-back", FB: "Full-back", CM: "Central midfield", W: "Winger", ST: "Striker" };
const METRIC_LABEL = { player_load: ["Player load", "AU"], total_distance: ["Total distance", "m"],
  high_speed_distance: ["High-speed running", "m"], sprint_distance: ["Sprint distance", "m"],
  duration: ["Session minutes", "min"], accelerations: ["Accelerations", ""] };

export function athleteHeader(id, d, active) {
  const p = d.profile;
  const pred = d.prediction || {};
  const status = p.availability_status === "available" ? badge(pred.risk_category, "lg")
    : badge("UNAVAILABLE", "lg");
  const tabs = [["overview", "Overview", `/athletes/${id}`], ["workload", "Workload", `/athletes/${id}/workload`],
                ["whatif", "What-if simulator", `/simulator/${id}`]];
  return `
    <div class="ath-head">
      <div class="ath-id">
        <div class="avatar">${esc(p.position)}</div>
        <div><h1>${esc(id)}</h1>
          <div class="muted">${POS[p.position] || esc(p.position)} · Team ${esc(p.team_id)} · age ${num(p.age, 0)} · ${p.sessions_tracked} sessions tracked</div></div>
      </div>
      <div>${status}</div>
    </div>
    <nav class="tabs" aria-label="Athlete sections">${tabs.map(([k, label, href]) =>
      `<a href="${href}" data-link ${k === active ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav>`;
}

function factorList(items, maxImpact, dir) {
  if (!items?.length) return `<p class="muted">No factor above the reporting threshold.</p>`;
  return `<ul class="factors">${items.map((f) => `
    <li class="factor ${dir}">
      <div><div class="txt">${esc(f.explanation)}</div><div class="meta">${esc(f.label)} · SHAP ${signed(f.impact, 3)}</div></div>
      <div class="impact" title="relative contribution"><i style="width:${Math.max(6, 100 * Math.abs(f.impact) / maxImpact)}%"></i></div>
    </li>`).join("")}</ul>`;
}

export async function athleteView(ctx) {
  const id = ctx.params.id;
  const d = await getAthlete(id);
  if (ctx.stale()) return;
  const p = d.profile, pred = d.prediction || {};
  ctx.title(id);
  ctx.crumbs([["Athletes", "/athletes"], [esc(id)]]);
  const available = !pred.unavailable_reason;
  const allF = [...(pred.top_risk_factors || []), ...(pred.protective_factors || [])];
  const maxImpact = Math.max(1e-6, ...allF.map((f) => Math.abs(f.impact)));

  const predictionBlock = available ? `
    <div class="kpis">
      ${kpi("7-day availability risk", pct(pred.risk_probability), `${fmtDate(pred.risk_window.start, { day: "numeric", month: "short" })} – ${fmtDate(pred.risk_window.end, { day: "numeric", month: "short" })}`)}
      ${kpi("Category", badge(pred.risk_category, "lg"), "relative to base rate")}
      ${kpi("Relative risk", `${num(pred.relative_risk, 1)}×`, `base rate ${pct(pred.base_rate)}`)}
      ${kpi("Percentile", num(pred.risk_percentile, 0), "of reference predictions")}
    </div>
    <p class="summary">${esc(pred.summary)}</p>
    ${(pred.warnings || []).map((w) => `<p class="callout">${esc(w)}</p>`).join("")}
    <div class="grid g-2">
      ${card("▲ Associated with higher risk", factorList(pred.top_risk_factors, maxImpact, "up"), "SHAP")}
      ${card("▼ Associated with lower risk", factorList(pred.protective_factors, maxImpact, "down"), "SHAP")}
    </div>
    <p class="callout info mt">${esc(pred.calibration?.interpretation || "")} Test ROC-AUC ${num(pred.calibration?.test_roc_auc, 3)} ·
      Brier ${num(pred.calibration?.test_brier, 4)}. ${esc(pred.disclaimer)}</p>`
    : `<p class="callout">${esc(pred.unavailable_reason)}</p>`;

  const bl = Object.entries(p.baseline || {});
  ctx.main.innerHTML = `${athleteHeader(id, d, "overview")}
    ${predictionBlock}
    <div class="mt">${card("Risk history", `<div class="chart-box tall"><canvas id="c-hist"></canvas></div>
      <p class="muted small">Gaps: athlete unavailable or off-season (the model is only defined for available, in-season days). ✕ = absence onset.</p>`, "last 12 months")}</div>
    <div class="grid g-2 mt">
      ${card("This week vs personal baseline", `<div class="table-wrap"><table>
        <thead><tr><th>Metric</th><th class="num">Last 7 days</th><th class="num">Normal week</th><th class="num">Deviation</th><th>Status</th></tr></thead>
        <tbody>${bl.map(([m, v]) => `<tr><td>${METRIC_LABEL[m]?.[0] || esc(m)}</td><td class="num">${num(v.current_7d)}</td>
          <td class="num">${num(v.baseline_weekly)}</td><td class="num">${signed(v.deviation_pct, 0, "%")}</td><td>${esc(v.status)}</td></tr>`).join("")}</tbody></table></div>
        <p class="muted small">Baseline = athlete's own trailing 12 weeks, ending 7 days ago, available days only.</p>`, "athlete-specific")}
      ${card("Absence history", p.injury_history?.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Onset</th><th>Type</th><th>Category</th><th class="num">Days lost</th></tr></thead>
        <tbody>${[...p.injury_history].reverse().map((h) => `<tr><td>${fmtDate(h.onset_date)}</td><td>${esc(h.absence_type)}</td>
          <td>${esc((h.category || "").replace("_", " "))}</td><td class="num">${h.days_lost}</td></tr>`).join("")}</tbody></table></div>`
        : `<p class="muted">No absences recorded.</p>`, `${p.prior_injury_count_before_dataset} prior injuries before the dataset`)}
    </div>`;

  const h = d.risk_history;
  const labels = h.date.map((x) => fmtDate(x, { day: "numeric", month: "short", year: "2-digit" }));
  const th = pred.thresholds || {};
  const [s1, , , , , s8] = series();
  const n = labels.length;
  chart(document.getElementById("c-hist"), {
    labels, yMin: 0, yFormat: (v) => pct(v, 0), tooltipFormat: (v) => pct(v), xTicks: 10,
    datasets: [
      line("Predicted 7-day risk", h.risk_probability, s1),
      { type: "line", label: "Absence onset", data: h.absence_onset.map((on) => (on ? 0 : null)), showLine: false,
        pointStyle: "crossRot", pointRadius: 7, pointBorderWidth: 2, borderColor: s8, backgroundColor: s8 },
      ...Object.entries(th).map(([c, v]) => threshold(`${c.replace("_", " ")} threshold`, v, n, css(CAT_VAR[c]))),
    ],
  });
}
