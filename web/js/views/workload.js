import { getAthlete } from "../data.js";
import { chart, line, series } from "../charts.js";
import { card, css, esc, fmtDate, num } from "../ui.js";
import { athleteHeader } from "./athlete.js";

export async function workloadView(ctx) {
  const id = ctx.params.id;
  const d = await getAthlete(id);
  if (ctx.stale()) return;
  const w = d.workload;
  const days = [28, 56, 90, 180].includes(+ctx.query.get("days")) ? +ctx.query.get("days") : 90;
  const start = Math.max(0, w.date.length - days);
  const sl = (k) => w[k].slice(start);
  const labels = sl("date").map((x) => fmtDate(x, { day: "numeric", month: "short" }));
  ctx.title(`${id} · Workload`);
  ctx.crumbs([["Athletes", "/athletes"], [esc(id), `/athletes/${id}`], ["Workload"]]);
  const last = (k) => w[k][w[k].length - 1];
  ctx.main.innerHTML = `${athleteHeader(id, d, "workload")}
    <div class="page-head" style="margin-bottom:14px">
      <p style="margin:0">Daily GPS-derived load and rolling workload features up to ${fmtDate(last("date"))}.</p>
      <div class="seg">${[28, 56, 90, 180].map((n) => `<a href="/athletes/${id}/workload?days=${n}" data-link class="${n === days ? "active" : ""}">${n} days</a>`).join("")}</div>
    </div>
    <div class="kpis">
      ${[["7-day player load", num(last("player_load_7d")), `chronic weekly ${num(last("player_load_28d") / 4)}`],
         ["7-day sprint distance", `${num(last("sprint_distance_7d"))} m`, `normal ${num(last("sprint_distance_baseline_weekly"))} m/week`],
         ["ACWR (rolling / EWMA)", `${num(last("acwr_total_load"), 2)} / ${num(last("ewma_acwr_total_load"), 2)}`, "context only – contested metric"],
         ["Rest days (7 d)", num(last("rest_days_7d")), `monotony ${num(last("monotony_7d"), 2)}`]]
        .map(([a, b, c]) => `<div class="kpi"><div class="label">${a}</div><div class="value">${b}</div><div class="sub">${c}</div></div>`).join("")}
    </div>
    <div class="grid g-2">
      ${card("Acute vs chronic load", `<div class="chart-box"><canvas id="c1"></canvas></div>`, "player load, AU per week")}
      ${card("Acute:chronic ratios", `<div class="chart-box"><canvas id="c2"></canvas></div>`, "shaded: commonly cited 0.8–1.3 band (contested)")}
      ${card("Sprint exposure vs personal baseline", `<div class="chart-box"><canvas id="c3"></canvas></div>`, "metres per 7 days")}
      ${card("High-speed running & accelerations", `<div class="chart-box"><canvas id="c4"></canvas></div>`, "7-day totals")}
      ${card("Daily load", `<div class="chart-box"><canvas id="c5"></canvas></div>`, "player load per day · matches highlighted")}
      ${card("Session RPE", `<div class="chart-box"><canvas id="c6"></canvas></div>`, "CR-10, session days")}
    </div>
    <p class="callout mt">The ACWR is shown because practitioners use it, not because the model relies on it. Its interpretation is debated
      (mathematical coupling, unstable ratios at low chronic load); see <a href="/about" data-link>Methodology</a>.</p>`;

  const [s1, s2, s3, , s7] = series();
  chart(document.getElementById("c1"), { labels, yMin: 0, yFormat: (v) => num(v), datasets: [
    line("Acute (7-day)", sl("player_load_7d"), s1), line("Chronic (28-day ÷ 4)", sl("player_load_28d").map((v) => v / 4), s2)] });
  const band = (v) => sl("date").map(() => v);
  chart(document.getElementById("c2"), { labels, yMin: 0, yFormat: (v) => num(v, 1), tooltipFormat: (v) => num(v, 2), datasets: [
    { label: "band low", data: band(0.8), borderWidth: 0, pointRadius: 0, hideLegend: true, hideTooltip: true },
    { label: "band", data: band(1.3), borderWidth: 0, pointRadius: 0, fill: "-1", backgroundColor: `${css("--muted")}22`, hideLegend: true, hideTooltip: true },
    line("Rolling (coupled)", sl("acwr_total_load"), s1), line("EWMA", sl("ewma_acwr_total_load"), s2),
    line("Sprint (rolling)", sl("acwr_sprint_load"), s3)] });
  chart(document.getElementById("c3"), { labels, yMin: 0, yFormat: (v) => num(v), datasets: [
    line("Sprint distance (7 d)", sl("sprint_distance_7d"), s1),
    line("Personal baseline", sl("sprint_distance_baseline_weekly"), css("--muted"), { borderDash: [6, 4] })] });
  chart(document.getElementById("c4"), { labels, yMin: 0, yFormat: (v) => num(v), datasets: [
    line("HSR distance (7 d, m)", sl("high_speed_distance_7d"), s1), line("Accelerations (7 d)", sl("accelerations_7d"), s2),
    line("Decelerations (7 d)", sl("decelerations_7d"), s3)] });
  chart(document.getElementById("c5"), { type: "bar", labels, legend: false, yMin: 0, yFormat: (v) => num(v), datasets: [
    { label: "Player load", data: sl("player_load"), backgroundColor: sl("is_match").map((m) => (m ? s2 : s1)) }] });
  chart(document.getElementById("c6"), { type: "bar", labels, legend: false, yMin: 0, yMax: 10, datasets: [
    { label: "RPE", data: sl("rpe"), backgroundColor: s7 }] });
}
