// Player pages: hero banner + tabs (Overview · Injury history · Prevention · Workload · What-if).
import { getAthlete } from "../data.js";
import { chart, line, series, threshold } from "../charts.js";
import { POS_LABEL, bodyMap, crest, portrait, severityChip, statusChip } from "../components.js";
import { CAT_VAR, badge, card, css, esc, fmtDate, hideTip, kpi, num, pct, showTip, signed } from "../ui.js";

const METRIC_LABEL = { player_load: "Player load", total_distance: "Total distance", high_speed_distance: "High-speed running",
  sprint_distance: "Sprint distance", duration: "Session minutes", accelerations: "Accelerations" };

export async function loadPlayer(ctx) {
  const id = ctx.params.id;
  const d = await getAthlete(id);
  return ctx.stale() ? null : d;
}

export function playerCrumbs(ctx, d, last) {
  const p = d.player;
  ctx.title(last ? `${p.name} · ${last}` : p.name);
  ctx.crumbs([["Players", "/players"], [esc(p.club.name), `/clubs/${p.team_id}`],
    ...(last ? [[esc(p.name), `/players/${p.athlete_id}`], [last]] : [[esc(p.name)]])]);
}

export function playerHero(d, active) {
  const p = d.player, c = p.club, pred = d.prediction || {};
  const available = !pred.unavailable_reason;
  const id = p.athlete_id;
  const tabs = [["overview", "Overview", ""], ["injuries", "Injury history", "/injuries"], ["prevention", "Prevention", "/prevention"],
                ["workload", "Workload", "/workload"], ["simulator", "What-if", "/simulator"]];
  const facts = [["Age", num(p.age, 0)], ["Height", p.height_cm ? `${p.height_cm} cm` : "–"], ["Weight", p.weight_kg ? `${p.weight_kg} kg` : "–"],
    ["Nationality", esc(p.nationality || "–")], ["Foot", esc(p.preferred_foot || "–")], ["Joined", p.joined ? fmtDate(p.joined, { month: "short", year: "numeric" }) : "–"]];
  const side = available
    ? `<div class="risk-label">7-day availability risk</div><div class="risk-big">${pct(pred.risk_probability)}</div>${badge(pred.risk_category, "lg")}`
    : `<div class="risk-label">Status</div>${statusChip(p.status)}<div class="small">${esc(p.detail || "")}${p.expected_return ? ` · est. return ${fmtDate(p.expected_return, { day: "numeric", month: "short" })}` : ""}</div>`;
  return `
    <section class="hero" style="--c1:${c.primary_color};--c2:${c.secondary_color}">
      <div class="hero-inner">
        ${portrait(p, 170)}
        <div class="who">
          <div class="kicker">${crest(c, 22)} <a href="/clubs/${c.team_id}" data-link style="color:inherit">${esc(c.name)}</a> · #${esc(p.shirt_number ?? "")} · ${POS_LABEL[p.position] || esc(p.position)}</div>
          <h1><span class="first">${esc(p.first_name || "")}</span>${esc(p.last_name || p.name)}</h1>
          <div class="facts">${facts.map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("")}</div>
        </div>
        <div class="side"><div class="bignum">${esc(p.shirt_number ?? "")}</div>${side}</div>
      </div>
    </section>
    <nav class="tabs" aria-label="Player sections">${tabs.map(([k, label, suffix]) =>
      `<a href="/players/${id}${suffix}" data-link ${k === active ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav>`;
}

function statStrip(p) {
  const s = p.stats.season || {}, inj = p.injury_summary || {};
  const items = [[`${s.name || "Season"} apps`, num(s.appearances)], ["Starts", num(s.starts)], ["Minutes", num(s.minutes)],
    ["Distance (km)", num(s.distance_km, 0)], ["Top speed (km/h)", s.top_speed_kmh ? `${num(s.top_speed_kmh, 1)}` : "–"],
    ["Injuries (12 mo)", num(inj.injuries_last_365d)], ["Days lost (12 mo)", num(inj.days_lost_last_365d)]];
  return `<div class="statstrip">${items.map(([k, v]) => `<div><b>${v}</b><span>${k}</span></div>`).join("")}</div>`;
}

function factorList(items, maxImpact, dir) {
  if (!items?.length) return `<p class="muted">No factor above the reporting threshold.</p>`;
  return `<ul class="factors">${items.map((f) => `
    <li class="factor ${dir}">
      <div><div class="txt">${esc(f.explanation)}</div><div class="meta">${esc(f.label)} · SHAP ${signed(f.impact, 3)}</div></div>
      <div class="impact"><i style="width:${Math.max(6, 100 * Math.abs(f.impact) / maxImpact)}%"></i></div>
    </li>`).join("")}</ul>`;
}

// ------------------------------------------------------------------ overview
export async function playerOverviewView(ctx) {
  const d = await loadPlayer(ctx);
  if (!d) return;
  playerCrumbs(ctx, d);
  const p = d.player, pred = d.prediction || {}, prof = d.profile;
  const available = !pred.unavailable_reason;
  const allF = [...(pred.top_risk_factors || []), ...(pred.protective_factors || [])];
  const maxImpact = Math.max(1e-6, ...allF.map((f) => Math.abs(f.impact)));
  const recent = (d.injuries.records || []).slice(0, 4);
  const topRec = (d.prevention.recommendations || []).slice(0, 3);

  ctx.main.innerHTML = `${playerHero(d, "overview")}
    ${statStrip(p)}
    ${available ? `<p class="summary">${esc(pred.summary)}</p>` : `<p class="callout">${esc(pred.unavailable_reason)}</p>`}
    <div class="grid g-21">
      <div class="grid">
        ${available ? `<div class="grid g-2">
          ${card("Associated with higher risk", factorList(pred.top_risk_factors, maxImpact, "up"), "SHAP")}
          ${card("Associated with lower risk", factorList(pred.protective_factors, maxImpact, "down"), "SHAP")}</div>` : ""}
        ${card("Risk history", `<div class="chart-box tall"><canvas id="c-hist"></canvas></div>
          <p class="muted small">Gaps: unavailable or off-season. ✕ = absence onset.</p>`, "last 12 months")}
        ${card("This week vs personal baseline", `<div class="table-wrap"><table>
          <thead><tr><th>Metric</th><th class="num">Last 7 days</th><th class="num">Normal week</th><th class="num">Deviation</th><th>Status</th></tr></thead>
          <tbody>${Object.entries(prof.baseline || {}).map(([m, v]) => `<tr><td>${METRIC_LABEL[m] || esc(m)}</td><td class="num">${num(v.current_7d)}</td>
            <td class="num">${num(v.baseline_weekly)}</td><td class="num">${signed(v.deviation_pct, 0, "%")}</td><td>${esc(v.status)}</td></tr>`).join("")}</tbody></table></div>`,
          "athlete-specific")}
      </div>
      <div class="grid" style="align-content:start">
        ${card("Recent injuries", recent.length ? `<ul class="feed">${recent.map((r) => `<li style="grid-template-columns:1fr">
          <div><div class="hl">${esc(r.diagnosis)}${r.side ? ` (${esc(r.side.toLowerCase())})` : ""}</div>
          <div class="meta">${fmtDate(r.onset_date)} · ${r.days_lost} days · ${severityChip(r.severity)} ${r.status === "Ongoing" ? statusChip("Out") : ""}</div></div></li>`).join("")}</ul>
          <a href="/players/${p.athlete_id}/injuries" data-link>Full injury history →</a>` : `<p class="muted">No absences recorded.</p>`)}
        ${card("Prevention focus", `${topRec.map((r) => `<p><span class="pill">${esc(r.priority)}</span> <strong>${esc(r.title)}</strong><br><span class="small muted">${esc(r.why)}</span></p>`).join("")}
          <a href="/players/${p.athlete_id}/prevention" data-link>All recommendations →</a>`)}
        ${available ? `<p class="callout info">${esc(pred.calibration?.interpretation || "")} ${esc(pred.disclaimer)}</p>` : ""}
      </div>
    </div>`;

  const h = d.risk_history;
  const labels = h.date.map((x) => fmtDate(x, { day: "numeric", month: "short", year: "2-digit" }));
  const [s1, , , , , s8] = series();
  chart(document.getElementById("c-hist"), {
    labels, yMin: 0, yFormat: (v) => pct(v, 0), tooltipFormat: (v) => pct(v), xTicks: 10,
    datasets: [line("Predicted 7-day risk", h.risk_probability, s1),
      { type: "line", label: "Absence onset", data: h.absence_onset.map((on) => (on ? 0 : null)), showLine: false,
        pointStyle: "crossRot", pointRadius: 7, pointBorderWidth: 2, borderColor: s8, backgroundColor: s8 },
      ...Object.entries(pred.thresholds || d.risk_thresholds || {}).map(([c, v]) => threshold(`${c.replace("_", " ")} threshold`, v, labels.length, css(CAT_VAR[c])))],
  });
}

// ------------------------------------------------------------------ injury history
export async function playerInjuriesView(ctx) {
  const d = await loadPlayer(ctx);
  if (!d) return;
  playerCrumbs(ctx, d, "Injury history");
  const inj = d.injuries, recs = inj.records || [];
  const injuries = recs.filter((r) => r.absence_type === "injury");
  const byRegion = {};
  injuries.forEach((r) => { byRegion[r.body_region || r.category] = (byRegion[r.body_region || r.category] || 0) + 1; });
  ctx.main.innerHTML = `${playerHero(d, "injuries")}
    <div class="kpis">
      ${kpi("Injuries recorded", num(inj.n_injuries), `${num(inj.n_illnesses)} illnesses`)}
      ${kpi("Days lost to injury", num(inj.days_lost_injury), "all recorded history")}
      ${kpi("Last 12 months", num(inj.injuries_last_365d), `${num(inj.days_lost_last_365d)} days lost`)}
      ${kpi("Recurrences", num(inj.recurrences), "same diagnosis & side")}
    </div>
    <div class="grid g-12">
      ${card("Body map", `${bodyMap(inj.body_map || {}, injuries)}
        <p class="muted small" style="text-align:center">Shaded areas = recorded injury sites (darker = more often). Hover for details.</p>
        <div class="catbars" style="margin-top:10px">${Object.entries(byRegion).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
          `<div class="catbar"><span class="small">${esc(k)}</span><div class="track"><div class="fill" style="width:${100 * v / Math.max(...Object.values(byRegion))}%;background:var(--very-high)"></div></div><span class="n">${v}</span></div>`).join("")}</div>`,
        "injuries only")}
      ${card("Injury log", recs.length ? `<div class="table-wrap"><table>
        <thead><tr><th>Date</th><th>Diagnosis</th><th>Side</th><th>Mechanism</th><th>Setting</th><th>Severity</th><th class="num">Days</th><th>Status</th></tr></thead>
        <tbody>${recs.map((r) => `<tr><td>${fmtDate(r.onset_date)}</td>
          <td class="wrap"><strong>${esc(r.diagnosis)}</strong>${r.recurrence ? ` <span class="pill pill-warn">${esc(r.recurrence)}</span>` : ""}<div class="small muted">${esc(r.body_region || "")}</div></td>
          <td>${esc(r.side || "–")}</td><td>${esc(r.mechanism || "–")}</td><td>${esc(r.occurred_in || "–")}</td><td>${severityChip(r.severity)}</td>
          <td class="num">${r.days_lost}</td><td>${r.status === "Ongoing" ? `${statusChip("Out")}<div class="small muted">est. ${fmtDate(r.return_date, { day: "numeric", month: "short" })}</div>` : `<span class="small">Returned ${fmtDate(r.return_date, { day: "numeric", month: "short", year: "2-digit" })}</span>`}</td></tr>`).join("")}</tbody></table></div>`
        : `<p class="muted">No absences recorded.</p>`, `${num(inj.recurrences)} recurrences`)}
    </div>
    <p class="callout mt">${esc(inj.note)} Severity follows the football consensus time-loss bands (slight ≤ 3 days, minimal 4–7, mild 8–28, moderate 29–90, severe &gt; 90).
      Muscle-injury grades are indicative from time lost.</p>`;

  ctx.main.querySelectorAll(".bm-zone.hit").forEach((z) => {
    z.addEventListener("mousemove", (e) => showTip(esc(z.dataset.tip), e.clientX, e.clientY));
    z.addEventListener("mouseleave", hideTip);
  });
}

// ------------------------------------------------------------------ prevention
export async function playerPreventionView(ctx) {
  const d = await loadPlayer(ctx);
  if (!d) return;
  playerCrumbs(ctx, d, "Prevention");
  const pv = d.prevention;
  ctx.main.innerHTML = `${playerHero(d, "prevention")}
    <p class="callout">${esc(pv.disclaimer)}</p>
    <div class="grid mt">${pv.recommendations.map((r) => `
      <article class="rec ${esc(r.priority)}">
        <div class="rhead"><h3>${esc(r.title)}</h3><span class="pill">${esc(r.priority)} priority</span></div>
        <div class="tags"><span class="pill">${esc(r.category)}</span><span class="pill">Target: ${esc(r.target)}</span></div>
        <p class="why"><strong>Why this player:</strong> ${esc(r.why)}</p>
        <ul>${r.actions.map((a) => `<li>${esc(a)}</li>`).join("")}</ul>
        <details><summary>Evidence</summary><p>${esc(r.evidence)}</p><ol>${r.references.map((x) => `<li>${esc(x.citation)}</li>`).join("")}</ol></details>
      </article>`).join("")}</div>`;
}
