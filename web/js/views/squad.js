import { getMeta, getPlayers, getSquad } from "../data.js";
import { portrait } from "../components.js";
import { chart, line, series } from "../charts.js";
import { navigate } from "../router.js";
import { CATEGORIES, CAT_LABEL, CAT_VAR, badge, card, esc, fmtDate, kpi, num, pct, riskCell, signed, sortableTable } from "../ui.js";

export async function squadView(ctx) {
  const [meta, squad, { players }] = await Promise.all([getMeta(), getSquad(), getPlayers()]);
  const byId = Object.fromEntries(players.map((p) => [p.athlete_id, p]));
  const team = meta.teams.includes(ctx.query.get("team")) ? ctx.query.get("team") : "ALL";
  const o = squad.overview[team];
  ctx.title("Risk center");
  ctx.crumbs([["Risk center"]]);
  const total = o.n_available || 1;
  const clubName = Object.fromEntries(players.map((p) => [p.team_id, p.club.short_name]));
  const teamTabs = ["ALL", ...meta.teams].map((t) =>
    `<a href="/squad${t === "ALL" ? "" : `?team=${t}`}" data-link class="${t === team ? "active" : ""}">${t === "ALL" ? "All clubs" : esc(clubName[t] || t)}</a>`).join("");

  ctx.main.innerHTML = `
    <div class="page-head">
      <div><h1>Risk center</h1>
        <p>Estimated probability of a new injury/illness absence in the next 7 days, for every available athlete,
        at the end of ${fmtDate(o.date)}.</p></div>
      <div class="seg" role="tablist" aria-label="Team">${teamTabs}</div>
    </div>
    <div class="kpis">
      ${kpi("Players", o.n_athletes, team === "ALL" ? `${meta.teams.length} clubs` : esc(clubName[team] || team))}
      ${kpi("Available", o.n_available, `${o.n_unavailable} unavailable`)}
      ${kpi("High or very high", o.n_high_risk, `≥ ${pct(o.thresholds.HIGH)} estimated risk`)}
      ${kpi("Average 7-day risk", pct(o.mean_risk), `reference base rate ${pct(o.base_rate)}`)}
    </div>
    <div class="grid g-12">
      ${card("Risk categories", `<div class="catbars">${CATEGORIES.map((c) => {
        const n = o.categories[c];
        return `<div class="catbar"><span>${badge(c)}</span><div class="track"><div class="fill" style="width:${100 * n / total}%;background:var(${CAT_VAR[c]})"></div></div><span class="n">${n}</span></div>`;
      }).join("")}</div>
      <p class="muted small" style="margin-top:14px">Categories are multiples of the reference base rate:
      ${CATEGORIES.slice(1).map((c) => `${CAT_LABEL[c]} ≥ ${pct(o.thresholds[c])}`).join(" · ")}.</p>`, "available athletes")}
      ${card("Average predicted risk", `<div class="chart-box"><canvas id="c-risk"></canvas></div>`, "last 8 weeks")}
    </div>
    <div class="grid g-2 mt">
      ${card("Squad workload", `<div class="chart-box short"><canvas id="c-load"></canvas></div>`, "mean 7-day player load (AU)")}
      ${card("Unavailable athletes", `<div class="chart-box short"><canvas id="c-unav"></canvas></div>`, "per day")}
    </div>
    <div class="mt">${card("Highest estimated risk", `<div id="t-top"></div>`, "click a row for the explanation")}</div>`;

  const labels = o.trend.map((r) => fmtDate(r.date, { day: "numeric", month: "short" }));
  const [s1, s2, , , s7] = series();
  chart(document.getElementById("c-risk"), { labels, legend: false, yMin: 0, yFormat: (v) => pct(v, 0), tooltipFormat: (v) => pct(v),
    datasets: [line("Mean risk", o.trend.map((r) => r.mean_risk), s1, { fill: true, backgroundColor: `${s1}22` })] });
  chart(document.getElementById("c-load"), { labels, legend: false, yFormat: (v) => num(v),
    datasets: [line("7-day player load", o.trend.map((r) => r.player_load_7d), s2)] });
  chart(document.getElementById("c-unav"), { type: "bar", labels, legend: false, yMin: 0,
    datasets: [{ label: "Unavailable", data: o.trend.map((r) => r.unavailable), backgroundColor: s7 }] });

  sortableTable(document.getElementById("t-top"), [
    { key: "athlete_id", label: "Player", sort: (r) => byId[r.athlete_id]?.last_name, render: (r) => `<div class="pcell">${portrait(byId[r.athlete_id] || r, 30)}<a class="nm" href="/players/${r.athlete_id}" data-link>${esc(byId[r.athlete_id]?.name || r.athlete_id)}</a></div>` },
    { key: "position", label: "Pos" },
    { key: "team_id", label: "Team" },
    { key: "risk_probability", label: "7-day risk", num: true, render: (r) => riskCell(r.risk_probability) },
    { key: "risk_category", label: "Category", render: (r) => badge(r.risk_category), sort: (r) => CATEGORIES.indexOf(r.risk_category) },
    { key: "acwr_total_load", label: "ACWR", num: true, render: (r) => num(r.acwr_total_load, 2) },
    { key: "sprint_distance_vs_baseline_pct", label: "Sprint vs baseline", num: true, render: (r) => signed(r.sprint_distance_vs_baseline_pct, 0, "%") },
  ], o.highest_risk, { initial: { key: "risk_probability", dir: "desc" }, onRowClick: (r) => navigate(`/players/${r.athlete_id}`) });
}
