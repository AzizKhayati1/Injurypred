import { getClubs, getInjuries, getPlayers } from "../data.js";
import { crest, portrait, severityChip, statusChip } from "../components.js";
import { chart, series } from "../charts.js";
import { card, esc, fmtDate, kpi, num } from "../ui.js";
import { newsItem } from "./home.js";

export async function injuriesView(ctx) {
  const [inj, { clubs }, { players }] = await Promise.all([getInjuries(), getClubs(), getPlayers()]);
  ctx.title("Injury report");
  ctx.crumbs([["Injury report"]]);
  const byId = Object.fromEntries(players.map((p) => [p.athlete_id, p]));
  const team = clubs.some((c) => c.team_id === ctx.query.get("club")) ? ctx.query.get("club") : "";
  const shown = clubs.filter((c) => !team || c.team_id === team);
  const top = inj.by_diagnosis.slice(0, 10);
  const totalDays = inj.by_diagnosis.reduce((s, r) => s + r.total_days_lost, 0);
  ctx.main.innerHTML = `
    <div class="page-head"><div><h1>Injury report</h1><p>Current absences as of ${fmtDate(inj.as_of)}. Diagnoses as recorded by medical staff (synthetic demo data);
      estimated return dates are clinical estimates, not model outputs.</p></div>
      <div class="seg"><a href="/injuries" data-link class="${team ? "" : "active"}">All clubs</a>${clubs.map((c) => `<a href="/injuries?club=${c.team_id}" data-link class="${c.team_id === team ? "active" : ""}">${esc(c.short_name)}</a>`).join("")}</div></div>
    <div class="kpis">
      ${kpi("Currently unavailable", num(inj.current.filter((x) => !team || x.team_id === team).length), "injury or illness")}
      ${kpi("Out", num(inj.current.filter((x) => x.status === "Out" && (!team || x.team_id === team)).length), "expected back in > 7 days")}
      ${kpi("Day-to-day", num(inj.current.filter((x) => x.status === "Day-to-day" && (!team || x.team_id === team)).length), "expected back within 7 days")}
      ${kpi("Injury days lost", num(totalDays), "all clubs, all seasons")}
    </div>
    <div class="grid g-21">
      <div class="grid" style="align-content:start">${shown.map((c) => {
        const rows = inj.current.filter((x) => x.team_id === c.team_id);
        return card(`<span style="display:inline-flex;gap:8px;align-items:center">${crest(c, 22)} ${esc(c.name)}</span>`, rows.length ? `<div class="table-wrap"><table>
          <thead><tr><th>Name</th><th>Pos</th><th>Est. return</th><th>Status</th><th>Comment</th></tr></thead>
          <tbody>${rows.map((x) => `<tr><td><div class="pcell">${portrait(byId[x.athlete_id] || x, 30)}<a class="nm" href="/players/${x.athlete_id}/injuries" data-link>${esc(x.name)}</a></div></td>
            <td>${esc(x.position)}</td><td>${fmtDate(x.return_date, { day: "numeric", month: "short" })}</td><td>${statusChip(x.status)}</td>
            <td class="wrap">${esc(x.diagnosis || x.absence_type)}${x.side ? ` (${esc(x.side.toLowerCase())})` : ""} ${severityChip(x.severity)}${x.recurrence ? ` <span class="pill pill-warn">${esc(x.recurrence)}</span>` : ""}
              <div class="small muted">Out since ${fmtDate(x.onset_date, { day: "numeric", month: "short" })} · ${x.days_out} day(s)</div></td></tr>`).join("")}</tbody></table></div>`
          : `<p class="muted">No players currently unavailable.</p>`, `${rows.length} unavailable`);
      }).join("")}</div>
      <div class="grid" style="align-content:start">
        ${card("Injuries by type", `<div class="chart-box tall"><canvas id="c-dx"></canvas></div>`, "all seasons · count")}
        ${card("Recent news", `<ul class="feed">${inj.news.filter((n) => !team || n.team_id === team).slice(0, 10).map((n) => newsItem(n, byId)).join("") || "<li>No recent events</li>"}</ul>`)}
      </div>
    </div>`;
  const [, s2] = series();
  chart(document.getElementById("c-dx"), { type: "bar", indexAxis: "y", legend: false, labels: top.map((r) => r.label),
    yFormat: (v) => num(v), datasets: [{ label: "Injuries", data: top.map((r) => r.count), backgroundColor: s2 }] });
}
