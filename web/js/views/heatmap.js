import { getHeatmap, getMeta, getPlayers, getSquad } from "../data.js";
import { navigate } from "../router.js";
import { card, esc, fmtDate, hideTip, pct, rampColor, showTip } from "../ui.js";

export async function heatmapView(ctx) {
  const [meta, squad, hm, { players }] = await Promise.all([getMeta(), getSquad(), getHeatmap(), getPlayers()]);
  const nameOf = Object.fromEntries(players.map((p) => [p.athlete_id, p]));
  ctx.title("Risk heatmap");
  ctx.crumbs([["Risk heatmap"]]);
  const team = meta.teams.includes(ctx.query.get("team")) ? ctx.query.get("team") : "";
  const days = [14, 28, 56].includes(+ctx.query.get("days")) ? +ctx.query.get("days") : 28;
  const teamOf = Object.fromEntries(squad.athletes.map((a) => [a.athlete_id, a]));
  const startIdx = Math.max(0, hm.dates.length - days);
  const dates = hm.dates.slice(startIdx);
  let rows = hm.athletes.map((a, i) => ({ id: a, info: teamOf[a], v: hm.values[i].slice(startIdx) }))
    .filter((r) => !team || r.info?.team_id === team);
  const lastVal = (r) => [...r.v].reverse().find((x) => x != null) ?? -1;
  rows.sort((a, b) => lastVal(b) - lastVal(a));
  const vmax = Math.max(0.3, ...rows.flatMap((r) => r.v.filter((x) => x != null)));
  const q = (t, d) => `/heatmap?${new URLSearchParams({ ...(t ? { team: t } : {}), days: d })}`;

  ctx.main.innerHTML = `
    <div class="page-head">
      <div><h1>Risk heatmap</h1><p>Each cell is an athlete's estimated 7-day risk at the end of that day. Sorted by the latest estimate; click an athlete to open their profile.</p></div>
      <div class="controls">
        <div class="seg">${["", ...meta.teams].map((t) => `<a href="${q(t, days)}" data-link class="${t === team ? "active" : ""}">${t ? esc(players.find((p) => p.team_id === t)?.club.short_name || t) : "All"}</a>`).join("")}</div>
        <div class="seg">${[14, 28, 56].map((n) => `<a href="${q(team, n)}" data-link class="${n === days ? "active" : ""}">${n} d</a>`).join("")}</div>
      </div>
    </div>
    ${card(`${rows.length} athletes · ${fmtDate(dates[0])} – ${fmtDate(dates[dates.length - 1])}`, `
      <div class="legend-ramp" style="margin-bottom:12px"><span>0%</span><span class="ramp"></span><span>${pct(vmax, 0)}+</span>
        <span class="cell na" style="display:inline-block;width:16px;height:12px;margin-left:12px"></span><span>unavailable / off-season</span></div>
      <div class="table-wrap"><div class="heat" id="heat" style="grid-template-columns: 120px repeat(${dates.length}, minmax(12px, 1fr)); min-width:${120 + dates.length * 14}px">
        ${rows.map((r) => `<div class="rl"><a href="/players/${r.id}" data-link>${esc(nameOf[r.id] ? `${nameOf[r.id].first_name[0]}. ${nameOf[r.id].last_name}` : r.id)}</a></div>${r.v.map((x, j) =>
          `<div class="cell ${x == null ? "na" : ""}" data-a="${r.id}" data-j="${j}" ${x == null ? "" : `style="background:${rampColor(x / vmax)}"`}></div>`).join("")}`).join("")}
        <div></div>${dates.map((dt, j) => `<div class="cl">${j % Math.ceil(dates.length / 14) === 0 ? fmtDate(dt, { day: "numeric", month: "short" }) : ""}</div>`).join("")}
      </div></div>`)}`;

  const grid = document.getElementById("heat");
  const byId = Object.fromEntries(rows.map((r) => [r.id, r]));
  grid.addEventListener("mousemove", (e) => {
    const cell = e.target.closest(".cell");
    if (!cell) return hideTip();
    const r = byId[cell.dataset.a], j = +cell.dataset.j, x = r.v[j];
    showTip(`<strong>${esc(nameOf[r.id]?.name || r.id)}</strong> · ${esc(r.info?.position || "")} · ${esc(r.info?.team_id || "")}<br>${fmtDate(dates[j])}<br>${x == null ? "unavailable / not scored" : `risk ${pct(x)}`}`, e.clientX, e.clientY);
  });
  grid.addEventListener("mouseleave", hideTip);
  grid.addEventListener("click", (e) => { const cell = e.target.closest(".cell"); if (cell) navigate(`/players/${cell.dataset.a}`); });
}
