import { getClubs, getInjuries, getPlayers } from "../data.js";
import { POS_GROUP, crest, portrait, riskChip, statusChip } from "../components.js";
import { NotFoundError } from "../data.js";
import { card, esc, fmtDate, kpi, num, pct } from "../ui.js";

const GROUP_ORDER = ["Goalkeepers", "Defenders", "Midfielders", "Forwards"];

export async function clubsView(ctx) {
  const { clubs } = await getClubs();
  ctx.title("Clubs");
  ctx.crumbs([["Clubs"]]);
  ctx.main.innerHTML = `<div class="page-head"><div><h1>Clubs</h1><p>${esc(clubs[0]?.league || "")} · all clubs are fictional.</p></div></div>
    <div class="grid g-2">${clubs.map((c) => `
      <a class="hero" href="/clubs/${c.team_id}" data-link style="--c1:${c.primary_color};--c2:${c.secondary_color};display:block;border-radius:6px">
        <div class="hero-inner" style="grid-template-columns:auto 1fr;padding:20px 24px">
          ${crest(c, 84)}
          <div class="who" style="padding:0"><div class="kicker">${esc(c.city || "")} · founded ${esc(c.founded || "")}</div>
            <h1 style="font-size:2.2rem">${esc(c.name)}</h1>
            <div class="facts"><div><span>Squad</span><b>${c.n_players}</b></div><div><span>Unavailable</span><b>${c.n_unavailable}</b></div>
              <div><span>High risk</span><b>${c.n_high_risk}</b></div><div><span>Avg risk</span><b>${pct(c.mean_risk)}</b></div></div></div>
        </div></a>`).join("")}</div>`;
}

export async function clubView(ctx) {
  const [{ clubs }, { players }, inj] = await Promise.all([getClubs(), getPlayers(), getInjuries()]);
  const c = clubs.find((x) => x.team_id === ctx.params.id);
  if (!c) throw new NotFoundError(ctx.params.id);
  ctx.title(c.name);
  ctx.crumbs([["Clubs", "/clubs"], [esc(c.name)]]);
  const squad = players.filter((p) => p.team_id === c.team_id).sort((a, b) => (a.shirt_number ?? 99) - (b.shirt_number ?? 99));
  const current = inj.current.filter((x) => x.team_id === c.team_id);
  const groups = GROUP_ORDER.map((g) => [g, squad.filter((p) => POS_GROUP[p.position] === g)]);
  ctx.main.innerHTML = `
    <section class="hero" style="--c1:${c.primary_color};--c2:${c.secondary_color};border-radius:6px 6px 0 0">
      <div class="hero-inner" style="grid-template-columns:auto 1fr auto;padding-bottom:18px">
        ${crest(c, 110)}
        <div class="who" style="padding:0"><div class="kicker">${esc(c.league || "")}</div><h1>${esc(c.name)}</h1>
          <div class="facts"><div><span>City</span><b>${esc(c.city || "–")}</b></div><div><span>Founded</span><b>${esc(c.founded || "–")}</b></div><div><span>Squad</span><b>${c.n_players}</b></div></div></div>
        <div class="side" style="padding:0"><div class="risk-label">Squad average 7-day risk</div><div class="risk-big">${pct(c.mean_risk)}</div></div>
      </div></section>
    <div class="kpis mt">
      ${kpi("Available", `${c.n_players - c.n_unavailable}/${c.n_players}`, "players")}
      ${kpi("Out / day-to-day", num(c.n_unavailable), current.map((x) => esc(x.name.split(" ").slice(-1)[0])).join(", ") || "none")}
      ${kpi("High or very high risk", num(c.n_high_risk), "available players")}
      ${kpi("Injuries in 12 months", num(squad.reduce((s, p) => s + p.injuries_last_365d, 0)), `${num(squad.reduce((s, p) => s + p.days_lost_last_365d, 0))} days lost`)}
    </div>
    <div class="grid g-21">
      ${card("Roster", `<div class="table-wrap"><table>
        <thead><tr><th>Name</th><th>Pos</th><th class="num">Age</th><th class="num">Ht</th><th class="num">Wt</th><th>Nat</th><th>Status</th><th class="num">Inj. 12 mo</th><th class="num">7-day risk</th></tr></thead>
        <tbody>${groups.filter(([, ps]) => ps.length).map(([g, ps]) => `<tr class="group-row"><td colspan="9">${g}</td></tr>${ps.map((p) => `
          <tr><td><div class="pcell">${portrait(p, 32)}<div><a class="nm" href="/players/${p.athlete_id}" data-link>${esc(p.name)}</a> <span class="muted">${esc(p.shirt_number ?? "")}</span></div></div></td>
          <td>${esc(p.position)}</td><td class="num">${num(p.age, 0)}</td><td class="num">${p.height_cm ? `${p.height_cm}` : "–"}</td><td class="num">${p.weight_kg ?? "–"}</td>
          <td>${esc(p.nationality_code || "")}</td><td>${statusChip(p.status)}</td><td class="num">${p.injuries_last_365d}</td>
          <td class="num">${riskChip(p.risk_probability, p.risk_category)}</td></tr>`).join("")}`).join("")}</tbody></table></div>`, "height cm · weight kg")}
      ${card("Club injury report", current.length ? `<ul class="feed">${current.map((x) => `<li style="grid-template-columns:1fr">
        <div><a class="hl" href="/players/${x.athlete_id}/injuries" data-link>${esc(x.name)}</a> ${statusChip(x.status)}
        <div class="meta">${esc(x.diagnosis || "")}${x.side ? ` (${esc(x.side.toLowerCase())})` : ""} · since ${fmtDate(x.onset_date, { day: "numeric", month: "short" })} · est. return ${fmtDate(x.return_date, { day: "numeric", month: "short" })}</div></div></li>`).join("")}</ul>`
        : `<p class="muted">Full squad available.</p>`, fmtDate(inj.as_of))}
    </div>`;
}
