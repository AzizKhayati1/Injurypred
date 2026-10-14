import { getClubs, getInjuries, getMeta, getPlayers } from "../data.js";
import { crest, portrait, riskChip, statusChip } from "../components.js";
import { card, esc, fmtDate, num, pct } from "../ui.js";

export function headline(n) {
  const name = esc(n.name);
  const dx = esc((n.diagnosis || "an injury").replace(/\s*\(.*\)$/, "").toLowerCase());
  if (n.event === "return") return `${name} back available after ${dx}`;
  if (n.event === "illness") return `${name} misses out through illness`;
  return `${name} sidelined with ${dx}`;
}

export function newsItem(n, byId) {
  const p = byId[n.athlete_id] || { athlete_id: n.athlete_id };
  const extra = n.event === "return" ? `${n.days_lost} day${n.days_lost === 1 ? "" : "s"} missed`
    : `${esc(n.severity || "")}${n.side ? ` · ${esc(n.side.toLowerCase())} side` : ""} · est. return ${fmtDate(n.return_date, { day: "numeric", month: "short" })}`;
  return `<li>${portrait(p, 44)}<div>
    <a class="hl" href="/players/${n.athlete_id}${n.event === "return" ? "" : "/injuries"}" data-link><span class="tag ${n.event}">${n.event}</span>${headline(n)}</a>
    <div class="meta">${fmtDate(n.date)} · ${esc(n.club)} · ${extra}</div></div></li>`;
}

export async function homeView(ctx) {
  const [meta, { players }, { clubs }, inj] = await Promise.all([getMeta(), getPlayers(), getClubs(), getInjuries()]);
  ctx.title("");
  ctx.crumbs([["Home"]]);
  const byId = Object.fromEntries(players.map((p) => [p.athlete_id, p]));
  const watch = players.filter((p) => p.risk_probability != null).sort((a, b) => b.risk_probability - a.risk_probability).slice(0, 6);
  const out = inj.current.filter((x) => x.status === "Out").length;
  ctx.main.innerHTML = `
    <section class="clubstrip">${clubs.map((c) => `
      <a class="clubtile" href="/clubs/${c.team_id}" data-link style="--c1:${c.primary_color}">${crest(c, 40)}
        <div><div class="nm">${esc(c.name)}</div>
        <div class="meta">${c.n_players - c.n_unavailable}/${c.n_players} available · ${c.n_unavailable} out · avg risk ${pct(c.mean_risk)}</div></div></a>`).join("")}
    </section>
    <div class="grid g-21">
      <div class="grid" style="align-content:start">
        <section class="lead">
          <div class="kicker">Injury report · ${fmtDate(meta.latest_date)}</div>
          <h1>${inj.n_current} players unavailable across the league</h1>
          <p>${out} out, ${inj.n_current - out} day-to-day. ${watch.length ? `${esc(watch[0].name)} (${esc(watch[0].club.name)}) carries the highest estimated 7-day availability risk at ${pct(watch[0].risk_probability)}.` : ""}</p>
          <p style="margin-top:12px"><a class="btn" href="/injuries" data-link>Full injury report →</a> <a class="btn" href="/squad" data-link>Risk center →</a></p>
        </section>
        ${card("Players to watch", `<div class="cards">${watch.map((p) => playerCard(p)).join("")}</div>
          <p class="muted small" style="margin-top:10px">Highest estimated 7-day availability risk among available players — a prompt for staff attention, not a prediction that they will be injured.</p>`, "highest estimated risk")}
      </div>
      <div class="grid" style="align-content:start">
        ${card("Latest", `<ul class="feed">${inj.news.slice(0, 12).map((n) => newsItem(n, byId)).join("")}</ul>`, "last 3 weeks")}
        ${card("Most common injuries", `<div class="table-wrap"><table><thead><tr><th>Injury</th><th class="num">Cases</th><th class="num">Median days</th></tr></thead>
          <tbody>${inj.by_diagnosis.slice(0, 6).map((r) => `<tr><td>${esc(r.label)}</td><td class="num">${r.count}</td><td class="num">${num(r.median_days_lost)}</td></tr>`).join("")}</tbody></table></div>`, "all seasons")}
      </div>
    </div>`;
}

export function playerCard(p) {
  const c = p.club || {};
  const injLine = p.status !== "Available"
    ? `${statusChip(p.status)} ${esc(p.detail || "")}`
    : `${num(p.injuries_last_365d)} injur${p.injuries_last_365d === 1 ? "y" : "ies"} in 12 months`;
  return `<a class="pcard" href="/players/${p.athlete_id}" data-link style="--c1:${c.primary_color};--c2:${c.secondary_color}">
    <div class="top"><span class="num">${esc(p.shirt_number ?? "")}</span>${crest(c, 26)}${portrait(p, 104)}</div>
    <div class="body">
      <div class="nm"><small>${esc(p.first_name || "")}</small>${esc(p.last_name || p.name)}</div>
      <div class="row"><span>${esc(p.position)} · ${esc(c.short_name || p.team_id)} · ${esc(p.nationality_code || "")}</span>${riskChip(p.risk_probability, p.risk_category)}</div>
      <div class="inj">${injLine}</div>
    </div></a>`;
}
