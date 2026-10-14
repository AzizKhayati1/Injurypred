import { getMeta, getSquad } from "../data.js";
import { navigate } from "../router.js";
import { CATEGORIES, badge, esc, riskCell, sortableTable } from "../ui.js";

const POSITIONS = { GK: "Goalkeeper", CB: "Centre-back", FB: "Full-back", CM: "Central midfield", W: "Winger", ST: "Striker" };

export async function athletesView(ctx) {
  const [meta, squad] = await Promise.all([getMeta(), getSquad()]);
  ctx.title("Athletes");
  ctx.crumbs([["Athletes"]]);
  const q = ctx.query;
  const state = { search: q.get("q") || "", team: q.get("team") || "", pos: q.get("pos") || "", status: q.get("status") || "" };
  ctx.main.innerHTML = `
    <div class="page-head">
      <div><h1>Athletes</h1><p>${squad.athletes.length} synthetic athletes. Select one to see their explained risk estimate, workload and what-if scenarios.</p></div>
    </div>
    <section class="card">
      <div class="controls" style="margin-bottom:12px">
        <input type="search" id="f-q" placeholder="Search athlete ID…" value="${esc(state.search)}" aria-label="Search athletes">
        <select id="f-team" aria-label="Team"><option value="">All teams</option>${meta.teams.map((t) => `<option ${t === state.team ? "selected" : ""}>${t}</option>`).join("")}</select>
        <select id="f-pos" aria-label="Position"><option value="">All positions</option>${Object.entries(POSITIONS).map(([k, v]) => `<option value="${k}" ${k === state.pos ? "selected" : ""}>${v}</option>`).join("")}</select>
        <select id="f-status" aria-label="Availability"><option value="">Any availability</option><option value="available" ${state.status === "available" ? "selected" : ""}>Available</option><option value="unavailable" ${state.status === "unavailable" ? "selected" : ""}>Unavailable</option></select>
        <span class="muted small" id="f-count"></span>
      </div>
      <div id="t-ath"></div>
    </section>`;

  const draw = () => {
    const rows = squad.athletes.filter((a) =>
      (!state.search || a.athlete_id.toLowerCase().includes(state.search.toLowerCase())) &&
      (!state.team || a.team_id === state.team) && (!state.pos || a.position === state.pos) &&
      (!state.status || (state.status === "available") === a.available));
    document.getElementById("f-count").textContent = `${rows.length} shown`;
    sortableTable(document.getElementById("t-ath"), [
      { key: "athlete_id", label: "Athlete", render: (r) => `<a href="/athletes/${r.athlete_id}" data-link>${esc(r.athlete_id)}</a>` },
      { key: "team_id", label: "Team" },
      { key: "position", label: "Position", render: (r) => POSITIONS[r.position] || esc(r.position) },
      { key: "available", label: "Status", render: (r) => (r.available ? "Available" : `<span class="muted">Unavailable</span>`), sort: (r) => (r.available ? 1 : 0) },
      { key: "latest_risk_probability", label: "7-day risk", num: true, render: (r) => riskCell(r.latest_risk_probability) },
      { key: "latest_risk_category", label: "Category", render: (r) => badge(r.available ? r.latest_risk_category : "UNAVAILABLE"),
        sort: (r) => (r.available ? CATEGORIES.indexOf(r.latest_risk_category) : -1) },
    ], rows, { initial: { key: "latest_risk_probability", dir: "desc" }, onRowClick: (r) => navigate(`/athletes/${r.athlete_id}`), empty: "No athletes match these filters" });
  };
  const sync = () => setQueryQuiet({ q: state.search, team: state.team, pos: state.pos, status: state.status });
  document.getElementById("f-q").addEventListener("input", (e) => { state.search = e.target.value.trim(); draw(); sync(); });
  for (const [id, key] of [["f-team", "team"], ["f-pos", "pos"], ["f-status", "status"]]) {
    document.getElementById(id).addEventListener("change", (e) => { state[key] = e.target.value; draw(); sync(); });
  }
  draw();
}

// keep filters in the URL (shareable, survives back/forward) without re-rendering the page
function setQueryQuiet(params) {
  const q = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => v && q.set(k, v));
  const s = q.toString();
  history.replaceState({}, "", `${location.pathname}${s ? `?${s}` : ""}`);
}
