import { getClubs, getPlayers } from "../data.js";
import { POS_LABEL, crest, portrait, riskChip, statusChip } from "../components.js";
import { navigate } from "../router.js";
import { esc, num, sortableTable } from "../ui.js";
import { playerCard } from "./home.js";

export async function playersView(ctx) {
  const [{ players }, { clubs }] = await Promise.all([getPlayers(), getClubs()]);
  ctx.title("Players");
  ctx.crumbs([["Players"]]);
  const q = ctx.query;
  const st = { q: q.get("q") || "", club: q.get("club") || "", pos: q.get("pos") || "", status: q.get("status") || "",
               sort: q.get("sort") || "risk", view: q.get("view") === "table" ? "table" : "cards" };
  ctx.main.innerHTML = `
    <div class="page-head"><div><h1>Players</h1><p>${players.length} fictional players across ${clubs.length} clubs. Open a player for their profile, injury history, prevention plan and explained risk estimate.</p></div>
      <div class="seg"><button data-view="cards" class="${st.view === "cards" ? "active" : ""}">Cards</button><button data-view="table" class="${st.view === "table" ? "active" : ""}">Table</button></div></div>
    <div class="controls" style="margin-bottom:14px">
      <input type="search" id="f-q" placeholder="Search name, nationality…" value="${esc(st.q)}" aria-label="Search players">
      <select id="f-club" aria-label="Club"><option value="">All clubs</option>${clubs.map((c) => `<option value="${c.team_id}" ${c.team_id === st.club ? "selected" : ""}>${esc(c.name)}</option>`).join("")}</select>
      <select id="f-pos" aria-label="Position"><option value="">All positions</option>${Object.entries(POS_LABEL).map(([k, v]) => `<option value="${k}" ${k === st.pos ? "selected" : ""}>${v}</option>`).join("")}</select>
      <select id="f-status" aria-label="Status"><option value="">Any status</option>${["Available", "Day-to-day", "Out"].map((s) => `<option ${s === st.status ? "selected" : ""}>${s}</option>`).join("")}</select>
      <select id="f-sort" aria-label="Sort"><option value="risk" ${st.sort === "risk" ? "selected" : ""}>Sort: risk</option><option value="name" ${st.sort === "name" ? "selected" : ""}>Sort: name</option><option value="injuries" ${st.sort === "injuries" ? "selected" : ""}>Sort: injuries (12 mo)</option><option value="number" ${st.sort === "number" ? "selected" : ""}>Sort: club & number</option></select>
      <span class="muted small" id="f-count"></span>
    </div>
    <div id="list"></div>`;

  const draw = () => {
    const s = st.q.toLowerCase();
    let rows = players.filter((p) => (!s || `${p.name} ${p.nationality} ${p.athlete_id}`.toLowerCase().includes(s))
      && (!st.club || p.team_id === st.club) && (!st.pos || p.position === st.pos) && (!st.status || p.status === st.status));
    const key = { risk: (p) => -(p.risk_probability ?? -1), name: (p) => p.last_name || p.name, injuries: (p) => -p.injuries_last_365d,
                  number: (p) => `${p.team_id}${String(p.shirt_number).padStart(3, "0")}` }[st.sort];
    rows = rows.sort((a, b) => (key(a) < key(b) ? -1 : key(a) > key(b) ? 1 : 0));
    document.getElementById("f-count").textContent = `${rows.length} shown`;
    const el = document.getElementById("list");
    if (st.view === "cards") {
      el.innerHTML = rows.length ? `<div class="cards">${rows.map(playerCard).join("")}</div>` : `<p class="empty">No players match these filters.</p>`;
    } else {
      sortableTable(el, [
        { key: "name", label: "Player", sort: (r) => r.last_name, render: (r) => `<div class="pcell">${portrait(r, 34)}<div><a class="nm" href="/players/${r.athlete_id}" data-link>${esc(r.name)}</a><div class="meta">#${esc(r.shirt_number ?? "")} · ${esc(r.nationality || "")}</div></div></div>` },
        { key: "team_id", label: "Club", render: (r) => `<span class="pcell">${crest(r.club, 18)} ${esc(r.club.short_name)}</span>` },
        { key: "position", label: "Pos" },
        { key: "age", label: "Age", num: true, render: (r) => num(r.age, 0) },
        { key: "status", label: "Status", render: (r) => statusChip(r.status) },
        { key: "injuries_last_365d", label: "Inj. 12 mo", num: true },
        { key: "risk_probability", label: "7-day risk", num: true, render: (r) => riskChip(r.risk_probability, r.risk_category),
          sort: (r) => r.risk_probability ?? -1 },
      ], rows, { initial: { key: "risk_probability", dir: "desc" }, onRowClick: (r) => navigate(`/players/${r.athlete_id}`) });
    }
    const qs = new URLSearchParams(Object.entries(st).filter(([k, v]) => v && !(k === "sort" && v === "risk") && !(k === "view" && v === "cards")));
    history.replaceState({}, "", `/players${qs.toString() ? `?${qs}` : ""}`);
  };
  document.getElementById("f-q").addEventListener("input", (e) => { st.q = e.target.value.trim(); draw(); });
  for (const [id, k] of [["f-club", "club"], ["f-pos", "pos"], ["f-status", "status"], ["f-sort", "sort"]]) {
    document.getElementById(id).addEventListener("change", (e) => { st[k] = e.target.value; draw(); });
  }
  ctx.main.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => {
    st.view = b.dataset.view;
    ctx.main.querySelectorAll("[data-view]").forEach((x) => x.classList.toggle("active", x === b));
    draw();
  }));
  draw();
}
