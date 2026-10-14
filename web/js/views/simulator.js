import { apiUrl, getAthlete, getMeta, getSquad, liveSimulate } from "../data.js";
import { chart, series } from "../charts.js";
import { navigate } from "../router.js";
import { CATEGORIES, CAT_VAR, badge, card, esc, fmtDate, kpi, num, pct, riskCell, signed } from "../ui.js";
import { athleteHeader } from "./athlete.js";

const DIMS = { total_load_pct: "Total load", sprint_load_pct: "Sprint load", hsr_pct: "High-speed running", duration_pct: "Session duration" };
const DISCLAIMER = "Model simulation only: how the model's statistical estimate responds to a hypothetical change in tomorrow's session. Not causal evidence, not medical advice.";

export async function simulatorIndexView(ctx) {
  const squad = await getSquad();
  ctx.title("What-if simulator");
  ctx.crumbs([["What-if simulator"]]);
  const avail = squad.athletes.filter((a) => a.available).sort((a, b) => b.latest_risk_probability - a.latest_risk_probability);
  ctx.main.innerHTML = `
    <div class="page-head"><div><h1>What-if simulator</h1>
      <p>Pick an available athlete to see how the estimated 7-day risk responds if tomorrow's session is heavier, lighter or replaced by recovery.</p></div>
      <div class="controls"><select id="pick" aria-label="Athlete"><option value="">Choose an athlete…</option>
        ${avail.map((a) => `<option value="${a.athlete_id}">${a.athlete_id} · ${a.position} · ${a.team_id} · ${pct(a.latest_risk_probability)}</option>`).join("")}</select></div>
    </div>
    <p class="callout">${DISCLAIMER}</p>
    <div class="mt">${card("Highest current estimates", `<div class="table-wrap"><table><thead><tr><th>Athlete</th><th>Pos</th><th>Team</th><th class="num">7-day risk</th><th>Category</th><th></th></tr></thead>
      <tbody>${avail.slice(0, 12).map((a) => `<tr><td><a href="/simulator/${a.athlete_id}" data-link>${a.athlete_id}</a></td><td>${a.position}</td><td>${a.team_id}</td>
        <td class="num">${riskCell(a.latest_risk_probability)}</td><td>${badge(a.latest_risk_category)}</td>
        <td><a class="btn" href="/simulator/${a.athlete_id}" data-link>Simulate →</a></td></tr>`).join("")}</tbody></table></div>`, "available athletes")}</div>`;
  document.getElementById("pick").addEventListener("change", (e) => e.target.value && navigate(`/simulator/${e.target.value}`));
}

function bars(rows, max) {
  return rows.map((r) => `<div class="sim-row ${r.base ? "base" : ""}"><span>${esc(r.name)}</span>
    <div class="track"><i style="width:${Math.max(1, 100 * r.p / max)}%;background:var(${CAT_VAR[r.cat]})"></i></div>
    <span class="num" style="text-align:right">${pct(r.p)}</span></div>`).join("");
}

export async function simulatorView(ctx) {
  const id = ctx.params.id;
  const [d, meta] = await Promise.all([getAthlete(id), getMeta()]);
  if (ctx.stale()) return;
  const label = Object.fromEntries(meta.features.map((f) => [f.name, f.label]));
  ctx.title(`${id} · What-if`);
  ctx.crumbs([["What-if simulator", "/simulator"], [esc(id)]]);
  const sim = d.simulation;
  if (!sim) {
    ctx.main.innerHTML = `${athleteHeader(id, d, "whatif")}<p class="callout">${esc(d.prediction?.unavailable_reason || "No simulation available: the athlete is not available on the latest date.")}</p>
      <p class="mt"><a class="btn" href="/simulator" data-link>← Choose another athlete</a></p>`;
    return;
  }
  const base = sim.baseline;
  const dim = DIMS[ctx.query.get("dim")] ? ctx.query.get("dim") : "sprint_load_pct";
  const rows = [{ name: "Planned session (no change)", p: base.risk_probability, cat: base.risk_category, base: true },
    ...sim.presets.map((s) => ({ name: s.name, p: s.risk_probability, cat: s.risk_category }))];
  const max = Math.max(...rows.map((r) => r.p)) * 1.15;
  const ps = sim.planned_session;

  ctx.main.innerHTML = `${athleteHeader(id, d, "whatif")}
    <p class="callout">${DISCLAIMER}</p>
    <div class="kpis mt">
      ${kpi("Risk today", pct(sim.current_risk.risk_probability), `data up to ${fmtDate(sim.as_of_date)}`)}
      ${kpi(`With the planned session on ${fmtDate(sim.simulated_day, { day: "numeric", month: "short" })}`, pct(base.risk_probability), badge(base.risk_category))}
      ${kpi("Planned session", `${num(ps.player_load)} AU`, `${num(ps.duration_minutes)} min · ${num(ps.sprint_distance)} m sprint`, ps.source)}
    </div>
    <div class="grid g-2">
      ${card("Scenario comparison", bars(rows, max), "simulated 7-day risk")}
      ${card("Explore one dimension", `
        <div class="seg" id="dims">${Object.entries(DIMS).map(([k, v]) => `<a href="/simulator/${id}?dim=${k}" data-link class="${k === dim ? "active" : ""}">${v}</a>`).join("")}</div>
        <div style="display:flex;align-items:baseline;gap:14px;margin:14px 0 4px">
          <span class="slider-value num" id="sv-risk"></span><span id="sv-delta" class="num"></span><span id="sv-cat"></span>
        </div>
        <label class="muted small" for="sv">Change to tomorrow's ${DIMS[dim].toLowerCase()}: <strong id="sv-lab"></strong></label>
        <input type="range" id="sv" min="0" step="1">
        <div class="chart-box short"><canvas id="c-sweep"></canvas></div>`, "pre-computed by the model")}
    </div>
    ${apiUrl ? `<div class="mt">${card("Custom combination (live API)", `
      <div class="grid g-2">${Object.entries(DIMS).map(([k, v]) => `<label class="small">${v}: <strong id="lv-${k}">0%</strong>
        <input type="range" data-k="${k}" min="-50" max="100" step="5" value="0"></label>`).join("")}</div>
      <label class="small"><input type="checkbox" id="lv-rest"> Replace with a recovery day</label>
      <div style="margin-top:12px;display:flex;gap:12px;align-items:center"><button class="btn btn-primary" id="lv-run">Simulate</button><span id="lv-out" class="num"></span></div>`,
      esc(apiUrl))}</div>` : ""}
    <div class="mt">${card("Feature context", `<dl class="kv">${Object.entries(base.key_features).map(([k, v]) =>
      `<dt>${esc(label[k] || k)}</dt><dd>${v == null ? "–" : num(v, 2)}</dd>`).join("")}</dl>`, "values with the planned session")}</div>`;

  const sweep = sim.sweeps[dim];
  const keys = Object.keys(sweep).map(Number).sort((a, b) => a - b);
  const slider = document.getElementById("sv");
  slider.max = keys.length - 1;
  slider.value = keys.indexOf(0);
  const [s1] = series();
  const c = chart(document.getElementById("c-sweep"), {
    labels: keys.map((k) => signed(k, 0, "%")), legend: false, yFormat: (v) => pct(v, 0), tooltipFormat: (v) => pct(v),
    datasets: [{ label: "Simulated risk", data: keys.map((k) => sweep[k]), borderColor: s1, backgroundColor: s1, stepped: false,
      pointRadius: keys.map(() => 3), pointHoverRadius: 6 }] });
  const update = () => {
    const k = keys[+slider.value], p = sweep[k];
    const delta = p - base.risk_probability;
    document.getElementById("sv-lab").textContent = signed(k, 0, "%");
    document.getElementById("sv-risk").textContent = pct(p);
    const el = document.getElementById("sv-delta");
    el.textContent = `${delta >= 0 ? "+" : ""}${(100 * delta).toFixed(1)} pp`;
    el.className = `num ${delta > 0.0005 ? "delta-up" : delta < -0.0005 ? "delta-down" : "muted"}`;
    const cat = categoryOf(p, d.prediction?.thresholds);
    document.getElementById("sv-cat").innerHTML = cat ? badge(cat) : "";
    if (c) { c.data.datasets[0].pointRadius = keys.map((_, i) => (i === +slider.value ? 7 : 3)); c.update("none"); }
  };
  slider.addEventListener("input", update);
  update();

  if (apiUrl) {
    const vals = {};
    document.querySelectorAll("input[data-k]").forEach((r) => r.addEventListener("input", () => {
      vals[r.dataset.k] = +r.value;
      document.getElementById(`lv-${r.dataset.k}`).textContent = signed(+r.value, 0, "%");
    }));
    document.getElementById("lv-run").addEventListener("click", async () => {
      const out = document.getElementById("lv-out");
      out.textContent = "Simulating…";
      try {
        const res = await liveSimulate(id, { name: "Custom", ...vals, rest_day: document.getElementById("lv-rest").checked });
        const s = res.scenarios[0];
        out.innerHTML = `${pct(s.risk_probability)} (${signed(100 * s.delta_probability, 1)} pp vs planned) ${badge(s.risk_category)}`;
      } catch (e) { out.textContent = `Live API unavailable: ${e.message}`; }
    });
  }
}

function categoryOf(p, th) {
  if (!th) return null;
  let cat = "LOW";
  for (const c of CATEGORIES.slice(1)) if (p >= th[c]) cat = c;
  return cat;
}
