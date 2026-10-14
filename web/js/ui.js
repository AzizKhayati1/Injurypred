// Small DOM / formatting helpers shared by the views.
export const CATEGORIES = ["LOW", "MODERATE", "HIGH", "VERY_HIGH"];
export const CAT_LABEL = { LOW: "Low", MODERATE: "Moderate", HIGH: "High", VERY_HIGH: "Very high", UNAVAILABLE: "Unavailable" };
export const CAT_VAR = { LOW: "--low", MODERATE: "--moderate", HIGH: "--high", VERY_HIGH: "--very-high" };

export function esc(v) {
  return String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const pct = (p, d = 1) => (p == null || Number.isNaN(p) ? "–" : `${(100 * p).toFixed(d)}%`);
export const num = (v, d = 0) => (v == null || Number.isNaN(v) ? "–" : Number(v).toLocaleString("en-GB", { maximumFractionDigits: d, minimumFractionDigits: d }));
export const signed = (v, d = 1, unit = "") => (v == null ? "–" : `${v > 0 ? "+" : ""}${Number(v).toFixed(d)}${unit}`);
export const fmtDate = (s, opts = { day: "numeric", month: "short", year: "numeric" }) =>
  s ? new Date(`${s}T00:00:00`).toLocaleDateString("en-GB", opts) : "–";

export function badge(cat, size = "") {
  const c = cat || "UNAVAILABLE";
  return `<span class="badge ${c} ${size}">${CAT_LABEL[c] || esc(c)}</span>`;
}
export function riskCell(p, max = 0.5) {
  if (p == null) return `<span class="muted">–</span>`;
  const w = Math.min(100, (100 * p) / max);
  return `<div class="risk-cell"><span class="num">${pct(p)}</span><span class="bar"><i style="width:${w}%"></i></span></div>`;
}
export function kpi(label, value, sub = "", help = "") {
  const h = help ? ` title="${esc(help)}"` : "";
  return `<div class="kpi"${h}><div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div></div>`;
}
export const card = (title, body, aside = "", cls = "") =>
  `<section class="card ${cls}"><div class="card-head"><h2>${title}</h2><span class="muted">${aside}</span></div>${body}</section>`;

// ---- tooltip
const tip = () => document.getElementById("tooltip");
export function showTip(html, x, y) {
  const t = tip();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  t.style.left = `${Math.min(x + 14, window.innerWidth - r.width - 8)}px`;
  t.style.top = `${Math.min(y + 14, window.innerHeight - r.height - 8)}px`;
}
export const hideTip = () => { tip().hidden = true; };

// ---- sortable table: rows = array of objects, columns = [{key, label, num, render, sort}]
export function sortableTable(container, columns, rows, { initial, onRowClick, empty = "No rows" } = {}) {
  let sortKey = initial?.key ?? columns[0].key;
  let dir = initial?.dir ?? "desc";
  const draw = () => {
    const col = columns.find((c) => c.key === sortKey);
    const val = (r) => (col.sort ? col.sort(r) : r[sortKey]);
    const sorted = [...rows].sort((a, b) => {
      const x = val(a), y = val(b);
      if (x == null && y == null) return 0;
      if (x == null) return 1;
      if (y == null) return -1;
      return (x < y ? -1 : x > y ? 1 : 0) * (dir === "asc" ? 1 : -1);
    });
    container.innerHTML = `<div class="table-wrap"><table><thead><tr>${columns.map((c) =>
      `<th class="sortable ${c.num ? "num" : ""}" data-key="${c.key}" ${c.key === sortKey ? `aria-sort="${dir === "asc" ? "ascending" : "descending"}"` : ""} tabindex="0">${c.label}</th>`).join("")}</tr></thead>
      <tbody>${sorted.length ? sorted.map((r, i) => `<tr class="${onRowClick ? "clickable" : ""}" data-i="${i}">${columns.map((c) =>
        `<td class="${c.num ? "num" : ""}">${c.render ? c.render(r) : esc(r[c.key])}</td>`).join("")}</tr>`).join("")
        : `<tr><td colspan="${columns.length}" class="empty">${empty}</td></tr>`}</tbody></table></div>`;
    container.querySelectorAll("th.sortable").forEach((th) => {
      const go = () => {
        if (sortKey === th.dataset.key) dir = dir === "asc" ? "desc" : "asc";
        else { sortKey = th.dataset.key; dir = "desc"; }
        draw();
      };
      th.addEventListener("click", go);
      th.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
    });
    if (onRowClick) container.querySelectorAll("tbody tr.clickable").forEach((tr) =>
      tr.addEventListener("click", (e) => { if (!e.target.closest("a")) onRowClick(sorted[+tr.dataset.i]); }));
  };
  draw();
}

// ---- sequential blue ramp for magnitudes (light → dark)
const RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"];
export function rampColor(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const a = RAMP[i].match(/\w\w/g).map((h) => parseInt(h, 16));
  const b = RAMP[i + 1].match(/\w\w/g).map((h) => parseInt(h, 16));
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * f)).join(",")})`;
}
