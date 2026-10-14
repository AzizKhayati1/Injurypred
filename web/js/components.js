// Reusable visual components: player portraits, club crests, status chips, body map.
import { CAT_LABEL, esc, pct } from "./ui.js";

function hash(str) {
  let h = 2166136261;
  for (const c of String(str)) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
const SKIN = ["#f1c7a5", "#e0ac84", "#c68b62", "#a86a45", "#7d4a2c", "#5b3520"];
const HAIR = ["#1d1611", "#2f2017", "#4a3222", "#6b4a2b", "#a5733c", "#c9b27a", "#0f0f0f"];

/** Player portrait: real photo when provided, otherwise a generated kit-coloured illustration. */
export function portrait(p, size = 64, cls = "") {
  const club = p.club || {};
  const alt = `Portrait of ${esc(p.name || p.athlete_id)}`;
  if (p.photo) {
    return `<img class="portrait ${cls}" src="${esc(p.photo)}" alt="${alt}" width="${size}" height="${size}" loading="lazy">`;
  }
  const h = hash(p.athlete_id);
  const skin = SKIN[h % SKIN.length];
  const hair = HAIR[(h >>> 4) % HAIR.length];
  const style = (h >>> 8) % 4;
  const kit = club.primary_color || "#33415c";
  const trim = club.secondary_color || "#ffffff";
  const id = `pg${h % 100000}`;
  const hairPath = [
    "M30 33c0-11 8-17 20-17s20 6 20 17c-3-5-9-8-20-8s-17 3-20 8z",
    "M29 36c-1-13 8-21 21-21s22 8 21 21c-4-4-8-10-21-10s-17 6-21 10z",
    "M31 31c2-9 9-14 19-14s17 5 19 14c-6-3-12-5-19-5s-13 2-19 5z",
    "M29 38c-2-16 7-23 21-23s23 7 21 23l-3-9c-5-3-11-4-18-4s-13 1-18 4z",
  ][style];
  return `<svg class="portrait ${cls}" viewBox="0 0 100 100" width="${size}" height="${size}" role="img" aria-label="${alt}">
    <defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#eef0f3"/><stop offset="1" stop-color="#cfd4db"/></linearGradient>
    <clipPath id="${id}c"><circle cx="50" cy="50" r="50"/></clipPath></defs>
    <g clip-path="url(#${id}c)">
      <rect width="100" height="100" fill="url(#${id})"/>
      <path d="M8 104c2-20 16-30 42-30s40 10 42 30z" fill="${kit}"/>
      <path d="M38 75l12 10 12-10" fill="none" stroke="${trim}" stroke-width="3"/>
      <text x="50" y="99" text-anchor="middle" font-size="12" font-weight="700" fill="${trim}" font-family="Barlow Condensed, sans-serif">${esc(p.shirt_number ?? "")}</text>
      <rect x="43" y="56" width="14" height="16" rx="6" fill="${skin}"/>
      <ellipse cx="50" cy="42" rx="17" ry="20" fill="${skin}"/>
      <path d="${hairPath}" fill="${hair}"/>
    </g></svg>`;
}

/** Club crest: shield in club colours with short name. */
export function crest(club, size = 28) {
  const c = club || {};
  const p = c.primary_color || "#33415c", s = c.secondary_color || "#fff";
  return `<svg class="crest" viewBox="0 0 40 46" width="${size}" height="${Math.round(size * 1.15)}" role="img" aria-label="${esc(c.name || "")} crest">
    <path d="M20 1l18 6v14c0 12-8 20-18 24C10 41 2 33 2 21V7z" fill="${p}" stroke="${s}" stroke-width="2"/>
    <path d="M2 21h36" stroke="${s}" stroke-width="1.5" opacity=".55"/>
    <text x="20" y="18" text-anchor="middle" font-size="9.5" font-weight="800" fill="${s}" font-family="Barlow Condensed, sans-serif">${esc((c.short_name || "").slice(0, 3))}</text>
    <circle cx="20" cy="30" r="5" fill="none" stroke="${s}" stroke-width="1.5"/></svg>`;
}

export function statusChip(status) {
  const s = status || "Available";
  const cls = { Available: "ok", "Day-to-day": "dtd", Out: "out" }[s] || "ok";
  return `<span class="status ${cls}">${esc(s)}</span>`;
}

export function riskChip(p, cat) {
  if (p == null || !cat) return `<span class="muted small">—</span>`;
  return `<span class="riskchip ${cat}" title="${CAT_LABEL[cat]} estimated 7-day risk"><b>${pct(p, 0)}</b> ${CAT_LABEL[cat]}</span>`;
}

export function severityChip(sev) {
  return sev ? `<span class="sev sev-${esc(sev.toLowerCase())}">${esc(sev)}</span>` : "";
}

export const playerName = (p) => esc(p.name || p.athlete_id);
export const POS_LABEL = { GK: "Goalkeeper", CB: "Centre-back", FB: "Full-back", CM: "Midfielder", W: "Winger", ST: "Forward" };
export const POS_GROUP = { GK: "Goalkeepers", CB: "Defenders", FB: "Defenders", CM: "Midfielders", W: "Forwards", ST: "Forwards" };

// ---------------------------------------------------------------- body map
// Zones (athlete's left/right). Front view: athlete's left is on the viewer's right.
const FRONT = {
  head: [["C", "circle", 60, 22, 13]],
  chest: [["C", "rect", 44, 40, 32, 30, 8]],
  groin: [["L", "rect", 61, 84, 11, 12, 4], ["R", "rect", 48, 84, 11, 12, 4]],
  thigh_front: [["L", "rect", 61, 97, 13, 34, 6], ["R", "rect", 46, 97, 13, 34, 6]],
  knee: [["L", "circle", 67, 137, 6], ["R", "circle", 53, 137, 6]],
  ankle: [["L", "circle", 67, 182, 4.5], ["R", "circle", 53, 182, 4.5]],
  foot: [["L", "rect", 63, 187, 12, 7, 3], ["R", "rect", 45, 187, 12, 7, 3]],
};
const BACK = {
  thigh_back: [["L", "rect", 46, 97, 13, 34, 6], ["R", "rect", 61, 97, 13, 34, 6]],
  calf: [["L", "rect", 47, 145, 11, 24, 5], ["R", "rect", 62, 145, 11, 24, 5]],
  achilles: [["L", "rect", 50, 170, 5, 10, 2], ["R", "rect", 65, 170, 5, 10, 2]],
};
const SILHOUETTE = `<g class="bm-body">
  <circle cx="60" cy="22" r="14"/><rect x="54" y="34" width="12" height="6" rx="2"/>
  <path d="M38 42c0-4 4-6 8-6h28c4 0 8 2 8 6l6 40-8 2-6-30v30H44V54l-6 30-8-2z"/>
  <path d="M44 84h32l-1 52-3 50h-9l-1-48-2-26-2 26-1 48h-9l-3-50z"/>
  <rect x="44" y="186" width="13" height="8" rx="3"/><rect x="63" y="186" width="13" height="8" rx="3"/></g>`;

function zoneShapes(zones, counts, records, view) {
  let out = "";
  for (const [zone, parts] of Object.entries(zones)) {
    for (const [side, shape, ...g] of parts) {
      const n = counts[`${zone}:${side}`] || 0;
      const fill = n ? `var(--very-high)` : "transparent";
      const op = n ? Math.min(0.95, 0.35 + 0.2 * n) : 0;
      const recs = records.filter((r) => r.body_zone === zone && ((r.side || "C")[0] === side)).map((r) => `${r.onset_date}: ${r.diagnosis}`);
      const title = n ? `${zone.replace("_", " ")} (${side === "C" ? "central" : side === "L" ? "left" : "right"}) – ${n} injur${n > 1 ? "ies" : "y"}\n${recs.join("\n")}` : "";
      const attrs = `class="bm-zone ${n ? "hit" : ""}" fill="${fill}" fill-opacity="${op}" data-tip="${esc(title)}" data-view="${view}"`;
      out += shape === "circle"
        ? `<circle cx="${g[0]}" cy="${g[1]}" r="${g[2]}" ${attrs}/>`
        : `<rect x="${g[0]}" y="${g[1]}" width="${g[2]}" height="${g[3]}" rx="${g[4]}" ${attrs}/>`;
    }
  }
  return out;
}

export function bodyMap(counts, records) {
  return `<div class="bodymap">
    <figure><svg viewBox="0 0 120 200" role="img" aria-label="Front view">${SILHOUETTE}${zoneShapes(FRONT, counts, records, "front")}
      <text x="30" y="198" class="bm-lr">R</text><text x="86" y="198" class="bm-lr">L</text></svg><figcaption>Front</figcaption></figure>
    <figure><svg viewBox="0 0 120 200" role="img" aria-label="Back view">${SILHOUETTE}${zoneShapes(BACK, counts, records, "back")}
      <text x="30" y="198" class="bm-lr">L</text><text x="86" y="198" class="bm-lr">R</text></svg><figcaption>Back</figcaption></figure>
  </div>`;
}
