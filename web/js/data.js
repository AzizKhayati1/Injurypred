// Data access: pre-computed JSON in /data (always), optional live API (window.PLR_CONFIG.apiUrl).
const cache = new Map();

export class NotFoundError extends Error {
  name = "NotFoundError";
}

export async function getJSON(path) {
  if (!cache.has(path)) {
    cache.set(path, (async () => {
      const res = await fetch(`/data/${path}`, { headers: { Accept: "application/json" } });
      const type = res.headers.get("content-type") || "";
      // On Netlify an unknown path falls through to the SPA rewrite (index.html, 200), so check the type too.
      if (!res.ok || !type.includes("json")) throw new NotFoundError(path);
      return res.json();
    })().catch((e) => { cache.delete(path); throw e; }));
  }
  return cache.get(path);
}

export const getMeta = () => getJSON("meta.json");
export const getSquad = () => getJSON("squad.json");
export const getHeatmap = () => getJSON("heatmap.json");

export async function getAthlete(id) {
  const squad = await getSquad();
  if (!squad.athletes.some((a) => a.athlete_id === id)) throw new NotFoundError(id);
  return getJSON(`athletes/${encodeURIComponent(id)}.json`);
}

export const apiUrl = ((window.PLR_CONFIG || {}).apiUrl || "").replace(/\/+$/, "");

export async function liveSimulate(athleteId, scenario) {
  const res = await fetch(`${apiUrl}/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ athlete_id: athleteId, scenarios: [scenario] }),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `API error ${res.status}`);
  return body;
}
