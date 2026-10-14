// History-API router: real URLs (/athletes/A172/workload), back/forward, deep links.
// Netlify serves index.html for unknown paths (see netlify.toml / _redirects), so every URL is loadable directly.
import { destroyAll } from "./charts.js";
import { hideTip } from "./ui.js";

const routes = [];
let current = 0;

export function route(pattern, view, nav) {
  const keys = [];
  const re = new RegExp(`^${pattern.replace(/\/:([a-zA-Z]+)/g, (_, k) => { keys.push(k); return "/([^/]+)"; })}/?$`);
  routes.push({ re, keys, view, nav });
}

export function navigate(to, { replace = false } = {}) {
  const url = new URL(to, location.origin);
  if (url.pathname + url.search === location.pathname + location.search && !replace) return;
  history[replace ? "replaceState" : "pushState"]({}, "", url.pathname + url.search + url.hash);
  render(true);
}

export function setQuery(params) {
  const q = new URLSearchParams(location.search);
  Object.entries(params).forEach(([k, v]) => (v == null || v === "" ? q.delete(k) : q.set(k, v)));
  const s = q.toString();
  navigate(location.pathname + (s ? `?${s}` : ""), { replace: true });
}

export async function render(scrollTop = false) {
  const token = ++current;
  const main = document.getElementById("main");
  const path = decodeURIComponent(location.pathname);
  let match = null;
  for (const r of routes) {
    const m = path.match(r.re);
    if (m) { match = { r, params: Object.fromEntries(r.keys.map((k, i) => [k, m[i + 1]])) }; break; }
  }
  destroyAll();
  hideTip();
  document.querySelectorAll("[data-nav]").forEach((a) => {
    if (match && a.dataset.nav === match.r.nav) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  closeMenu();
  main.setAttribute("aria-busy", "true");
  const ctx = {
    main,
    params: match ? match.params : {},
    query: new URLSearchParams(location.search),
    stale: () => token !== current,
    title: (t) => { document.title = t ? `${t} · PlayerLoad-Risk` : "PlayerLoad-Risk"; },
    crumbs: (items) => {
      document.getElementById("crumbs").innerHTML = items.map(([label, href], i) =>
        (href && i < items.length - 1 ? `<a href="${href}" data-link>${label}</a>` : `<span>${label}</span>`)
      ).join('<span class="sep">/</span>');
    },
  };
  try {
    await (match ? match.r.view(ctx) : routes.notFound(ctx));
  } catch (err) {
    if (token !== current) return;
    if (err?.name !== "NotFoundError" && err?.constructor?.name !== "NotFoundError") console.error(err);
    await routes.notFound(ctx, err);
  }
  if (token !== current) return;
  main.removeAttribute("aria-busy");
  if (scrollTop) { window.scrollTo(0, 0); main.focus({ preventScroll: true }); }
}

export function setNotFound(view) { routes.notFound = view; }

function closeMenu() {
  document.getElementById("sitenav").classList.remove("open");
  document.getElementById("menu-btn").setAttribute("aria-expanded", "false");
}

export function startRouter() {
  document.addEventListener("click", (e) => {
    const a = e.target.closest("a[href]");
    if (!a || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    if (a.target === "_blank" || a.hasAttribute("download")) return;
    const url = new URL(a.href, location.href);
    if (url.origin !== location.origin || url.pathname.startsWith("/data/")) return;
    e.preventDefault();
    navigate(url.pathname + url.search + url.hash);
  });
  window.addEventListener("popstate", () => render(false));
  render(false);
}
