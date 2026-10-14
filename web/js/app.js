// Entry point: routes, theme, mobile navigation.
import { getMeta } from "./data.js";
import { navigate, render, route, setNotFound, startRouter } from "./router.js";
import { fmtDate } from "./ui.js";
import { aboutView } from "./views/about.js";
import { clubView, clubsView } from "./views/clubs.js";
import { heatmapView } from "./views/heatmap.js";
import { homeView } from "./views/home.js";
import { injuriesView } from "./views/injuries.js";
import { modelView } from "./views/model.js";
import { notFoundView } from "./views/notfound.js";
import { playerInjuriesView, playerOverviewView, playerPreventionView } from "./views/player.js";
import { playersView } from "./views/players.js";
import { simulatorIndexView, simulatorView } from "./views/simulator.js";
import { squadView } from "./views/squad.js";
import { workloadView } from "./views/workload.js";

const redirect = (to) => async (ctx) => navigate(typeof to === "function" ? to(ctx) : to, { replace: true });

route("/", homeView, "home");
route("/players", playersView, "players");
route("/players/:id", playerOverviewView, "players");
route("/players/:id/injuries", playerInjuriesView, "players");
route("/players/:id/prevention", playerPreventionView, "players");
route("/players/:id/workload", workloadView, "players");
route("/players/:id/simulator", simulatorView, "players");
route("/clubs", clubsView, "clubs");
route("/clubs/:id", clubView, "clubs");
route("/injuries", injuriesView, "injuries");
route("/squad", squadView, "squad");
route("/heatmap", heatmapView, "heatmap");
route("/simulator", simulatorIndexView, "simulator");
route("/simulator/:id", redirect((c) => `/players/${c.params.id}/simulator${location.search}`), "simulator");
route("/model", modelView, "model");
route("/about", aboutView, "about");
// backwards-compatible URLs from the first version of the site
route("/athletes", redirect("/players"), "players");
route("/athletes/:id", redirect((c) => `/players/${c.params.id}`), "players");
route("/athletes/:id/workload", redirect((c) => `/players/${c.params.id}/workload${location.search}`), "players");
setNotFound(notFoundView);

// ---- theme
document.getElementById("theme-btn").addEventListener("click", () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === "dark"
    : matchMedia("(prefers-color-scheme: dark)").matches;
  const next = dark ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("plr-theme", next); } catch (e) { /* storage unavailable */ }
  render(false); // charts read colours from CSS variables
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => render(false));

// ---- mobile navigation
const nav = document.getElementById("sitenav");
const menuBtn = document.getElementById("menu-btn");
menuBtn.addEventListener("click", () => {
  const open = !nav.classList.contains("open");
  nav.classList.toggle("open", open);
  menuBtn.setAttribute("aria-expanded", String(open));
});
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && nav.classList.contains("open")) menuBtn.click(); });

getMeta().then((m) => {
  document.getElementById("asof").textContent = `Data to ${fmtDate(m.latest_date)}`;
  document.getElementById("build-info").textContent =
    `Model ${m.model.model_version} (${m.model.model_type}) · exported ${fmtDate(m.generated_at?.slice(0, 10))}`;
}).catch(() => {});

startRouter();
