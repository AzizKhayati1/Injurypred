// Entry point: routes, theme, mobile navigation.
import { getMeta } from "./data.js";
import { navigate, render, route, setNotFound, startRouter } from "./router.js";
import { fmtDate } from "./ui.js";
import { aboutView } from "./views/about.js";
import { athleteView } from "./views/athlete.js";
import { athletesView } from "./views/athletes.js";
import { heatmapView } from "./views/heatmap.js";
import { modelView } from "./views/model.js";
import { notFoundView } from "./views/notfound.js";
import { simulatorIndexView, simulatorView } from "./views/simulator.js";
import { squadView } from "./views/squad.js";
import { workloadView } from "./views/workload.js";

route("/", async () => navigate("/squad", { replace: true }), "squad");
route("/squad", squadView, "squad");
route("/athletes", athletesView, "athletes");
route("/athletes/:id", athleteView, "athletes");
route("/athletes/:id/workload", workloadView, "athletes");
route("/heatmap", heatmapView, "heatmap");
route("/simulator", simulatorIndexView, "simulator");
route("/simulator/:id", simulatorView, "simulator");
route("/model", modelView, "model");
route("/about", aboutView, "about");
setNotFound(notFoundView);

// ---- theme
const themeBtn = document.getElementById("theme-btn");
themeBtn.addEventListener("click", () => {
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
const sidebar = document.getElementById("sidebar");
const scrim = document.getElementById("scrim");
const menuBtn = document.getElementById("menu-btn");
menuBtn.addEventListener("click", () => {
  const open = !sidebar.classList.contains("open");
  sidebar.classList.toggle("open", open);
  scrim.hidden = !open;
  menuBtn.setAttribute("aria-expanded", String(open));
});
scrim.addEventListener("click", () => menuBtn.click());
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && sidebar.classList.contains("open")) menuBtn.click(); });

// ---- header info
getMeta().then((m) => {
  document.getElementById("asof").textContent = `Data to ${fmtDate(m.latest_date)}`;
  document.getElementById("build-info").textContent = `Model ${m.model.model_version} · ${m.model.model_type}`;
}).catch(() => {});

// Chart.js is loaded with `defer`; module scripts run after deferred classic scripts, so it is ready here.
startRouter();
