// Thin, theme-aware wrapper around Chart.js (vendored in /vendor).
import { css } from "./ui.js";

const live = new Set();

export function destroyAll() {
  live.forEach((c) => c.destroy());
  live.clear();
}

function theme() {
  return { text: css("--text-2"), muted: css("--muted"), grid: css("--grid"), surface: css("--surface"), ink: css("--text") };
}

export const series = () => ["--s1", "--s2", "--s3", "--s4", "--s7", "--s8"].map(css);

export function chart(canvas, { type = "line", labels, datasets, yFormat, yMin, yMax, legend = true, xTicks = 8,
                               stacked = false, indexAxis = "x", tooltipFormat } = {}) {
  if (!window.Chart) return null;
  const t = theme();
  const fmt = yFormat || ((v) => v);
  const c = new window.Chart(canvas, {
    type,
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 }, indexAxis,
      interaction: { mode: "index", intersect: false },
      layout: { padding: { top: 4 } },
      plugins: {
        legend: { display: legend, position: "bottom", labels: { color: t.text, boxWidth: 10, boxHeight: 10, usePointStyle: true, padding: 14,
          filter: (item, data) => !data.datasets[item.datasetIndex].hideLegend } },
        tooltip: { backgroundColor: t.ink, titleColor: t.surface, bodyColor: t.surface, padding: 10, cornerRadius: 8, boxPadding: 4,
          filter: (item) => !item.dataset.hideTooltip,
          callbacks: { label: (ctx) => `${ctx.dataset.label}: ${(tooltipFormat || fmt)(indexAxis === "y" ? ctx.parsed.x : ctx.parsed.y)}` } },
      },
      scales: {
        // only the value axis gets a formatter; the category axis keeps Chart.js' default (the label text)
        x: { stacked, grid: { display: indexAxis === "y", color: t.grid }, border: { color: t.grid },
             ticks: { color: t.muted, maxTicksLimit: xTicks, maxRotation: 0, autoSkip: true,
                      ...(indexAxis === "y" ? { callback: (v) => fmt(v) } : {}) } },
        y: { stacked, min: yMin, max: yMax, grid: { color: t.grid, display: indexAxis !== "y" }, border: { display: false },
             ticks: { color: t.muted, ...(indexAxis === "y" ? { autoSkip: false } : { maxTicksLimit: 6, callback: (v) => fmt(v) }) } },
      },
      elements: { line: { borderWidth: 2, tension: 0.25 }, point: { radius: 0, hoverRadius: 4 }, bar: { borderRadius: 4 } },
    },
  });
  live.add(c);
  return c;
}

export const line = (label, data, color, extra = {}) =>
  ({ label, data, borderColor: color, backgroundColor: color, spanGaps: false, ...extra });

export const threshold = (label, value, n, color) =>
  ({ label, data: Array(n).fill(value), borderColor: color, borderDash: [5, 4], borderWidth: 1.2, pointRadius: 0,
     hoverRadius: 0, hideTooltip: true, tension: 0 });
