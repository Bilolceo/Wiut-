// Renders the page from data/site.json (built by scripts/build_site_data.py). No framework, no build step.
"use strict";

const SVG = "http://www.w3.org/2000/svg";
// Colour follows the class, never its rank: fixed slot per class id.
const CLASS_SLOT = {
  red_light: "--s1", stop_line: "--s2", jaywalking: "--s3", failure_to_yield: "--s4",
  stopped_vehicle: "--s5", congestion: "--s6", wrong_way: "--s7",
};
const VIDEO_SLOTS = ["--s1", "--s2", "--s3", "--s4"];
const tip = document.getElementById("tooltip");

const el = (tag, attrs = {}, parent) => {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
};
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const label = (id) => id.replace(/_/g, " ");

function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 12, r = tip.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
const hideTip = () => { tip.hidden = true; };

function tiles(container, items) {
  container.innerHTML = items.map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
}

function table(container, head, rows, numeric = []) {
  const th = head.map((h, i) => `<th class="${numeric.includes(i) ? "num" : ""}">${h}</th>`).join("");
  const tr = rows.map((r) => `<tr>${r.map((c, i) => `<td class="${numeric.includes(i) ? "num" : ""}">${c}</td>`).join("")}</tr>`).join("");
  container.innerHTML = `<table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table>`;
}

// ------------------------------------------------------------------ line chart
// series: [{name, color, points: [[x, y], ...]}]; one y-axis only.
function lineChart(container, series, { yMax, yLabel, xMax, refY, height = 240, fmtY = (v) => v.toFixed(2) }) {
  container.innerHTML = "";
  const W = 960, H = height, m = { l: 44, r: series.length > 1 ? 90 : 16, t: 10, b: 28 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": yLabel }, container);
  const x = (v) => m.l + (v / xMax) * (W - m.l - m.r);
  const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: i ? "gridline" : "axis" }, svg);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = fmtY(v);
  }
  const step = xMax > 240 ? 60 : 30;
  for (let t = 0; t <= xMax; t += step) el("text", { x: x(t), y: H - 8, "text-anchor": "middle", class: "tick" }, svg).textContent = fmtTime(t);
  if (refY !== undefined) el("line", { x1: m.l, x2: W - m.r, y1: y(refY), y2: y(refY), class: "ref" }, svg);
  for (const s of series) {
    const d = s.points.map(([px, py], i) => `${i ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`).join("");
    el("path", { d, class: "line", stroke: `var(${s.color})` }, svg);
  }
  if (series.length > 1) {  // direct labels at the line ends, nudged apart so they never collide
    const labels = series.map((s) => { const [lx, ly] = s.points[s.points.length - 1]; return { s, x: x(lx) + 6, y: y(ly) + 4 }; })
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
    for (const l of labels) el("text", { x: l.x, y: l.y, class: "series-label" }, svg).textContent = l.s.name;
  }
  // crosshair + tooltip
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (evt) => {
    const box = svg.getBoundingClientRect();
    const t = ((evt.clientX - box.left) / box.width * W - m.l) / (W - m.l - m.r) * xMax;
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("visibility", "visible");
    const rows = series.map((s) => {
      const p = s.points.reduce((a, b) => (Math.abs(b[0] - t) < Math.abs(a[0] - t) ? b : a));
      return `<div><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(${s.color});margin-right:6px"></i>${s.name}: <b>${fmtY(p[1])}</b></div>`;
    });
    showTip(evt, `<div>${fmtTime(Math.max(0, t))}</div>${rows.join("")}`);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  if (series.length > 1) {
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = series.map((s) => `<span><i style="background:var(${s.color})"></i>${s.name}</span>`).join("");
    container.appendChild(lg);
  }
}

// ---------------------------------------------------------------- event timeline
function timeline(container, events, classes, duration) {
  container.innerHTML = "";
  const rows = classes;
  const rowH = 26, W = 960, m = { l: 140, r: 16, t: 6, b: 26 };
  const H = m.t + m.b + rowH * rows.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Event timeline by class" }, container);
  const x = (v) => m.l + (v / duration) * (W - m.l - m.r);
  const step = duration > 240 ? 60 : 30;
  for (let t = 0; t <= duration; t += step) {
    el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "gridline" }, svg);
    el("text", { x: x(t), y: H - 8, "text-anchor": "middle", class: "tick" }, svg).textContent = fmtTime(t);
  }
  rows.forEach((cls, i) => {
    const yc = m.t + i * rowH;
    el("text", { x: m.l - 10, y: yc + rowH / 2 + 4, "text-anchor": "end", class: "row-label" }, svg).textContent = label(cls);
    el("line", { x1: m.l, x2: W - m.r, y1: yc + rowH, y2: yc + rowH, class: "gridline" }, svg);
    const mine = events.filter((e) => e[2] === cls);
    if (!mine.length) {
      el("text", { x: m.l + 6, y: yc + rowH / 2 + 4, class: "empty" }, svg).textContent = "none";
    }
    for (const [s, e] of mine) {
      const r = el("rect", {
        x: x(s), y: yc + 5, width: Math.max(4, x(e) - x(s)), height: rowH - 10, class: "seg",
        fill: `var(${CLASS_SLOT[cls] || "--muted"})`, tabindex: 0, "aria-label": `${label(cls)} ${fmtTime(s)} to ${fmtTime(e)}`,
      }, svg);
      const html = `<b>${label(cls)}</b><div>${fmtTime(s)} – ${fmtTime(e)} (${(e - s).toFixed(1)} s)</div>`;
      r.addEventListener("pointermove", (evt) => showTip(evt, html));
      r.addEventListener("pointerleave", hideTip);
      r.addEventListener("focus", () => { const b = r.getBoundingClientRect(); showTip({ clientX: b.right, clientY: b.top }, html); });
      r.addEventListener("blur", hideTip);
    }
  });
}

// ------------------------------------------------------------------------- page
function renderVideo(data, v, enabled) {
  const counts = v.events.length;
  tiles(document.getElementById("video-tiles"), [
    [fmtTime(v.duration_sec), "duration"],
    [counts, "events"],
    [v.runtime.total_sec ? `${(v.runtime.total_sec / v.duration_sec).toFixed(2)}×` : "–", "runtime A+B / duration (limit 3×)"],
    [v.risk_summary.mean !== null ? v.risk_summary.mean.toFixed(3) : "–", "mean risk"],
  ]);
  timeline(document.getElementById("timeline-chart"), v.events, enabled, v.duration_sec);
  lineChart(document.getElementById("risk-chart"), [{ name: "risk", color: "--risk", points: v.risk }],
    { yMax: 1, yLabel: "Accident risk over time", xMax: v.duration_sec, refY: 0.5, height: 200 });
  table(document.getElementById("events-table"), ["Start", "End", "Class"],
    v.events.map(([s, e, c]) => [s.toFixed(2), e.toFixed(2), label(c)]), [0, 1]);
}

async function main() {
  const data = await (await fetch("data/site.json")).json();
  const vids = data.videos;
  const enabled = Object.entries(data.classes).filter(([, on]) => on).map(([k]) => k);
  const totalMin = vids.reduce((a, v) => a + v.duration_sec, 0) / 60;
  const events = vids.reduce((a, v) => a + v.events.length, 0);
  const factor = Math.max(...vids.map((v) => (v.runtime.total_sec || 0) / v.duration_sec));

  tiles(document.getElementById("hero-tiles"), [
    [`${enabled.length} / 14`, "classes emitted"], [events, "events on the samples"],
    [`${factor.toFixed(2)}×`, "worst runtime (limit 3×)"], [vids.length, "sample videos"],
  ]);
  tiles(document.getElementById("eda-tiles"), [
    [`${vids[0].width}×${vids[0].height}`, "resolution (4K)"], [`${vids[0].fps} fps`, "frame rate"],
    [`${totalMin.toFixed(1)} min`, "total sample footage"], ["0", "night videos"],
  ]);

  const bSeries = vids.map((v, i) => ({ name: v.id, color: VIDEO_SLOTS[i], points: v.brightness }));
  lineChart(document.getElementById("brightness-chart"), bSeries,
    { yMax: 120, yLabel: "Mean brightness over time per video", xMax: Math.max(...vids.map((v) => v.duration_sec)), fmtY: (v) => v.toFixed(0) });
  table(document.getElementById("brightness-table"), ["Video", "Mean", "Min", "Max"], vids.map((v) => {
    const b = v.brightness.map((p) => p[1]);
    return [v.id, (b.reduce((a, c) => a + c, 0) / b.length).toFixed(0), Math.min(...b).toFixed(0), Math.max(...b).toFixed(0)];
  }), [1, 2, 3]);
  table(document.getElementById("shift-table"), ["Video", "Shift x (px)", "Shift y (px)"],
    vids.map((v) => [v.id, v.shift_px ? v.shift_px[0] : "–", v.shift_px ? v.shift_px[1] : "–"]), [1, 2]);

  const tabs = document.getElementById("video-tabs");
  vids.forEach((v, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.role = "tab"; b.textContent = v.id; b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", () => {
      tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", "false"));
      b.setAttribute("aria-selected", "true");
      renderVideo(data, v, enabled);
    });
    tabs.appendChild(b);
  });
  renderVideo(data, vids[0], enabled);
}

document.getElementById("demo-form").addEventListener("submit", (e) => e.preventDefault());
main().catch((err) => {
  document.getElementById("hero-tiles").textContent = `Could not load data/site.json (${err.message}). Serve the folder over HTTP: python -m http.server -d site`;
});
