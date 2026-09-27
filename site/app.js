// Renders the page from data/site.json + data/renders.json (built by the scripts in the repository)
// and drives the live demo. No framework, no build step.
"use strict";

const SVG = "http://www.w3.org/2000/svg";
// Colour follows the class, never its rank: fixed slot per class id.
const CLASS_SLOT = {
  red_light: "--s1", stop_line: "--s2", jaywalking: "--s3", failure_to_yield: "--s4",
  stopped_vehicle: "--s5", congestion: "--s6", wrong_way: "--s7", accident: "--s8", near_miss: "--s8", road_obstacle: "--muted",
};
const VIDEO_SLOTS = ["--s1", "--s2", "--s3", "--s4"];
const MAX_UPLOAD_MB = 600;
const MAX_UPLOAD_SEC = 120;
const tip = document.getElementById("tooltip");

const el = (tag, attrs = {}, parent) => {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
};
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const label = (id) => String(id).replace(/_/g, " ");

function showTip(evt, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const pad = 12, r = tip.getBoundingClientRect();
  let x = evt.clientX + pad, y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
const hideTip = () => { tip.hidden = true; };

function tiles(container, items) {
  container.innerHTML = items.map(([v, k]) => `<div class="tile"><div class="v">${esc(v)}</div><div class="k">${esc(k)}</div></div>`).join("");
}

function table(container, head, rows, numeric = []) {
  const cls = (i) => (numeric.includes(i) ? ' class="num"' : "");
  const th = head.map((h, i) => `<th${cls(i)}>${esc(h)}</th>`).join("");
  const tr = rows.map((r) => `<tr>${r.map((c, i) => `<td${cls(i)}>${esc(c)}</td>`).join("")}</tr>`).join("");
  container.innerHTML = `<div class="table-wrap"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`;
}

// Draw at the container's real width (readable on a phone) and redraw when it changes.
const observed = new WeakMap();
const resizer = new ResizeObserver((entries) => {
  for (const e of entries) {
    const draw = observed.get(e.target);
    const w = Math.round(e.contentRect.width);
    if (draw && w && w !== draw.w) { draw.w = w; draw.fn(w); }
  }
});
function responsive(container, fn) {
  const entry = { fn, w: container.clientWidth };
  observed.set(container, entry);
  resizer.observe(container);
  fn(container.clientWidth || 640);
}

function timeTicks(duration, width) {
  const step = [5, 10, 15, 30, 60, 120, 300].find((s) => duration / s <= Math.max(2, width / 80)) || 600;
  const out = [];
  for (let t = 0; t <= duration + 1e-6; t += step) out.push(t);
  return out;
}

// ------------------------------------------------------------------ line chart
// series: [{name, color, points: [[x, y], ...]}]; one y-axis only.
function lineChart(container, series, opts) {
  responsive(container, (W) => drawLine(container, series, opts, Math.max(300, W)));
}

function drawLine(container, series, { yMax, yLabel, xMax, refY, height = 220, fmtY = (v) => v.toFixed(2) }, W) {
  container.innerHTML = "";
  const multi = series.length > 1;
  const H = height, m = { l: 40, r: multi ? (W < 500 ? 12 : 64) : 12, t: 10, b: 26 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": yLabel }, container);
  const x = (v) => m.l + (v / xMax) * (W - m.l - m.r);
  const y = (v) => H - m.b - (Math.min(v, yMax) / yMax) * (H - m.t - m.b);
  for (let i = 0; i <= 4; i++) {
    const v = (yMax * i) / 4;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), class: i ? "gridline" : "axis" }, svg);
    el("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end", class: "tick" }, svg).textContent = fmtY(v);
  }
  for (const t of timeTicks(xMax, W)) el("text", { x: x(t), y: H - 6, "text-anchor": "middle", class: "tick" }, svg).textContent = fmtTime(t);
  if (refY !== undefined) el("line", { x1: m.l, x2: W - m.r, y1: y(refY), y2: y(refY), class: "ref" }, svg);
  for (const s of series) {
    if (!s.points.length) continue;
    const d = s.points.map(([px, py], i) => `${i ? "L" : "M"}${x(px).toFixed(1)},${y(py).toFixed(1)}`).join("");
    if (!multi) {  // soft area under a single series
      const last = s.points[s.points.length - 1], first = s.points[0];
      el("path", { d: `${d}L${x(last[0]).toFixed(1)},${y(0)}L${x(first[0]).toFixed(1)},${y(0)}Z`, class: "area", fill: `var(${s.color})` }, svg);
    }
    el("path", { d, class: "line", stroke: `var(${s.color})` }, svg);
  }
  if (multi && W >= 500) {  // direct labels at line ends (nudged apart); on phones the legend alone carries identity
    const labels = series.filter((s) => s.points.length).map((s) => { const [lx, ly] = s.points[s.points.length - 1]; return { s, x: x(lx) + 6, y: y(ly) + 4 }; })
      .sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) labels[i].y = Math.max(labels[i].y, labels[i - 1].y + 14);
    for (const l of labels) el("text", { x: l.x, y: l.y, class: "series-label" }, svg).textContent = l.s.name;
  }
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (evt) => {
    const box = svg.getBoundingClientRect();
    const t = Math.min(xMax, Math.max(0, ((evt.clientX - box.left) / box.width * W - m.l) / (W - m.l - m.r) * xMax));
    cross.setAttribute("x1", x(t)); cross.setAttribute("x2", x(t)); cross.setAttribute("visibility", "visible");
    const rows = series.filter((s) => s.points.length).map((s) => {
      const p = s.points.reduce((a, b) => (Math.abs(b[0] - t) < Math.abs(a[0] - t) ? b : a));
      return `<div><i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:var(${s.color});margin-right:6px"></i>${esc(s.name)}: <b>${fmtY(p[1])}</b></div>`;
    });
    showTip(evt, `<div>${fmtTime(t)}</div>${rows.join("")}`);
  });
  hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
  if (multi) {
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = series.map((s) => `<span><i style="background:var(${s.color})"></i>${esc(s.name)}</span>`).join("");
    container.appendChild(lg);
  }
}

// ---------------------------------------------------------------- event timeline
function timeline(container, events, classes, duration, onSeek) {
  responsive(container, (W) => drawTimeline(container, events, classes, duration, onSeek, Math.max(300, W)));
}

function drawTimeline(container, events, classes, duration, onSeek, W) {
  container.innerHTML = "";
  const rowH = 28, m = { l: W < 500 ? 108 : 140, r: 12, t: 6, b: 24 };
  const H = m.t + m.b + rowH * classes.length;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Event timeline by class" }, container);
  const x = (v) => m.l + (Math.min(v, duration) / duration) * (W - m.l - m.r);
  for (const t of timeTicks(duration, W - m.l)) {
    el("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, class: "gridline" }, svg);
    el("text", { x: x(t), y: H - 6, "text-anchor": "middle", class: "tick" }, svg).textContent = fmtTime(t);
  }
  classes.forEach((cls, i) => {
    const yc = m.t + i * rowH;
    el("text", { x: m.l - 8, y: yc + rowH / 2 + 4, "text-anchor": "end", class: "row-label" }, svg).textContent = label(cls);
    el("line", { x1: m.l, x2: W - m.r, y1: yc + rowH, y2: yc + rowH, class: "gridline" }, svg);
    const mine = events.filter((e) => e[2] === cls);
    if (!mine.length) el("text", { x: m.l + 6, y: yc + rowH / 2 + 4, class: "empty" }, svg).textContent = "none";
    for (const [s, e] of mine) {
      const r = el("rect", {
        x: x(s), y: yc + 5, width: Math.max(6, x(e) - x(s)), height: rowH - 10, class: "seg",
        fill: `var(${CLASS_SLOT[cls] || "--muted"})`, tabindex: 0, role: "button",
        "aria-label": `${label(cls)} ${fmtTime(s)} to ${fmtTime(e)}${onSeek ? ", play" : ""}`,
      }, svg);
      const html = `<b>${esc(label(cls))}</b><div>${fmtTime(s)} – ${fmtTime(e)} (${(e - s).toFixed(1)} s)</div>`;
      r.addEventListener("pointermove", (evt) => showTip(evt, html));
      r.addEventListener("pointerleave", hideTip);
      if (onSeek) {
        r.addEventListener("click", () => onSeek(s));
        r.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); onSeek(s); } });
      }
      r.addEventListener("focus", () => { const b = r.getBoundingClientRect(); showTip({ clientX: b.right, clientY: b.top }, html); });
      r.addEventListener("blur", hideTip);
    }
  });
}

// ------------------------------------------------------------------------- page
function seekTo(player, t) {
  if (!player.currentSrc && !player.src) return;
  player.currentTime = Math.max(0, t - 1);
  player.play().catch(() => {});
  player.scrollIntoView({ behavior: "smooth", block: "center" });
}

function gallery(container, items, caption, empty = "None in this video.") {
  container.innerHTML = items.length
    ? items.map((it) => {
      const { html, text } = caption(it);
      return `<figure><button type="button" class="zoom" data-src="${esc(it.img)}" data-cap="${esc(text)}" aria-label="Enlarge: ${esc(text)}"><img src="${esc(it.img)}" alt="${esc(text)}" loading="lazy" width="1280" height="720"></button><figcaption>${html}</figcaption></figure>`;
    }).join("")
    : `<p class="empty-note">${esc(empty)}</p>`;
}

// ------------------------------------------------------------------ lightbox
const lightbox = document.getElementById("lightbox");
function openLightbox(src, cap) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox-img").alt = cap;
  document.getElementById("lightbox-cap").textContent = cap;
  if (lightbox.showModal) lightbox.showModal(); else window.open(src, "_blank", "noopener");
}
document.addEventListener("click", (e) => {
  const z = e.target.closest("button.zoom");
  if (z) return openLightbox(z.dataset.src, z.dataset.cap);
  const img = e.target.closest("img.zoomable");
  if (img) return openLightbox(img.currentSrc || img.src, img.alt);
  if (e.target === lightbox || e.target.closest(".lb-close")) lightbox.close();
});

const exampleCaption = (e) => ({
  html: `<span class="tag">${esc(label(e.label))}</span>${fmtTime(e.start)} – ${fmtTime(e.end)}`,
  text: `${label(e.label)} from ${fmtTime(e.start)} to ${fmtTime(e.end)}`,
});

function renderVideo(v, enabled, renders) {
  const player = document.getElementById("sample-player");
  const r = (renders && renders.videos[v.id]) || { examples: [], failures: [] };
  if (r.render) {
    player.poster = (r.examples[0] || r.failures[0] || {}).img || "";
    player.src = r.render;
    player.hidden = false;
  } else {
    player.removeAttribute("src");
    player.hidden = true;
  }
  gallery(document.getElementById("examples"), r.examples, exampleCaption);
  gallery(document.getElementById("failures"), r.failures, (f) => ({
    html: `<span class="tag fail">${esc(f.kind)}</span><span class="tag">${esc(label(f.label))} · ${fmtTime(f.t)}</span><br><b>${esc(f.title)}</b><br><span class="muted">${esc(f.note)}</span>`,
    text: `${f.title}: ${label(f.label)} at ${fmtTime(f.t)}`,
  }));
  tiles(document.getElementById("video-tiles"), [
    [fmtTime(v.duration_sec), "duration"],
    [v.events.length, "events"],
    [v.runtime.total_sec ? `${(v.runtime.total_sec / v.duration_sec).toFixed(2)}×` : "–", "processing time vs length"],
    [v.risk_summary.mean !== null ? v.risk_summary.mean.toFixed(3) : "–", "mean risk"],
  ]);
  timeline(document.getElementById("timeline-chart"), v.events, enabled, v.duration_sec, (t) => seekTo(player, t));
  lineChart(document.getElementById("risk-chart"), [{ name: "risk", color: "--risk", points: v.risk }],
    { yMax: 1, yLabel: "Accident risk over time", xMax: v.duration_sec, refY: 0.5, height: 180 });
  table(document.getElementById("events-table"), ["Start (s)", "End (s)", "Class"],
    v.events.map(([s, e, c]) => [s.toFixed(2), e.toFixed(2), label(c)]), [0, 1]);
}

async function main() {
  const data = await (await fetch("data/site.json")).json();
  const renders = await fetch("data/renders.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const vids = data.videos;
  const enabled = Object.entries(data.classes).filter(([, on]) => on).map(([k]) => k);
  window.__enabledClasses = enabled;
  const totalMin = vids.reduce((a, v) => a + v.duration_sec, 0) / 60;
  const events = vids.reduce((a, v) => a + v.events.length, 0);
  const factor = Math.max(...vids.map((v) => (v.runtime.total_sec || 0) / v.duration_sec));

  document.getElementById("class-chips").innerHTML = enabled.map((c) =>
    `<li><i style="background:var(${CLASS_SLOT[c] || "--muted"})"></i>${esc(label(c))}</li>`).join("");
  tiles(document.getElementById("hero-tiles"), [
    [`${enabled.length} / 14`, "event classes emitted"], [events, "events found on the samples"],
    [`${factor.toFixed(2)}×`, "worst runtime (limit 3×)"], [`${totalMin.toFixed(0)} min`, "of 4K sample footage"],
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
    b.type = "button";
    b.setAttribute("role", "tab");
    b.textContent = v.id;
    b.setAttribute("aria-selected", String(i === 0));
    b.title = `${fmtTime(v.duration_sec)} · ${v.events.length} events`;
    b.addEventListener("click", () => {
      tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", "false"));
      b.setAttribute("aria-selected", "true");
      renderVideo(v, enabled, renders);
    });
    tabs.appendChild(b);
  });
  renderVideo(vids[0], enabled, renders);
}

// ------------------------------------------------------------------ live demo
// Same origin when served by a local demo server (development); otherwise the deployed API from the <meta> tag.
const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const API = LOCAL ? "" : (document.querySelector('meta[name="demo-api"]')?.content || "").replace(/\/$/, "");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function readJson(resp) {
  // A proxy error page (HTML) must not surface as "Unexpected token <".
  const text = await resp.text();
  let body;
  try { body = JSON.parse(text); } catch { return { detail: `the server answered ${resp.status} ${resp.statusText || ""}`.trim() }; }
  if (body && body.detail && typeof body.detail !== "string") body.detail = "the server rejected the request";
  return body;
}

function clipDuration(file) {
  // Read the duration locally so an over-long clip is refused before a large upload.
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const url = URL.createObjectURL(file);
    const done = (d) => { URL.revokeObjectURL(url); resolve(d); };
    v.preload = "metadata";
    v.onloadedmetadata = () => done(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => done(null);
    setTimeout(() => done(null), 4000);  // unknown duration: the server still enforces the limit
    v.src = url;
  });
}

function showDemoResult(res) {
  const player = document.getElementById("demo-player");
  player.src = API + res.render;
  tiles(document.getElementById("demo-tiles"), [
    [fmtTime(res.duration_sec), "duration"], [res.events.length, "events"],
    [`${(res.timing_sec.part_a + res.timing_sec.part_b).toFixed(1)} s`, "processing (A + B)"], [res.vlm ? "on" : "off", "VLM verifier"],
  ]);
  const classes = window.__enabledClasses || [...new Set(res.events.map((e) => e[2]))];
  document.getElementById("demo-result").hidden = false;  // visible first, so charts measure their real width
  timeline(document.getElementById("demo-timeline"), res.events, classes, res.duration_sec, (t) => seekTo(player, t));
  lineChart(document.getElementById("demo-risk"), [{ name: "risk", color: "--risk", points: res.risk }],
    { yMax: 1, yLabel: "Accident risk over time", xMax: res.duration_sec, refY: 0.5, height: 180 });
  gallery(document.getElementById("demo-examples"), res.examples.map((e) => ({ ...e, img: API + e.img })), exampleCaption, "No events in this clip.");
  table(document.getElementById("demo-events"), ["Start (s)", "End (s)", "Class"], res.events.map(([s, e, c]) => [s.toFixed(2), e.toFixed(2), label(c)]), [0, 1]);
}

let demoBusy = false;
document.getElementById("demo-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (demoBusy) return;  // a second click while checking or uploading must not start a second job
  const input = document.getElementById("demo-file");
  const file = input.files[0];
  const status = document.getElementById("demo-status"), bar = document.getElementById("demo-bar"), stage = document.getElementById("demo-stage");
  const button = document.getElementById("demo-submit");
  const fail = (msg) => { status.textContent = msg; status.classList.add("error"); };
  const lock = (on) => { demoBusy = on; button.disabled = on; input.disabled = on; };
  status.classList.remove("error");
  if (!file) return fail("Choose an MP4 file first.");
  if (!/\.mp4$/i.test(file.name)) return fail("Only .mp4 files are accepted.");
  if (file.size === 0) return fail("The file is empty.");
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return fail(`The file is ${(file.size / 2 ** 20).toFixed(0)} MB; the limit is ${MAX_UPLOAD_MB} MB.`);

  lock(true);
  document.getElementById("demo-result").hidden = true;
  try {
    status.textContent = "Checking the clip…";
    const duration = await clipDuration(file);
    if (duration !== null && duration > MAX_UPLOAD_SEC + 0.5) return fail(`The clip is ${Math.round(duration)} s long; the demo accepts up to ${MAX_UPLOAD_SEC} s.`);

    const body = new FormData();
    body.append("video", file);
    body.append("full", document.getElementById("demo-full").checked ? "true" : "false");
    document.getElementById("demo-progress").hidden = false;
    stage.textContent = "uploading…";
    bar.removeAttribute("value");
    status.textContent = "Working…";
    const resp = await fetch(`${API}/api/jobs`, { method: "POST", body });
    const created = await readJson(resp);
    if (!resp.ok) throw new Error(created.detail || `upload failed (${resp.status})`);
    let misses = 0;
    for (;;) {
      await sleep(1500);
      let job;
      try {
        const r = await fetch(`${API}/api/jobs/${encodeURIComponent(created.id)}`);
        job = await readJson(r);
        if (!r.ok) throw new Error(job.detail || `status ${r.status}`);
        misses = 0;
      } catch (err) {
        if (++misses > 10) throw new Error(`lost contact with the server (${err.message})`);
        stage.textContent = "reconnecting…";
        continue;
      }
      bar.value = job.progress;
      stage.textContent = job.status === "queued" ? `waiting in queue (position ${job.queue_position + 1})` : `${job.stage} · ${Math.round(job.progress * 100)}%`;
      if (job.status === "done") { status.textContent = "Done."; showDemoResult(job.result); break; }
      if (job.status === "error") throw new Error(job.error || "processing failed");
    }
  } catch (err) {
    const offline = err instanceof TypeError;  // fetch() itself failed: no network / server down / CORS
    fail(offline ? "Could not reach the demo server. Check your connection and try again in a minute." : `Demo failed: ${err.message}`);
  } finally {
    lock(false);
    document.getElementById("demo-progress").hidden = true;
  }
});

// ------------------------------------------------------------------ theme, nav, dropzone
document.getElementById("theme-btn").addEventListener("click", () => {
  const root = document.documentElement;
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch { /* private mode: theme just is not remembered */ }
});

const navLinks = [...document.querySelectorAll("nav a")];
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    navLinks.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
  }
}, { rootMargin: "-45% 0px -50% 0px" });
navLinks.forEach((a) => { const sec = document.querySelector(a.getAttribute("href")); if (sec) spy.observe(sec); });

const drop = document.getElementById("dropzone"), fileInput = document.getElementById("demo-file");
const showFile = () => {
  const f = fileInput.files[0];
  drop.classList.toggle("has-file", !!f);
  document.getElementById("drop-title").textContent = f ? `${f.name} · ${(f.size / 2 ** 20).toFixed(1)} MB` : "Drop an MP4 here or click to choose";
};
fileInput.addEventListener("change", showFile);
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) { fileInput.files = e.dataTransfer.files; showFile(); }
});

main().catch((err) => {
  document.getElementById("hero-tiles").textContent = `Could not load the results data (${err.message}).`;
});
