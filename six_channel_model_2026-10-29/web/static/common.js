"use strict";
// Shared helpers for the six-channel pages: DOM/SVG builders, scales, axes, tooltip,
// legends and a crosshair line chart.  Data labels always go in through textContent.

const NS = "http://www.w3.org/2000/svg";
const tooltip = document.getElementById("tooltip");
const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const levelColor = (level) => css(level === 1 ? "--series-1" : level === 2 ? "--series-2" : level === 3 ? "--series-3" : "--neutral-mark");
const levelName = (level) => (level ? `L${level}` : "OOD");
const DASHES = ["", "6 4", "2 3"];

const LABELS = {
  roll_t_r: "滚转模态 T_R", roll_spiral: "螺旋模态", roll_dutch_roll: "荷兰滚（含 φ/β 修正）",
  roll_sensitivity_1s: "1 s 滚转灵敏度", roll_oscillation: "滚转振荡", sideslip_excursion: "侧滑偏离 Δβ/k",
  steady_sideslip_sense: "定常侧滑方向", pitch_frequency: "短周期频率 / CAP", pitch_damping: "短周期阻尼",
  pitch_delay: "俯仰等效延迟", pitch_stick_force: "每 g 杆力", long_period: "长周期",
  t_r_beyond_level_3: "T_R 超出 3 级", spiral_beyond_level_3: "螺旋超出 3 级",
  dutch_roll_beyond_level_3: "荷兰滚超出 3 级", sensitivity_1s_above_level_2_ungraded: "灵敏度超出 2 级",
  l_fa_outside_iv_force_input_evidence: "L_Fa 超出证据范围", r_omega_outside_evidence_envelope: "ωφ/ωd 超出研究包络",
  zeta_phi_outside_research_envelope: "ζφ 超出研究包络", transport_delay_above_research_envelope: "滚转时延超出研究包络",
  long_period_beyond_level_3: "长周期发散（不属任何等级）",
  roll_delay: "滚转等效延迟", pitch_inv_t_theta1: "长周期零点 1/Tθ1",
  steady_sideslip_sense_beyond_level_3: "定常侧滑方向不满足", pitch_inv_t_theta1_beyond_level_3: "1/Tθ1 ≤ 0",
  roll_delay_beyond_level_3: "滚转延迟超出 3 级", roll_dutch_roll_beyond_level_3: "荷兰滚（修正后）超出 3 级",
  sideslip_excursion_beyond_level_3: "侧滑偏离超出 2 级",
};
// ---------------------------------------------------------------- small helpers
function svgEl(tag, attrs = {}, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}
function htmlEl(tag, attrs = {}, text, parent) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined && text !== null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}
function fmt(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const a = Math.abs(value);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return value.toExponential(2);
  return Number(value.toPrecision(digits)).toLocaleString("en-US", { maximumFractionDigits: 6 });
}
function quantile(sorted, q) {
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}
function niceTicks(lo, hi, count = 5) {
  if (!(hi > lo)) return [lo];
  const step0 = (hi - lo) / count, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Number(v.toPrecision(12)));
  return ticks;
}
function logTicks(lo, hi) {
  const ticks = [];
  for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
    for (const m of [1, 2, 5]) { const v = m * 10 ** e; if (v >= lo * 0.999 && v <= hi * 1.001) ticks.push(v); }
  return ticks;
}
function scale(domain, range, log) {
  const [d0, d1] = log ? domain.map(Math.log10) : domain;
  const f = (v) => range[0] + ((log ? Math.log10(v) : v) - d0) / (d1 - d0 || 1) * (range[1] - range[0]);
  f.invert = (p) => { const v = d0 + (p - range[0]) / (range[1] - range[0]) * (d1 - d0); return log ? 10 ** v : v; };
  return f;
}
function showTooltip(event, head, rows) {
  tooltip.replaceChildren();
  htmlEl("div", { class: "head" }, head, tooltip);
  for (const row of rows) {
    const line = htmlEl("div", { class: "row" }, null, tooltip);
    if (row.color) {
      const key = htmlEl("span", { class: "line-key" }, null, line);
      key.style.borderTopColor = row.color;
      if (row.dash) key.style.borderTopStyle = "dashed";
    }
    htmlEl("b", {}, row.value, line);
    htmlEl("span", { class: "name" }, row.name, line);
  }
  tooltip.style.display = "block";
  const pad = 14, w = tooltip.offsetWidth, h = tooltip.offsetHeight;
  let x = event.clientX + pad, y = event.clientY + pad;
  if (x + w > window.innerWidth - 4) x = event.clientX - w - pad;
  if (y + h > window.innerHeight - 4) y = event.clientY - h - pad;
  tooltip.style.left = `${x}px`; tooltip.style.top = `${y}px`;
}
const hideTooltip = () => { tooltip.style.display = "none"; };
function svgPoint(svg, event) {
  const p = svg.createSVGPoint(); p.x = event.clientX; p.y = event.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}
function legend(container, items) {
  container.replaceChildren();
  for (const item of items) {
    const span = htmlEl("span", { class: "item" }, null, container);
    const svg = svgEl("svg", { width: 18, height: 10 }, span);
    if (item.kind === "dot") svgEl("circle", { cx: 9, cy: 5, r: 4, fill: item.color }, svg);
    else if (item.kind === "ring") svgEl("circle", { cx: 9, cy: 5, r: 4, fill: "none", stroke: item.color, "stroke-width": 1.5 }, svg);
    else if (item.kind === "tick") svgEl("line", { x1: 9, x2: 9, y1: 0, y2: 10, stroke: item.color, "stroke-width": 2 }, svg);
    else if (item.kind === "cross") svgEl("path", { d: "M5,1L13,9M5,9L13,1", stroke: item.color, "stroke-width": 2 }, svg);
    else svgEl("line", { x1: 0, x2: 18, y1: 5, y2: 5, stroke: item.color, "stroke-width": 2, "stroke-dasharray": item.dash || "" }, svg);
    htmlEl("span", {}, item.label, span);
  }
}
function axes(svg, x, y, xt, yt, box, xfmt = fmt, yfmt = fmt) {
  for (const v of yt) {
    svgEl("line", { x1: box.l, x2: box.r, y1: y(v), y2: y(v), class: "grid-line" }, svg);
    svgEl("text", { x: box.l - 6, y: y(v) + 3.5, "text-anchor": "end" }, svg).textContent = yfmt(v);
  }
  svgEl("line", { x1: box.l, x2: box.r, y1: box.b, y2: box.b, class: "axis-line" }, svg);
  for (const v of xt) svgEl("text", { x: x(v), y: box.b + 14, "text-anchor": "middle" }, svg).textContent = xfmt(v);
}

// ---------------------------------------------------------------- charts
function lineChart(container, { title, t, series, xLabel = "s" }) {
  container.replaceChildren();
  const W = 380, H = 196, box = { l: 46, r: W - 10, t: 32, b: H - 24 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, container);
  svgEl("text", { x: 4, y: 14, class: "title" }, svg).textContent = title;
  const values = series.flatMap((s) => s.values).filter(Number.isFinite);
  let lo = Math.min(0, ...values), hi = Math.max(0, ...values);
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const yt = niceTicks(lo, hi, 4);
  lo = Math.min(lo, yt[0]); hi = Math.max(hi, yt[yt.length - 1]);
  const x = scale([t[0], t[t.length - 1]], [box.l, box.r]), y = scale([lo, hi], [box.b, box.t]);
  axes(svg, x, y, niceTicks(t[0], t[t.length - 1], 4), yt, box, (v) => `${fmt(v)}${v === 0 ? "" : ""}`);
  svgEl("text", { x: box.r, y: H - 2, "text-anchor": "end" }, svg).textContent = xLabel;
  series.forEach((s, i) => {
    const d = s.values.map((v, k) => (Number.isFinite(v) ? `${k && Number.isFinite(s.values[k - 1]) ? "L" : "M"}${x(t[k]).toFixed(1)},${y(v).toFixed(1)}` : "")).join("");
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash || DASHES[i] || "", "stroke-linejoin": "round" }, svg);
  });
  const hair = svgEl("line", { y1: box.t, y2: box.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden" }, svg);
  const hit = svgEl("rect", { x: box.l, y: box.t, width: box.r - box.l, height: box.b - box.t, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const p = svgPoint(svg, event);
    const tv = x.invert(p.x);
    let k = Math.round((tv - t[0]) / (t[1] - t[0])); k = Math.max(0, Math.min(t.length - 1, k));
    hair.setAttribute("x1", x(t[k])); hair.setAttribute("x2", x(t[k])); hair.setAttribute("visibility", "visible");
    showTooltip(event, `t = ${fmt(t[k])} s`, series.map((s, i) => ({ name: s.name, value: fmt(s.values[k]), color: s.color, dash: s.dash || DASHES[i] })));
  });
  hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); hideTooltip(); });
}
