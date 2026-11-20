"use strict";
// Library viewer page (library.html); helpers come from common.js.

const AXES = [
  ["omega_d", "ωd 荷兰滚频率 (rad/s)", true], ["zeta_d", "ζd 荷兰滚阻尼", false], ["t_r", "T_R 滚转模态 (s)", true],
  ["lambda_s", "λs 螺旋根 (1/s)", false], ["r_omega", "ωφ/ωd", false], ["zeta_phi", "ζφ", false],
  ["sensitivity", "1 s 灵敏度 (deg/N)", true], ["eta", "副翼偏航 N′a/L′a", false], ["phi_over_beta", "|φ/β|d", true],
  ["dihedral", "上反比 −L′β/N′β", false], ["dbeta_over_k", "侧滑偏离 Δβ/k (deg)", true], ["psi_beta", "ψβ (deg)", false],
  ["v0", "V0 (m/s)", false], ["omega_sp", "ωsp (rad/s)", true], ["zeta_sp", "ζsp", false], ["cap", "CAP", true],
  ["tau_theta", "τθ (s)", false], ["zeta_p", "ζp 长周期阻尼", false], ["y_b", "Yβ (1/s)", false],
];
const AXIS = Object.fromEntries(AXES.map(([key, label, log]) => [key, { key, label, log }]));

function histChart(container, { title, log, series, rug, bins = 30 }) {
  container.replaceChildren();
  const W = 320, H = 206, box = { l: 38, r: W - 8, t: 32, b: H - 34 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, container);
  svgEl("text", { x: 4, y: 14, class: "title" }, svg).textContent = title;
  const pool = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v) && (!log || v > 0)).sort((a, b) => a - b);
  let lo = quantile(pool, 0.005), hi = quantile(pool, 0.995);
  if (rug) { const r = rug.values.filter((v) => Number.isFinite(v) && (!log || v > 0)); lo = Math.min(lo, ...r); hi = Math.max(hi, ...r); }
  const edges = Array.from({ length: bins + 1 }, (_, i) => (log ? 10 ** (Math.log10(lo) + i / bins * (Math.log10(hi) - Math.log10(lo))) : lo + i / bins * (hi - lo)));
  const binOf = (v) => { for (let i = 0; i < bins; i++) if (v >= edges[i] && (v < edges[i + 1] || (i === bins - 1 && v <= edges[bins]))) return i; return -1; };
  const shares = series.map((s) => {
    const counts = new Array(bins).fill(0); let n = 0;
    for (const v of s.values) { if (!Number.isFinite(v)) continue; n++; const b = binOf(v); if (b >= 0) counts[b]++; }
    return counts.map((c) => 100 * c / Math.max(n, 1));
  });
  const top = Math.max(...shares.flat(), 1);
  const yt = niceTicks(0, top, 3);
  const x = scale([lo, hi], [box.l, box.r], log), y = scale([0, Math.max(top, yt[yt.length - 1])], [box.b, box.t]);
  axes(svg, x, y, log ? logTicks(lo, hi) : niceTicks(lo, hi, 4), yt, box, fmt, (v) => `${fmt(v)}%`);
  series.forEach((s, i) => {
    let d = `M${x(edges[0]).toFixed(1)},${y(0).toFixed(1)}`;
    shares[i].forEach((p, b) => { d += `V${y(p).toFixed(1)}H${x(edges[b + 1]).toFixed(1)}`; });
    d += `V${y(0).toFixed(1)}`;
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.8, "stroke-dasharray": s.dash || "", "stroke-linejoin": "round" }, svg);
  });
  const rugCounts = new Array(bins).fill(0);
  if (rug) for (const v of rug.values) {
    if (!Number.isFinite(v) || v < lo || v > hi) continue;
    rugCounts[binOf(v)]++;
    svgEl("line", { x1: x(v), x2: x(v), y1: box.b + 18, y2: box.b + 26, stroke: rug.color, "stroke-width": 1.4 }, svg);
  }
  const band = svgEl("rect", { y: box.t, height: box.b - box.t, fill: css("--grid"), opacity: 0.6, visibility: "hidden" }, svg);
  band.parentNode.insertBefore(band, band.parentNode.children[1]);
  const hit = svgEl("rect", { x: box.l, y: box.t, width: box.r - box.l, height: box.b + 26 - box.t, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const b = binOf(x.invert(svgPoint(svg, event).x)); if (b < 0) return;
    band.setAttribute("x", x(edges[b])); band.setAttribute("width", Math.max(1, x(edges[b + 1]) - x(edges[b]))); band.setAttribute("visibility", "visible");
    const rows = series.map((s, i) => ({ name: s.name, value: `${fmt(shares[i][b], 2)}%`, color: s.color, dash: s.dash }));
    if (rug) rows.push({ name: rug.name, value: `${rugCounts[b]} 个`, color: rug.color });
    showTooltip(event, `${fmt(edges[b])} – ${fmt(edges[b + 1])}`, rows);
  });
  hit.addEventListener("pointerleave", () => { band.setAttribute("visibility", "hidden"); hideTooltip(); });
}

function barChart(container, { title, items, color, total }) {
  container.replaceChildren();
  htmlEl("h3", {}, title, container);
  const W = 400, row = 20, H = items.length * row + 8, labelW = 150;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, container);
  const max = Math.max(...items.map((i) => i.value), 1);
  items.forEach((item, k) => {
    const yv = 4 + k * row;
    svgEl("text", { x: labelW - 8, y: yv + 12, "text-anchor": "end" }, svg).textContent = item.label;
    const w = (W - labelW - 46) * item.value / max;
    const bar = svgEl("rect", { x: labelW, y: yv + 3, width: Math.max(w, 1), height: row - 8, rx: 2, fill: color }, svg);
    svgEl("text", { x: labelW + w + 6, y: yv + 12 }, svg).textContent = String(item.value);
    const hit = svgEl("rect", { x: 0, y: yv, width: W, height: row, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", (event) => {
      bar.setAttribute("opacity", 0.8);
      showTooltip(event, item.label, [{ name: total ? `占 ${fmt(100 * item.value / total, 2)}%` : "", value: `${item.value} 个对象` }]);
    });
    hit.addEventListener("pointerleave", () => { bar.removeAttribute("opacity"); hideTooltip(); });
  });
}

// ---------------------------------------------------------------- state and data
const state = { data: null, objects: [], selected: null, x: "omega_d", y: "zeta_d", levels: new Set([1, 2, 3, 0]),
  split: "all", real: true, steps: null, maneuvers: null };

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

function renderTiles() {
  const box = document.getElementById("tiles");
  box.replaceChildren();
  const m = state.data.manifest;
  const ratio = (arr) => { const s = arr.filter(Number.isFinite).sort((a, b) => a - b); return `${fmt(quantile(s, 0.05), 2)}–${fmt(quantile(s, 0.95), 3)}`; };
  const tiles = [
    ["5,000", "对象", "1/2/3 级各 1,500 · OOD 500"],
    [`${Math.round(100 * m.valid_candidates / m.candidate_count)}%`, "候选能对应真实飞机", `${m.valid_candidates.toLocaleString()} / ${m.candidate_count.toLocaleString()} 个候选`],
    [ratio(state.objects.map((o) => o.r_omega)), "ωφ/ωd（5–95 百分位，推导量）",
      `真实飞机 ${ratio(state.data.real.map((r) => r.r_omega))} · 原库 ${ratio(state.data.original.r_omega)}`],
    ["45 / 45", "项测试通过", "精确性、真实范围、等级复算、各级通用要求、可仿真"],
  ];
  for (const [value, label, sub] of tiles) {
    const card = htmlEl("div", { class: "card tile" }, null, box);
    htmlEl("div", { class: "value" }, value, card);
    htmlEl("div", { class: "label" }, label, card);
    htmlEl("div", { class: "sub" }, sub, card);
  }
}

function renderBinding() {
  const box = document.getElementById("binding");
  box.replaceChildren();
  for (const [level, title, color] of [[2, "L2：1,500 个对象的定级来源", "--series-2"], [3, "L3：1,500 个对象的定级来源", "--series-3"], [0, "OOD：500 个对象的原因", "--neutral-mark"]]) {
    const counts = {};
    for (const o of state.objects) if (o.level === level) counts[o.group] = (counts[o.group] || 0) + 1;
    const items = Object.entries(counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: LABELS[k] || k, value: v }));
    const card = htmlEl("div", { class: "card chart" }, null, box);
    barChart(card, { title, items, color: css(color), total: level ? 1500 : 500 });
  }
}

function renderDistributions() {
  const panels = [["t_r", "T_R 滚转模态 (s)", true], ["omega_d", "ωd 荷兰滚频率 (rad/s)", true], ["zeta_d", "ζd 荷兰滚阻尼", false],
    ["lambda_s", "λs 螺旋根 (1/s)", false], ["r_omega", "ωφ/ωd（v4 为推导量）", false], ["zeta_phi", "ζφ（v4 为推导量）", false],
    ["sensitivity", "1 s 灵敏度 (deg/N)", true], ["eta", "副翼偏航 N′a/L′a", false]];
  const v3 = css("--series-1"), old = css("--series-2"), real = css("--series-3");
  legend(document.getElementById("dist-legend"), [
    { label: "v4 对象库", color: v3 }, { label: "原滚转库", color: old, dash: "6 4" }, { label: "真实飞机（CR-2144，94 个状态）", color: real, kind: "tick" }]);
  const box = document.getElementById("distributions");
  box.replaceChildren();
  const table = document.getElementById("dist-table");
  table.replaceChildren();
  const head = htmlEl("tr", {}, null, table);
  for (const h of ["量", "v4 5%", "v4 中位", "v4 95%", "原库 5%", "原库 中位", "原库 95%", "真实 5%", "真实 中位", "真实 95%"]) htmlEl("th", {}, h, head);
  for (const [key, title, log] of panels) {
    const card = htmlEl("div", { class: "card chart" }, null, box);
    const series = [{ name: "v4", values: state.objects.map((o) => o[key]), color: v3 }];
    if (state.data.original[key]) series.push({ name: "原滚转库", values: state.data.original[key], color: old, dash: "6 4" });
    const realValues = state.data.real.map((r) => r[key]).filter((v) => v !== undefined);
    histChart(card, { title, log, series, rug: realValues.length ? { name: "真实飞机", values: realValues, color: real } : null });
    const row = htmlEl("tr", {}, null, table);
    htmlEl("td", {}, title, row);
    for (const values of [series[0].values, state.data.original[key] || [], realValues]) {
      const s = values.filter(Number.isFinite).sort((a, b) => a - b);
      for (const q of [0.05, 0.5, 0.95]) htmlEl("td", {}, s.length ? fmt(quantile(s, q)) : "—", row);
    }
  }
}

// ---------------------------------------------------------------- object browser
function renderFilters() {
  const box = document.getElementById("filters");
  box.replaceChildren();
  const counts = {}; for (const o of state.objects) counts[o.level] = (counts[o.level] || 0) + 1;
  for (const level of [1, 2, 3, 0]) {
    const label = htmlEl("label", {}, null, box);
    const input = htmlEl("input", { type: "checkbox" }, null, label);
    input.checked = state.levels.has(level);
    input.addEventListener("change", () => { input.checked ? state.levels.add(level) : state.levels.delete(level); renderScatter(); });
    const key = htmlEl("span", { class: "chip" }, null, label);
    htmlEl("span", { class: "key" }, null, key).style.background = levelColor(level);
    key.appendChild(document.createTextNode(`${levelName(level)} · ${counts[level] || 0}`));
  }
  const splitLabel = htmlEl("label", {}, "划分", box);
  const split = htmlEl("select", {}, null, splitLabel);
  for (const value of ["all", "train_core", "train_boundary", "validation", "id_test", "ood_test"]) {
    const option = htmlEl("option", { value }, value === "all" ? "全部" : value, split);
    if (value === state.split) option.selected = true;
  }
  split.addEventListener("change", () => { state.split = split.value; renderScatter(); });
  for (const [which, text] of [["x", "横轴"], ["y", "纵轴"]]) {
    const label = htmlEl("label", {}, text, box);
    const select = htmlEl("select", {}, null, label);
    for (const [key, name] of AXES) { const o = htmlEl("option", { value: key }, name, select); if (key === state[which]) o.selected = true; }
    select.addEventListener("change", () => { state[which] = select.value; renderScatter(); });
  }
  const realLabel = htmlEl("label", {}, null, box);
  const real = htmlEl("input", { type: "checkbox" }, null, realLabel);
  real.checked = state.real;
  realLabel.appendChild(document.createTextNode("显示真实飞机"));
  real.addEventListener("change", () => { state.real = real.checked; renderScatter(); });
  const searchLabel = htmlEl("label", {}, "对象", box);
  const search = htmlEl("input", { type: "search", placeholder: "如 train_core-0001", list: "ids", size: 18 }, null, searchLabel);
  const list = htmlEl("datalist", { id: "ids" }, null, box);
  for (const o of state.objects) htmlEl("option", { value: o.id }, null, list);
  search.addEventListener("change", () => { if (state.objects.some((o) => o.id === search.value)) select(search.value); });
}

function renderScatter() {
  const container = document.getElementById("scatter");
  container.replaceChildren();
  const ax = AXIS[state.x], ay = AXIS[state.y];
  const points = state.objects.filter((o) => state.levels.has(o.level) && (state.split === "all" || o.split === state.split)
    && Number.isFinite(o[ax.key]) && Number.isFinite(o[ay.key]) && (!ax.log || o[ax.key] > 0) && (!ay.log || o[ay.key] > 0));
  const real = state.real ? state.data.real.filter((r) => Number.isFinite(r[ax.key]) && Number.isFinite(r[ay.key])
    && (!ax.log || r[ax.key] > 0) && (!ay.log || r[ay.key] > 0)) : [];
  legend(document.getElementById("scatter-legend"), [
    ...[1, 2, 3, 0].filter((l) => state.levels.has(l)).map((l) => ({ label: levelName(l), color: levelColor(l), kind: "dot" })),
    ...(real.length ? [{ label: "真实飞机", color: css("--text-primary"), kind: "ring" }] : [])]);
  const W = 640, H = 470, box = { l: 54, r: W - 14, t: 14, b: H - 40 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `${ax.label} – ${ay.label}` }, container);
  const domain = (axis, rows) => {
    const s = rows.map((r) => r[axis.key]).sort((a, b) => a - b);
    let lo = quantile(s, 0.002), hi = quantile(s, 0.998);
    if (!(hi > lo)) { lo -= 1; hi += 1; }
    if (axis.log) return [lo / 1.08, hi * 1.08];
    const pad = (hi - lo) * 0.04; return [lo - pad, hi + pad];
  };
  const both = [...points, ...real];
  if (!both.length) return;
  const x = scale(domain(ax, both), [box.l, box.r], ax.log), y = scale(domain(ay, both), [box.b, box.t], ay.log);
  const [xl, xh] = domain(ax, both), [yl, yh] = domain(ay, both);
  axes(svg, x, y, ax.log ? logTicks(xl, xh) : niceTicks(xl, xh, 6), ay.log ? logTicks(yl, yh) : niceTicks(yl, yh, 6), box);
  svgEl("text", { x: (box.l + box.r) / 2, y: H - 6, "text-anchor": "middle" }, svg).textContent = ax.label;
  const yLabel = svgEl("text", { x: 14, y: (box.t + box.b) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(box.t + box.b) / 2})` }, svg);
  yLabel.textContent = ay.label;
  const clip = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const screen = [];
  for (const o of points) {
    const px = clip(x(o[ax.key]), box.l, box.r), py = clip(y(o[ay.key]), box.t, box.b);
    screen.push([px, py, o]);
    svgEl("circle", { cx: px.toFixed(1), cy: py.toFixed(1), r: 2.6, fill: levelColor(o.level), "fill-opacity": 0.72 }, svg);
  }
  for (const r of real) {
    const px = clip(x(r[ax.key]), box.l, box.r), py = clip(y(r[ay.key]), box.t, box.b);
    screen.push([px, py, { real: r }]);
    svgEl("circle", { cx: px.toFixed(1), cy: py.toFixed(1), r: 4.2, fill: "none", stroke: css("--text-primary"), "stroke-width": 1.3 }, svg);
  }
  const sel = state.objects.find((o) => o.id === state.selected);
  if (sel && points.includes(sel))
    svgEl("circle", { cx: x(sel[ax.key]), cy: y(sel[ay.key]), r: 7, fill: "none", stroke: css("--text-primary"), "stroke-width": 2 }, svg);
  const hover = svgEl("circle", { r: 6, fill: "none", stroke: css("--text-secondary"), "stroke-width": 1.5, visibility: "hidden" }, svg);
  const nearest = (event) => {
    const p = svgPoint(svg, event); let best = null, bd = 24 * 24;
    for (const s of screen) { const d = (s[0] - p.x) ** 2 + (s[1] - p.y) ** 2; if (d < bd) { bd = d; best = s; } }
    return best;
  };
  const hit = svgEl("rect", { x: box.l - 10, y: box.t - 10, width: box.r - box.l + 20, height: box.b - box.t + 20, fill: "transparent", cursor: "crosshair" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const s = nearest(event);
    if (!s) { hover.setAttribute("visibility", "hidden"); hideTooltip(); return; }
    hover.setAttribute("cx", s[0]); hover.setAttribute("cy", s[1]); hover.setAttribute("visibility", "visible");
    if (s[2].real) {
      const r = s[2].real;
      showTooltip(event, `真实飞机 ${r.aircraft} · 状态 ${r.fc}`, [{ name: ax.label, value: fmt(r[ax.key]) }, { name: ay.label, value: fmt(r[ay.key]) }]);
    } else {
      const o = s[2];
      showTooltip(event, `${o.id} · ${levelName(o.level)} · ${LABELS[o.group] || (o.group === "level_1" ? "全部 1 级" : o.group)}`,
        [{ name: ax.label, value: fmt(o[ax.key]) }, { name: ay.label, value: fmt(o[ay.key]) }, { name: "点击查看", value: "" }]);
    }
  });
  hit.addEventListener("pointerleave", () => { hover.setAttribute("visibility", "hidden"); hideTooltip(); });
  hit.addEventListener("click", (event) => { const s = nearest(event); if (s && !s[2].real) select(s[2].id); });
}

// ---------------------------------------------------------------- selected object
function paramTable(container, groups) {
  const table = htmlEl("table", { class: "params" }, null, container);
  for (const [group, rows] of groups) {
    const g = htmlEl("tr", { class: "group" }, null, table);
    htmlEl("td", { colspan: 2 }, group, g);
    for (const [name, value] of rows) {
      const tr = htmlEl("tr", {}, null, table);
      htmlEl("td", {}, name, tr); htmlEl("td", {}, value, tr);
    }
  }
}

async function select(id) {
  state.selected = id;
  history.replaceState(null, "", `#id=${encodeURIComponent(id)}`);
  renderScatter();
  const detailBox = document.getElementById("detail");
  detailBox.classList.add("loading");
  document.getElementById("steps").classList.add("loading");
  const [d, steps] = await Promise.all([getJSON(`api/object?id=${encodeURIComponent(id)}`), getJSON(`api/steps?id=${encodeURIComponent(id)}`)]);
  if (state.selected !== id) return;
  state.steps = steps; state.detail = d; state.maneuvers = null;
  renderDetail(); renderSteps();
  document.getElementById("maneuvers").replaceChildren();
  document.getElementById("maneuver-warn").replaceChildren();
  detailBox.classList.remove("loading"); document.getElementById("steps").classList.remove("loading");
}

function renderDetail() {
  const d = state.detail; if (!d) return;
  const box = document.getElementById("detail");
  box.replaceChildren();
  const head = htmlEl("div", { class: "detail-head" }, null, box);
  htmlEl("span", { class: "id" }, d.plant_id, head);
  const level = d.joint_level || 0;
  const chip = htmlEl("span", { class: "chip" }, null, head);
  htmlEl("span", { class: "key" }, null, chip).style.background = levelColor(level);
  chip.appendChild(document.createTextNode(level ? `综合 ${levelName(level)}` : "OOD"));
  htmlEl("span", { class: "chip" }, d.split, head);
  htmlEl("span", { class: "chip" }, `速度段 ${d.envelope.speed_range}`, head);
  const reason = level ? (level === 1 ? "全部判据均为 1 级" : `由「${LABELS[d.binding_group] || d.binding_group}」定级`)
    : `OOD：${d.ood_reasons.map((r) => LABELS[r] || r).join("、")}`;
  htmlEl("p", { class: "note" }, reason, box);
  const levels = htmlEl("div", { class: "levels" }, null, box);
  for (const [name, value] of Object.entries(d.component_levels)) {
    const c = htmlEl("span", { class: `level-chip ${value === 2 ? "l2" : value === 3 ? "l3" : ""}` }, null, levels);
    c.appendChild(document.createTextNode(`${LABELS[name] || name} `));
    htmlEl("b", {}, value ? `L${value}` : "超出", c);
  }
  const r = d.roll, lat = d.lateral, lon = d.longitudinal_loes, c = d.criteria, ph = d.longitudinal.phugoid;
  paramTable(box, [
    ["滚转通道（与原库同一类 P 通道）", [["T_R (s)", fmt(r.t_r)], ["螺旋根 λs (1/s)", fmt(r.lambda_s)], ["荷兰滚 ζd / ωd", `${fmt(r.zeta_d)} / ${fmt(r.omega_d)} rad/s`],
      ["零点 ζφ / ωφ/ωd（推导）", `${fmt(r.zeta_phi)} / ${fmt(r.r_omega)}`], ["1 s 灵敏度 (deg/N)", fmt(r.sensitivity_1s_deg_per_n)], ["滚转时延 τp (s)", fmt(r.tau_p)]]],
    ["横航向耦合（导数，稳定轴）", [["Yβ / L′β / N′β", `${fmt(lat.y_b)} / ${fmt(lat.l_b)} / ${fmt(lat.n_b)}`], ["L′p / L′r / N′p / N′r", `${fmt(lat.l_p)} / ${fmt(lat.l_r)} / ${fmt(lat.n_p)} / ${fmt(lat.n_r)}`],
      ["副翼偏航 η = N′a/L′a", fmt(d.eta)], ["上反比 −L′β/N′β", fmt(d.dihedral_ratio)], ["|φ/β|d、ψβ", `${fmt(c.dutch_roll_shape.phi_over_beta)}、${fmt(c.psi_beta_deg)}°`],
      ["侧滑偏离 Δβ/k", `${fmt(c.sideslip_excursion.delta_beta_over_k_deg)}°（限 ${fmt(c.sideslip_excursion.limit_level_1)} / ${fmt(c.sideslip_excursion.limit_level_2)}）`],
      ["每度侧滑：杆力 / 脚蹬 / 坡度", `${fmt(c.steady_sideslip.roll_force_n_per_deg)} N / ${fmt(c.steady_sideslip.pedal_force_n_per_deg)} N / ${fmt(c.steady_sideslip.bank_deg_per_deg)}°`]]],
    ["俯仰与长周期", [["V0 (m/s)、包线", `${fmt(lon.v0)}（${fmt(d.envelope.v_min)}–${fmt(d.envelope.v_max)}）`], ["ωsp / ζsp", `${fmt(lon.omega_sp)} rad/s / ${fmt(lon.zeta_sp)}`],
      ["CAP、Fs/n", `${fmt(d.loes_derived.cap)}、${fmt(lon.fs_per_g)} N/g`], ["τθ (s)", fmt(lon.tau_theta)],
      ["长周期", ph ? `ωp ${fmt(ph.omega_p)} rad/s，ζp ${fmt(ph.zeta_p)}` : c.long_period.form === "divergent" ? `发散，倍幅 ${fmt(c.long_period.time_to_double_s)} s` : "两个稳定实根"],
      ["惯量比 Iy/Ix、Iz/Ix、Ixz/Ix", `${fmt(d.inertia.iy)}、${fmt(d.inertia.iz)}、${fmt(d.inertia.ixz)}`]]],
  ]);
}

function renderSteps() {
  const s = state.steps; if (!s) return;
  const color = levelColor(state.detail.joint_level || 0);
  document.getElementById("steps-note").textContent =
    `${state.detail.plant_id}：小幅阶跃，保持 8 s——横向杆 ${fmt(s.inputs.stick_n)} N（1 s 内约 2° 坡度）、脚蹬 ${fmt(s.inputs.pedal_n)} N（荷兰滚近似下 0.5° 稳态侧滑）、俯仰杆 ${fmt(s.inputs.pitch_n)} N（0.5 g）。横向杆保持时滚转持续，坡度会逐渐变大；大幅机动见下一节。`;
  const box = document.getElementById("steps");
  box.replaceChildren();
  const panels = [
    ["横向杆 → p (deg/s)", s.stick.p], ["横向杆 → β (deg)", s.stick.beta], ["横向杆 → r (deg/s)", s.stick.r],
    ["脚蹬 → β (deg)", s.pedal.beta], ["脚蹬 → r (deg/s)", s.pedal.r], ["脚蹬 → p (deg/s)", s.pedal.p],
    ["俯仰杆 → n_z 增量 (g)", s.pitch.nz], ["俯仰杆 → q (deg/s)", s.pitch.q], ["俯仰杆 → 速度变化 (m/s)", s.pitch.dv]];
  for (const [title, values] of panels) {
    const card = htmlEl("div", { class: "card chart" }, null, box);
    lineChart(card, { title, t: s.t, series: [{ name: state.detail.plant_id, values, color }] });
  }
}

async function runManeuvers() {
  if (!state.selected) return;
  const button = document.getElementById("maneuver-button");
  button.disabled = true; button.textContent = "计算中…";
  try {
    state.maneuvers = await getJSON(`api/maneuvers?id=${encodeURIComponent(state.selected)}`);
    renderManeuvers();
  } finally { button.disabled = false; button.textContent = "计算选中对象的大机动"; }
}

function renderManeuvers() {
  const m = state.maneuvers; if (!m) return;
  const a = css("--series-1"), b = css("--series-2");
  const box = document.getElementById("maneuvers");
  box.replaceChildren();
  const panels = [
    ["压坡度后的高度变化 (m)", m.t_turn, [["不拉杆", m.no_pull.altitude, a], ["拉 1 g", m.pull_1g.altitude, b]]],
    ["压坡度后的坡度 φ (deg)", m.t_turn, [["不拉杆", m.no_pull.phi, a], ["拉 1 g", m.pull_1g.phi, b]]],
    ["360° 滚转：迎角变化 Δα (deg)", m.t_roll, [["有惯性耦合", m.with_coupling.alpha, a], ["无惯性耦合", m.without_coupling.alpha, b]]],
    ["360° 滚转：侧滑 β (deg)", m.t_roll, [["有惯性耦合", m.with_coupling.beta, a], ["无惯性耦合", m.without_coupling.beta, b]]],
  ];
  for (const [title, t, series] of panels) {
    const card = htmlEl("div", { class: "card chart" }, null, box);
    const lg = htmlEl("div", { class: "legend" }, null, card);
    legend(lg, series.map(([name, , color], i) => ({ label: name, color, dash: DASHES[i] })));
    const holder = htmlEl("div", {}, null, card);
    lineChart(holder, { title, t, series: series.map(([name, values, color], i) => ({ name, values, color, dash: DASHES[i] })) });
  }
  const warn = document.getElementById("maneuver-warn");
  warn.replaceChildren();
  const w = m.with_coupling, wo = m.without_coupling;
  const [bank, tReq] = m.inputs.roll_requirement;
  const ph = m.phillips;
  const text = `滚转杆 ${fmt(m.inputs.roll_stick_n)} N（按本对象速度段的 1 级滚转性能 ${bank}°/${tReq} s 标定），` +
    `转满 360° 用时 ${fmt(w.released_s)} s（无耦合 ${fmt(wo.released_s)} s）后松杆；最大滚转角速度 ${fmt(w.max_roll_rate_deg_s)}°/s，` +
    `Phillips 临界滚转角速度：俯仰 ${fmt(ph.p_pitch_deg_s)}°/s、偏航 ${fmt(ph.p_yaw_deg_s)}°/s。` +
    `最大 |Δα| ${fmt(w.max_abs_d_alpha_deg)}°（无耦合 ${fmt(wo.max_abs_d_alpha_deg)}°），最大 |β| ${fmt(w.max_abs_beta_deg)}°。`;
  htmlEl("p", { class: "note" }, text, warn);
  const left = [["压坡度不拉杆", m.no_pull], ["压坡度拉 1 g", m.pull_1g], ["360° 滚转（有耦合）", w], ["360° 滚转（无耦合）", wo]]
    .filter(([, run]) => run.left_linear_range_s !== null && run.left_linear_range_s !== undefined);
  if (left.length)
    htmlEl("div", { class: "warn" }, `${left.map(([name, run]) => `${name}在 ${fmt(run.left_linear_range_s)} s`).join("，")}` +
      " 时迎角变化或侧滑超过 15°，已超出线性气动导数的可信范围，曲线到此截止。", warn);
  else if (Math.max(w.max_abs_d_alpha_deg, wo.max_abs_d_alpha_deg) > 20)
    htmlEl("div", { class: "warn" }, "迎角偏离超过 20°，接近线性气动导数的适用边界；曲线只示意耦合趋势。", warn);
}

// ---------------------------------------------------------------- boot
function renderAll() {
  renderTiles(); renderBinding(); renderDistributions(); renderFilters(); renderScatter();
  renderDetail(); renderSteps(); renderManeuvers();
}

(async function main() {
  try {
    const [data, health] = await Promise.all([getJSON("api/summary"), getJSON("healthz")]);
    state.data = data;
    state.objects = data.rows.map((row) => Object.fromEntries(data.fields.map((f, i) => [f, row[i]])));
    document.getElementById("revision").textContent = health.revision;
    document.getElementById("maneuver-button").addEventListener("click", runManeuvers);
    renderAll();
    const hash = new URLSearchParams(location.hash.slice(1));
    const wanted = hash.get("id");
    await select(state.objects.some((o) => o.id === wanted) ? wanted : "train_core-0000");
    if (hash.has("maneuvers")) await runManeuvers();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderAll);
  } catch (error) {
    document.querySelector("main").prepend(htmlEl("p", { class: "warn" }, `加载失败：${error.message}`));
  }
})();
