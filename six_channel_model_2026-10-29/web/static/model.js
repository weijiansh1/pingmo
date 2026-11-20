"use strict";
// Formula page: the six-channel chain from LOES to dynamics, with the selected object's numbers.
// TeX lives in String.raw literals; numeric boxes are rebuilt whenever the object changes.

const R = String.raw;
const page = { model: null, summary: null, id: null, bode: new Set(["q_Fs", "p_Fas", "beta_Frp"]) };

// ---------------------------------------------------------------- TeX helpers
function tex(target, source, display = false) {
  katex.render(source, target, { displayMode: display, throwOnError: false, strict: "ignore" });
  return target;
}
function rich(parent, tag, text, attrs = {}) {
  // text with $inline math$; plain parts go in as text nodes
  const node = htmlEl(tag, attrs, null, parent);
  text.split(/(\$[^$]+\$)/g).forEach((part) => {
    if (part.startsWith("$") && part.endsWith("$") && part.length > 2) tex(htmlEl("span", {}, null, node), part.slice(1, -1));
    else if (part) node.appendChild(document.createTextNode(part));
  });
  return node;
}
function eq(parent, source) { return tex(htmlEl("div", { class: "eq" }, null, parent), source, true); }
function tn(x, digits = 4) {
  if (x === null || x === undefined || !Number.isFinite(x)) return R`\text{—}`;
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a < 1e-3 || a >= 1e5) {
    const e = Math.floor(Math.log10(a));
    return `${+(x / 10 ** e).toPrecision(3)}\\times 10^{${e}}`;
  }
  return String(+x.toPrecision(digits));
}
function polyTeX(c, v = "s") {
  const n = c.length - 1, max = Math.max(...c.map(Math.abs));
  let out = "";
  c.forEach((k, i) => {
    const p = n - i;
    if (Math.abs(k) <= 1e-12 * max) return;
    const mag = Math.abs(k);
    const coef = Math.abs(mag - 1) < 1e-9 && p > 0 ? "" : tn(mag);
    const term = coef + (p === 0 ? "" : (coef ? R`\,` : "") + (p === 1 ? v : `${v}^{${p}}`));
    out += out ? (k < 0 ? ` - ${term}` : ` + ${term}`) : (k < 0 ? `-${term}` : term);
  });
  return out || "0";
}
function factorsTeX(roots) {
  let zeros = 0, out = "";
  for (const [re, im] of roots) {
    const scale = Math.max(1, Math.hypot(re, im));
    if (Math.hypot(re, im) < 1e-9 * scale) { zeros++; continue; }
    if (Math.abs(im) > 1e-9 * scale) {
      if (im < 0) continue;
      const b = -2 * re, c = re * re + im * im;
      out += `(s^{2} ${b < 0 ? "-" : "+"} ${tn(Math.abs(b))}\\,s + ${tn(c)})`;
    } else {
      out += `(s ${re > 0 ? "-" : "+"} ${tn(Math.abs(re))})`;
    }
  }
  return (zeros ? (zeros === 1 ? "s" : `s^{${zeros}}`) : "") + out;
}
function tfTeX(ch, out, inp) {
  const gain = ch.num[0] / ch.den[0];
  const num = factorsTeX(ch.zeros) || "1";
  const delay = ch.delay > 0 ? R`\,e^{-${tn(ch.delay, 3)}\,s}` : "";
  return R`\frac{${out}}{${inp}} = ${tn(gain)}\,\dfrac{${num}}{${factorsTeX(ch.poles)}}${delay}`;
}
function matTeX(rows) {
  return R`\begin{bmatrix}` + rows.map((r) => r.map((x) => (Math.abs(x) < 1e-15 ? "0" : tn(x))).join(" & ")).join(R` \\ `) + R`\end{bmatrix}`;
}
function kv(parent, pairs) {
  const box = htmlEl("div", { class: "kv" }, null, parent);
  for (const [name, value] of pairs) {
    const row = htmlEl("div", {}, null, box);
    rich(row, "span", name);
    htmlEl("span", {}, value, row);
  }
}

// ---------------------------------------------------------------- charts specific to this page
function bodeChart(container, { title, freqs, series, field }) {
  container.replaceChildren();
  const W = 520, H = 230, box = { l: 50, r: W - 10, t: 30, b: H - 26 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, container);
  svgEl("text", { x: 4, y: 14, class: "title" }, svg).textContent = title;
  const values = series.flatMap((s) => s.values).filter(Number.isFinite);
  if (!values.length) return;
  let lo = Math.min(...values), hi = Math.max(...values);
  if (field === "db") { lo = Math.max(lo, hi - 160); }   // lightly damped phugoids make tall peaks
  const yt = niceTicks(lo, hi, 5);
  const x = scale([freqs[0], freqs[freqs.length - 1]], [box.l, box.r], true);
  const y = scale([Math.min(lo, yt[0]), Math.max(hi, yt[yt.length - 1])], [box.b, box.t]);
  axes(svg, x, y, logTicks(freqs[0], freqs[freqs.length - 1]).filter((v) => [0.01, 0.1, 1, 10, 100].includes(v) || v === 0.5 || v === 5), yt, box);
  svgEl("text", { x: box.r, y: H - 2, "text-anchor": "end" }, svg).textContent = "ω (rad/s)";
  const clipLo = y.invert ? Math.min(lo, yt[0]) : lo;
  series.forEach((s, i) => {
    let d = "";
    s.values.forEach((v, k) => {
      if (!Number.isFinite(v) || v < clipLo) { return; }
      d += `${d && Number.isFinite(s.values[k - 1]) && s.values[k - 1] >= clipLo ? "L" : "M"}${x(freqs[k]).toFixed(1)},${y(v).toFixed(1)}`;
    });
    svgEl("path", { d, fill: "none", stroke: s.color, "stroke-width": 1.8, "stroke-dasharray": s.dash || "" }, svg);
  });
  const hair = svgEl("line", { y1: box.t, y2: box.b, stroke: css("--axis"), visibility: "hidden" }, svg);
  const hit = svgEl("rect", { x: box.l, y: box.t, width: box.r - box.l, height: box.b - box.t, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const w = x.invert(svgPoint(svg, event).x);
    let k = 0, best = Infinity;
    freqs.forEach((f, i) => { const d = Math.abs(Math.log(f / w)); if (d < best) { best = d; k = i; } });
    hair.setAttribute("x1", x(freqs[k])); hair.setAttribute("x2", x(freqs[k])); hair.setAttribute("visibility", "visible");
    showTooltip(event, `ω = ${fmt(freqs[k])} rad/s`, series.map((s) => ({ name: s.name, value: `${fmt(s.values[k])} ${field === "db" ? "dB" : "°"}`, color: s.color, dash: s.dash })));
  });
  hit.addEventListener("pointerleave", () => { hair.setAttribute("visibility", "hidden"); hideTooltip(); });
}

function poleZeroChart(container, { title, sets }) {
  container.replaceChildren();
  const W = 520, H = 300, box = { l: 50, r: W - 12, t: 30, b: H - 28 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": title }, container);
  svgEl("text", { x: 4, y: 14, class: "title" }, svg).textContent = title;
  const poles = sets.filter((s) => s.kind === "pole").flatMap((s) => s.roots);
  const reach = Math.max(0.5, ...poles.map(([re, im]) => Math.hypot(re, im))) * 1.25;
  const x = scale([-reach, Math.max(0.25 * reach, ...poles.map(([re]) => re * 1.3))], [box.l, box.r]);
  const y = scale([-reach * 0.75, reach * 0.75], [box.b, box.t]);
  const [x0, x1] = [-reach, Math.max(0.25 * reach, ...poles.map(([re]) => re * 1.3))];
  axes(svg, x, y, niceTicks(x0, x1, 6), niceTicks(-reach * 0.75, reach * 0.75, 5), box);
  svgEl("line", { x1: x(0), x2: x(0), y1: box.t, y2: box.b, class: "axis-line" }, svg);
  svgEl("text", { x: box.r, y: H - 4, "text-anchor": "end" }, svg).textContent = "实部 (1/s)";
  svgEl("text", { x: box.l + 4, y: box.t + 10 }, svg).textContent = "虚部";
  const marks = [];
  sets.forEach((set) => {
    for (const [re, im] of set.roots) {
      const outside = re < x0 || re > x1 || Math.abs(im) > reach * 0.75;
      const px = Math.max(box.l + 4, Math.min(box.r - 4, x(re))), py = Math.max(box.t + 4, Math.min(box.b - 4, y(im)));
      if (set.kind === "pole") {
        svgEl("path", { d: `M${px - 5},${py - 5}L${px + 5},${py + 5}M${px - 5},${py + 5}L${px + 5},${py - 5}`, stroke: set.color, "stroke-width": 2 }, svg);
      } else {
        svgEl("circle", { cx: px, cy: py, r: 5, fill: "none", stroke: set.color, "stroke-width": 1.8, "stroke-dasharray": outside ? "2 2" : "" }, svg);
      }
      marks.push([px, py, set, re, im, outside]);
    }
  });
  const hit = svgEl("rect", { x: box.l, y: box.t, width: box.r - box.l, height: box.b - box.t, fill: "transparent" }, svg);
  hit.addEventListener("pointermove", (event) => {
    const p = svgPoint(svg, event);
    let best = null, bd = 24 * 24;
    for (const m of marks) { const d = (m[0] - p.x) ** 2 + (m[1] - p.y) ** 2; if (d < bd) { bd = d; best = m; } }
    if (!best) { hideTooltip(); return; }
    const [, , set, re, im, outside] = best;
    const rows = [{ name: "实部", value: fmt(re) }, { name: "虚部", value: fmt(im) }];
    if (Math.abs(im) > 1e-9) { const w = Math.hypot(re, im); rows.push({ name: "ω", value: fmt(w) }, { name: "ζ", value: fmt(-re / w) }); }
    else if (re !== 0) rows.push({ name: "时间常数 (s)", value: fmt(-1 / re) });
    if (outside) rows.push({ name: "（超出画幅，画在边缘）", value: "" });
    showTooltip(event, set.name, rows);
  });
  hit.addEventListener("pointerleave", hideTooltip);
}

// ---------------------------------------------------------------- the document
const DOC = [];
const H2 = (id, title) => DOC.push(["h2", id, title]);
const H3 = (id, title) => DOC.push(["h3", id, title]);
const P = (text) => DOC.push(["p", text]);
const EQ = (source) => DOC.push(["eq", source]);
const UL = (items) => DOC.push(["ul", items]);
const CITE = (text) => DOC.push(["cite", text]);
const BOX = (title, render) => DOC.push(["box", title, render]);
const TABLE = (head, rows) => DOC.push(["table", head, rows]);

H2("overview", "0 总览：一条链路");
DOC.push(["flow"]);
P(R`输入为俯仰杆力 $F_s$、横向杆力 $F_{as}$、脚蹬力 $F_{rp}$（单位 N，拉杆、右压杆、右蹬为正），另有油门 $\delta_T$（推力增量/重量）与直接法向力 $\delta_n$（法向力/重量）。动力学状态是稳定轴小扰动量 $[\,u,\ \alpha,\ q,\ \beta,\ p_s,\ r_s\,]$，运动学给出姿态与位置，合成 xmind 的状态向量：`);
EQ(R`\mathbf{x} = [\,V,\ \alpha,\ q,\ \theta,\ \beta,\ p,\ r,\ \phi,\ \psi,\ X,\ Y,\ Z\,]^{\mathsf T}`);
P(R`坐标约定：机体轴 $x$ 向前、$y$ 向右、$z$ 向下；地面系为北-东-地。稳定轴是绕机体 $y$ 轴转过配平迎角 $\alpha$ 的机体轴，横航向导数 $L'_{(\cdot)},N'_{(\cdot)}$ 均为稳定轴下计入 $I_{xz}$ 的带撇导数。$g/V_0$ 记为 $g'$。`);

H2("channels", "1 六个通道的低阶等效系统（品模）");
P(R`GJB 2874-97 用低阶等效系统（LOES）描述各通道对驾驶员操纵力的响应，并把品质要求写在等效参数上（正文 4.9：“承制方应给出与真实飞机响应最接近的等效经典系统”）。xmind 的六个通道是 V、α、q（纵向）与 p、β、r（横航向）。`);
H3("ch-pitch", "1.1 俯仰：q、α、nz");
EQ(R`\frac{q}{F_s} = \frac{K_\theta\,(s + 1/T_{\theta 2})\,e^{-\tau_\theta s}}{s^{2} + 2\zeta_{sp}\omega_{sp}\,s + \omega_{sp}^{2}}, \qquad \frac{n_z}{F_s} = \frac{K_{n_z}\,e^{-\tau_\theta s}}{s^{2} + 2\zeta_{sp}\omega_{sp}\,s + \omega_{sp}^{2}}`);
CITE(R`GJB A5.1.1.1（PDF p.76–77）。扫描页写作 $\dot\theta/F_s$，正文为“俯仰速率”；xmind 截图丢了上方的点。两式应同时拟配（表 A4 注）。`);
P(R`翼平小扰动时 $\dot\theta = q\cos\phi - r\sin\phi = q$，故 $\theta = q/s$。航迹对姿态是一阶滞后、迎角是两者之差（xmind）：`);
EQ(R`\frac{\gamma}{\theta} = \frac{1}{T_{\theta2}s + 1},\qquad \alpha = \theta - \gamma \;\Rightarrow\; \frac{\alpha}{F_s} = \frac{\theta}{F_s}\cdot\frac{T_{\theta2}s}{T_{\theta2}s+1} = \frac{K_\theta\,e^{-\tau_\theta s}}{s^{2} + 2\zeta_{sp}\omega_{sp}s + \omega_{sp}^{2}}`);
P(R`法向过载 $n_z = (V_0/g)\,\dot\gamma$，于是增益之间不是独立参数：`);
EQ(R`\frac{n}{\alpha} = \frac{V_0}{g\,T_{\theta2}},\qquad K_{n_z} = K_\theta\,\frac{n}{\alpha},\qquad \mathrm{CAP} = \frac{\omega_{sp}^{2}}{n/\alpha},\qquad \left.\frac{n_z}{F_s}\right|_{s\to0} = \frac{1}{F_s/n}\;\Rightarrow\; K_\theta = \frac{\mathrm{CAP}}{F_s/n}`);
P(R`采样量为 $\omega_{sp},\ \zeta_{sp},\ n/\alpha,\ V_0,\ F_s/n,\ \tau_\theta$；$T_{\theta2}$、CAP、$K_\theta$、$K_{n_z}$ 由上式导出。`);
H3("ch-speed", "1.2 速度 V：短周期假设与长周期");
P(R`短周期近似假设速度不变，$\Delta V = 0$。速度响应来自长周期，对应 MIL-STD-1797A 4.2.1.1 的三自由度等效式（GJB 表 A5 给出 $\zeta_p$ 要求）：`);
EQ(R`\frac{\theta}{F_s} = \frac{K_\theta\,(s + 1/T_{\theta1})(s + 1/T_{\theta2})\,e^{-\tau_\theta s}}{(s^{2}+2\zeta_p\omega_p s+\omega_p^{2})(s^{2}+2\zeta_{sp}\omega_{sp}s+\omega_{sp}^{2})}`);
CITE("MIL-STD-1797A PDF p.297；GJB 表 A5，PDF p.77。本项目中 ωp、ζp、1/Tθ1 由第 2 节的物理导数得出，不单独采样。");
H3("ch-roll", "1.3 滚转 p");
EQ(R`\frac{p}{F_{as}} = \frac{L_{F_a}\,s\,(s^{2}+2\zeta_\phi\omega_\phi s+\omega_\phi^{2})\,e^{-\tau_p s}}{(s-\lambda_s)(s+1/T_R)(s^{2}+2\zeta_d\omega_d s+\omega_d^{2})},\qquad \lambda_s = -\frac{1}{T_s}`);
CITE(R`GJB A5.4.1.1（PDF p.215，表 A18）。GJB p.215 印作 $P/F_{as}$ 而分子没有 $s$（实为 $\phi/F_{as}$ 的形式），p.240 的分子为 $s^{2}$；本项目取 MIL-STD-1797A PDF p.550 的 $p/F_{as}$（分子一个 $s$，即 $p = s\phi$，与 p.522 的 $\phi/F_{as}$ 一致），与原滚转库相同。$L_{F_a}$ 由 1 s 滚转灵敏度 $S_{1s} = |\phi(1\,\mathrm{s})|/F$ 标定（表 A31，PDF p.276）。`);
H3("ch-pedal", "1.4 脚蹬 β、r（xmind 的解耦写法）");
EQ(R`\frac{\beta}{F_{rp}} = \frac{K_\beta\,e^{-\tau_\beta s}}{s^{2}+2\zeta_d\omega_d s+\omega_d^{2}},\qquad \frac{r}{\delta_r} = \frac{A_r\,(s^{2}+2\zeta_r\omega_r s+\omega_r^{2})}{(s+1/T_s)(s^{2}+2\zeta_d\omega_d s+\omega_d^{2})},\qquad \frac{r}{F_{rp}} = \frac{r}{\delta_r}\cdot\frac{\delta_r}{F_{rp}}`);
P(R`三条横航向通道共用同一组模态（$T_R$、$\lambda_s$、$\zeta_d$、$\omega_d$）。这说明它们不是相互独立的传递函数，而是同一个横航向动力学系统的不同输出；第 3 节由此建立耦合模型，$\beta/F_{as}$、$r/F_{as}$、$p/F_{rp}$ 等交叉通道随之确定，xmind 的写法是其中的近似。`);
BOX("代入当前对象：低阶等效系统", renderLoesBox);

H2("longitudinal", "2 从传递函数到动力学：纵向");
H3("lon-sp", "2.1 短周期：两状态与 LOES 一一对应");
EQ(R`\begin{aligned} \dot\alpha &= Z_\alpha\,\alpha + q \\ \dot q &= M_\alpha\,\alpha + M_q\,q + K_\theta\,F_s(t-\tau_\theta) \end{aligned}`);
P(R`特征多项式 $(s - Z_\alpha)(s - M_q) - M_\alpha$ 与 1.1 节分母逐项相等，零点 $s = Z_\alpha$ 与 $-1/T_{\theta2}$ 相等，解得：`);
EQ(R`Z_\alpha = -\frac{1}{T_{\theta2}},\qquad M_q = \frac{1}{T_{\theta2}} - 2\zeta_{sp}\omega_{sp},\qquad M_\alpha = -\omega_{sp}^{2} - \frac{M_q}{T_{\theta2}},\qquad n_z = -\frac{V_0}{g}Z_\alpha\,\alpha = \frac{n}{\alpha}\,\alpha`);
P(R`此时 $q/F_s$、$\alpha/F_s$、$n_z/F_s$ 与 GJB 三式完全相同（测试误差 < 1e-6；这一点只对两状态成立，四状态见 2.2 节），而 $q$ 成为显式状态，惯性耦合项可以直接加在 $\dot q$ 上（xmind “组合轴”）。`);
H3("lon-full", "2.2 加入速度与油门：四状态");
EQ(R`\begin{aligned} \dot u &= X_u\,u + X_\alpha\,\alpha - g\,\theta + g\,\delta_T \\ \dot\alpha &= Z_u\,u + Z_\alpha\,\alpha + q \\ \dot q &= M_u\,u + M_\alpha\,\alpha + M_q\,q + K_\theta\,F_s(t-\tau_\theta) \\ \dot\theta &= q \end{aligned}`);
P(R`速度导数按物理关系给定：升力正比 $V^{2}$ 给出 $Z_u = -2g/V_0^{2}$；推力不随速度变化时 $X_u = -2g/\bigl(V_0\,(L/D)\bigr)$；稳定轴下 $X_\alpha = g - (\partial D/\partial\alpha)/m$；$M_u$ 取亚音速数据。有效 $L/D$、阻力斜率、$M_u$ 按 CR-2144 中 47 个亚音速飞行状态的 5–95 百分位抽取。`);
EQ(R`\Delta_{\mathrm{lon}}(s) = s^{4} + b_3 s^{3} + b_2 s^{2} + b_1 s + b_0,\quad \begin{aligned} b_3 &= -(M_q + X_u + Z_\alpha) \\ b_2 &= -M_\alpha + M_q X_u + M_q Z_\alpha - X_\alpha Z_u + X_u Z_\alpha \\ b_1 &= M_\alpha X_u + M_q X_\alpha Z_u - M_q X_u Z_\alpha - M_u X_\alpha + M_u\,g \\ b_0 &= g\,(M_\alpha Z_u - M_u Z_\alpha) \end{aligned}`);
P(R`要求四状态模型保持 GJB 短周期不变：$\Delta_{\mathrm{lon}} = (s^{2} + p_1 s + p_0)(s^{2} + 2\zeta_{sp}\omega_{sp}s + \omega_{sp}^{2})$，且 $\theta/F_s$ 仍有零点 $-1/T_{\theta2}$。记 $z_2 = 1/T_{\theta2}$，先由零点条件解出 $Z_\alpha$，其余两式对长周期系数 $(p_0, p_1)$ 是线性的：`);
EQ(R`Z_\alpha = \frac{X_\alpha Z_u - z_2^{2} - X_u z_2}{z_2 + X_u},\qquad M_q = -p_1 - 2\zeta_{sp}\omega_{sp} - X_u - Z_\alpha,\qquad M_\alpha = \frac{p_0\,\omega_{sp}^{2}/g + M_u Z_\alpha}{Z_u}`);
P(R`代入 $b_2$、$b_1$ 两式得到关于 $(p_0,p_1)$ 的 2×2 线性方程，唯一解出长周期；$\omega_p,\zeta_p$（或两个实根）与 $1/T_{\theta1}$ 都是物理结果。经典近似 $\omega_p \approx \sqrt2\,g/V_0$、$\zeta_p \approx \tfrac{1}{\sqrt2}\,D/L$（Lanchester）可作校核：v4 库 $\omega_p$ 与其比值的中位数 0.80，真实飞机 0.87。`);
P(R`四状态保持了短周期极点和零点 $-1/T_{\theta2}$，但 $Z_\alpha$ 不再等于 $-1/T_{\theta2}$，仿真中的 $n_z/\alpha = -V_0Z_\alpha/g$ 与定级用的 $n/\alpha$ 相差：v4 中位数 0.4%、95 百分位 3.9%、最大 22%。GJB 表 A4 还要求 $1/T_{\theta1} > 0$（不分等级），v4 把不满足的对象归入 OOD。`);
H3("lon-tf", "2.3 纵向传递函数");
EQ(R`\begin{aligned} \frac{\theta}{F_s} &= \frac{K_\theta\,\bigl(s^{2} - (X_u + Z_\alpha)s + X_u Z_\alpha - X_\alpha Z_u\bigr)}{\Delta_{\mathrm{lon}}}\,e^{-\tau_\theta s}, & \frac{q}{F_s} &= s\,\frac{\theta}{F_s} \\ \frac{\alpha}{F_s} &= \frac{K_\theta\,(s^{2} - X_u s - g Z_u)}{\Delta_{\mathrm{lon}}}\,e^{-\tau_\theta s}, & \frac{u}{F_s} &= \frac{K_\theta\,\bigl((X_\alpha - g)\,s + g Z_\alpha\bigr)}{\Delta_{\mathrm{lon}}}\,e^{-\tau_\theta s} \\ \frac{u}{\delta_T} &= \frac{g\,s\,\bigl(s^{2} - (Z_\alpha + M_q)s + M_q Z_\alpha - M_\alpha\bigr)}{\Delta_{\mathrm{lon}}\,(\tau_e s + 1)}, & \frac{\theta}{\delta_T} &= \frac{g\,(M_u s + M_\alpha Z_u - M_u Z_\alpha)}{\Delta_{\mathrm{lon}}\,(\tau_e s + 1)} \end{aligned}`);
P(R`$n_z = -\tfrac{V_0}{g}(Z_u u + Z_\alpha \alpha)$。V 通道就是 $V = V_0 + u$，由油门 $\delta_T$ 控制；发动机按一阶滞后 $\tau_e = 1\,\mathrm{s}$（假设）。$\theta/F_s$ 的分子二次式恰为 $(s + 1/T_{\theta1})(s + 1/T_{\theta2})$，即 1.2 节的三自由度 LOES。`);
BOX("代入当前对象：纵向导数、矩阵与传递函数", renderLongitudinalBox);

H2("lateral", "3 从传递函数到动力学：横航向");
H3("lat-eq", "3.1 稳定轴四状态");
EQ(R`\begin{aligned} \dot\beta &= Y_\beta\,\beta - r + g'\,\phi + Y_{rp}\,F_{rp}(t-\tau_{rp}) \\ \dot p &= L'_\beta\,\beta + L'_p\,p + L'_r\,r + L_a\,F_{as}(t-\tau_p) + L'_{rp}\,F_{rp}(t-\tau_{rp}) \\ \dot r &= N'_\beta\,\beta + N'_p\,p + N'_r\,r + N_a\,F_{as}(t-\tau_p) + N'_{rp}\,F_{rp}(t-\tau_{rp}) \\ \dot\phi &= p \end{aligned}`);
CITE("Etkin & Reid；Nelson 第 5 章；Davidson & Andrisani NASA TM-110306（PDF p.4–7）同取 Y_p = Y_r = 0。");
H3("lat-poly", "3.2 特征多项式与滚转零点");
EQ(R`\Delta_{\mathrm{lat}} = s^{4} + a_3 s^{3} + a_2 s^{2} + a_1 s + a_0,\quad \begin{aligned} a_3 &= -(L'_p + N'_r + Y_\beta) \\ a_2 &= L'_p N'_r - L'_r N'_p + N'_\beta + Y_\beta (L'_p + N'_r) \\ a_1 &= L'_\beta N'_p - g' L'_\beta - L'_p N'_\beta - Y_\beta (L'_p N'_r - L'_r N'_p) \\ a_0 &= g'\,(L'_\beta N'_r - L'_r N'_\beta) \end{aligned}`);
EQ(R`\frac{p}{F_{as}} = \frac{L_a\,s\,\bigl[s^{2} + (L'_r\eta - N'_r - Y_\beta)\,s + N'_\beta + N'_r Y_\beta - \eta\,(L'_\beta + L'_r Y_\beta)\bigr]}{\Delta_{\mathrm{lat}}}\,e^{-\tau_p s},\qquad \eta = \frac{N_a}{L_a}`);
P(R`与 1.3 节对照：$2\zeta_\phi\omega_\phi = L'_r\eta - N'_r - Y_\beta$，$\omega_\phi^{2} = N'_\beta + N'_r Y_\beta - \eta(L'_\beta + L'_r Y_\beta)$，$\Delta_{\mathrm{lat}} = (s-\lambda_s)(s + 1/T_R)(s^{2} + 2\zeta_d\omega_d s + \omega_d^{2})$。`);
H3("lat-inv", "3.3 由品质模态解出导数（v3、v4）");
P(R`四个极点按 GJB 等级采样；形状量 $Y_\beta$、$c_1 = N'_p/L'_p$、$c_2 = L'_r/N'_r$ 取自真实飞机。记 $S = -a_3 - Y_\beta$，则所有导数都是 $L'_p$ 的函数：`);
EQ(R`N'_r = S - L'_p,\quad N'_p = c_1 L'_p,\quad L'_r = c_2 N'_r,\quad N'_\beta = a_2 - L'_p N'_r - Y_\beta S + L'_r N'_p,\quad L'_\beta = \frac{a_1 + L'_p N'_\beta + Y_\beta\,(L'_p N'_r - L'_r N'_p)}{N'_p - g'}`);
EQ(R`g'\,\bigl(L'_\beta N'_r - L'_r N'_\beta\bigr) = a_0 \;\Longrightarrow\; \text{关于 } L'_p \text{ 的四次方程}`);
P(R`取落在真实飞机范围内的实根；再取副翼偏航 $\eta$，零点 $\zeta_\phi,\omega_\phi$ 由 3.2 节两式算出，$L_a$ 按 $S_{1s}$ 标定。滚转零点因此是推导量，而原滚转库是独立采样的。`);
H3("lat-cross", "3.4 交叉通道（同一组极点）");
EQ(R`\begin{aligned} N_{\beta}^{F_{as}} &= -N_a s^{2} + \bigl(L_a g' - L_a N'_p + L'_p N_a\bigr)s + g'\,(L'_r N_a - L_a N'_r) \\ N_{r}^{F_{as}} &= N_a s^{3} + \bigl(L_a N'_p - L'_p N_a - N_a Y_\beta\bigr)s^{2} + Y_\beta\,(L'_p N_a - L_a N'_p)\,s + g'\,(L_a N'_\beta - L'_\beta N_a) \\ N_{\beta}^{F_{rp}} &= Y_{rp}s^{3} - \bigl(L'_p Y_{rp} + N'_{rp} + N'_r Y_{rp}\bigr)s^{2} + \bigl(g' L'_{rp} - L'_{rp}N'_p + L'_p N'_{rp} + L'_p N'_r Y_{rp} - L'_r N'_p Y_{rp}\bigr)s + g'\,(L'_r N'_{rp} - L'_{rp} N'_r) \\ N_{p}^{F_{rp}} &= s\Bigl[L'_{rp}s^{2} + \bigl(L'_\beta Y_{rp} - L'_{rp}N'_r - L'_{rp}Y_\beta + L'_r N'_{rp}\bigr)s + L'_{rp}N'_\beta - L'_\beta N'_{rp} + Y_\beta(L'_{rp}N'_r - L'_r N'_{rp}) + Y_{rp}(L'_r N'_\beta - L'_\beta N'_r)\Bigr] \\ N_{r}^{F_{rp}} &= N'_{rp}s^{3} + \bigl(L'_{rp}N'_p - L'_p N'_{rp} + N'_\beta Y_{rp} - N'_{rp}Y_\beta\bigr)s^{2} + \bigl(Y_{rp}(L'_\beta N'_p - L'_p N'_\beta) + Y_\beta(L'_p N'_{rp} - L'_{rp}N'_p)\bigr)s + g'\,(L'_{rp}N'_\beta - L'_\beta N'_{rp}) \end{aligned}`);
P(R`各通道 $y/u = N_y^{u}/\Delta_{\mathrm{lat}}$，乘对应输入的延迟；$\phi/F_{as} = (p/F_{as})/s$。这些分子由 sympy 从 3.1 节方程逐项推出，代码中按状态空间直接计算。`);
H3("lat-phys", "3.5 物理关系");
UL([
  R`螺旋根决定上反：$a_0 = g'(L'_\beta N'_r - L'_r N'_\beta) = -\lambda_s\,\omega_d^{2}/T_R$。螺旋稳定要求 $L'_\beta N'_r > L'_r N'_\beta$（$N'_r<0$），滚转越快需要的上反越强；螺旋发散只说明上反相对 $L'_r N'_\beta/|N'_r|$ 不足，不必是下反（v4 库 1,665 个螺旋发散对象中 $L'_\beta > 0$ 的只有 83 个）。`,
  R`滚转零点与副翼偏航：$\omega_\phi^{2}/\omega_d^{2} \approx 1 - \eta\,L'_\beta/N'_\beta \approx 1 + \eta\,|\phi/\beta|_d$（Davidson & Andrisani 式 3.22，假定荷兰滚中 $\angle\phi/\beta \approx 0$ 且正上反；在 v4 库上第一式相对误差中位数 4%、95 百分位 20%）。正上反时 $\omega_\phi < \omega_d$ 对应不利偏航。`,
  R`脚蹬：$N'_{rp}$ 由侧滑梯度 $S_\beta$ 给定（荷兰滚近似下 $\beta_{ss}/F_{rp} = -N'_{rp}/\omega_d^{2}$）；$L'_{rp} = -f\,(-L'_\beta/N'_\beta)\,N'_{rp}$、$Y_{rp} = -Q\,(|Y_\beta|/N'_\beta)\,N'_{rp}$，$f$、$Q$ 按真实飞机分位数抽取，保持“垂尾同时决定方向舵力矩与侧滑导数”的相关性；下反（$L'_\beta \ge 0$）对象的 $L'_{rp}/N'_{rp}$ 改在真实 5–95 百分位 $[-4.22, -0.031]$ 内均匀抽取。`,
  R`这些导数是增稳后飞机的等效导数，不是裸机导数：真实性只约束 7 个归一化比值，量纲值偏大。v4 中 $N'_r$ 中位数 −2.59（真实裸机 −0.24，87% 的对象超出真实范围），$L'_r$ 中位数 3.6（v3 为 4.6；真实最大 7.3，22% 超出），四状态 $M_q$ 中位数 −4.2（真实 $M_q+M_{\dot\alpha}$ 为 −0.69）。原因是 GJB 1 级要求的荷兰滚、短周期阻尼比裸机高得多，相当于含偏航、俯仰阻尼器。v4 要求 $N'_r < 0$、$M_q < 0$。`,
]);
H3("lat-xmind", "3.6 与 xmind 解耦式的关系");
P(R`若 $Y_{rp}\approx 0$，且 $N_\beta^{F_{rp}}$ 中的实零点与滚转、螺旋两极点近似对消，则 $\beta/F_{rp} \approx -N'_{rp}/(s^{2}+2\zeta_d\omega_d s+\omega_d^{2})$，即 xmind 的 $K_\beta = -A_rG_r$ 形式；$r/F_{rp}$ 中的实零点与滚转模态近似对消，剩下 xmind 的三阶式。Navion 数据中该实零点为 −8.64、滚转模态为 −8.43。v2 中两者的阶跃响应对照见 figures/cross_channels_v2.png。`);
BOX("代入当前对象：横航向导数、矩阵、全部通道与模态", renderLateralBox);
BOX("代入当前对象：零极点与频率响应", renderFrequencyBox);

H2("full", "4 全量动力学：大机动、惯性耦合与阵风");
H3("full-force", "4.1 风轴力方程");
P(R`小扰动部分用第 2、3 节的线性导数；重力与运动学项保留非线性形式（Stevens & Lewis 风轴力方程）。$\mathbf g_w = C_{wb}(\alpha,\beta)\,C_{bn}^{\mathsf T}[0,0,g]^{\mathsf T}$ 为风轴重力：`);
EQ(R`\begin{aligned} \dot V &= X_u\,\Delta u + (X_\alpha - g)\,\Delta\alpha + g_{w,x} + g\,T \\ \dot\alpha &= q - \tan\beta\;p_w + Z_u\,\Delta u + Z_\alpha\,\Delta\alpha + \frac{g_{w,z} - g}{V\cos\beta} - \frac{g}{V}\,\delta_n \\ \dot\beta &= -r_w + Y_\beta\,\beta + Y_{rp}F_{rp} + \frac{g_{w,y}}{V} \end{aligned},\qquad \begin{aligned} p_w &= p_s\cos\Delta\alpha + r_s\sin\Delta\alpha \\ r_w &= -p_s\sin\Delta\alpha + r_s\cos\Delta\alpha \end{aligned}`);
P(R`$p_w, r_w$ 是绕当前速度方向的角速度（4.3 节）。配平时 $\mathbf g_w = [0,0,g]$，重力项都为零；线性化后 $g_{w,x}\approx -g\gamma$、$g_{w,y}/V \approx g'\phi$，回到第 2、3 节。压坡度时升力的垂直分量减小，飞机掉高并向坡度一侧转弯。`);
H3("full-moment", "4.2 力矩方程与惯性耦合");
EQ(R`\begin{aligned} I_x\dot p - I_{xz}\dot r &= (I_y - I_z)\,qr + I_{xz}\,pq + \mathcal L \\ I_y\dot q &= (I_z - I_x)\,pr + I_{xz}\,(r^{2} - p^{2}) + \mathcal M \\ I_z\dot r - I_{xz}\dot p &= (I_x - I_y)\,pq - I_{xz}\,qr + \mathcal N \end{aligned}`);
P(R`中间一式即 xmind “组合轴：滚转中的惯性耦合”。气动力矩 $\mathcal L,\mathcal M,\mathcal N$ 由导数模型给出；$I_y/I_x$ 取 CR-2144 战斗机子集（NT-33A、F-104A、F-4C）的范围，$I_z$ 按 $I_x + I_y$ 的 0.93–1.0 抽取；v4 的 $I_{xz}$ 由同一子集的惯性主轴倾角 $\varepsilon \in [0.82^\circ, 2.86^\circ]$ 给出，$I_{xz} = \tfrac12(I_z - I_x)\tan2\varepsilon$（v3 曾在 $[0.02, 0.734]$ 内独立抽取 $I_{xz}/I_x$，与小 $I_y/I_x$ 组合时主轴倾角达 10° 以上）。MIL-STD-1797A 4.8.1 要求 IV 类做 360° 全速滚转检查（PDF p.748；GJB PDF p.377）。`);
H3("full-axes", "4.3 稳定轴与机体轴");
EQ(R`p_b = p_s\cos\alpha - r_s\sin\alpha,\qquad r_b = p_s\sin\alpha + r_s\cos\alpha`);
P(R`稳定轴按配平迎角 $\alpha_0$ 固连在机体上，状态 $p_s, r_s$ 是绕它的角速度，导数也定义在这组轴上，所以从机体轴加速度换算时没有附加的转动项；上式取 $\alpha = \alpha_0$。迎角偏离配平时，绕稳定轴滚转会把迎角换成侧滑：$\dot\beta$ 中的 $-r_w = p_s\sin\Delta\alpha - r_s\cos\Delta\alpha$ 就是运动学耦合。等迎角压坡度时速度方向不偏。v4 中 $\alpha_0 = 1/(n/\alpha) + \delta$：平飞时零升起算的迎角，加上机身基准线与零升线的偏差 $\delta \in [-4.3^\circ, 3.0^\circ]$（CR-2144 战斗机清洁构型、亚音速的 15 个状态）；库中 $\alpha_0$ 中位数 4.2°，5–95 百分位 −1.1°…12.6°（v3 全部取 0.05 rad）。`);
CITE("2026-10-31 修正：此前换算用当前迎角却漏了 $\\dot\\alpha r_s$、$-\\dot\\alpha p_s$ 两项，大机动中角动量不守恒；见第 9 节的无力矩翻滚测试。");
H3("full-gust", "4.4 阵风与 Dryden 湍流");
P(R`阵风只经气动项进入（MIL-STD-1797A 4.9.3），即气动力按相对气流状态 $\mathbf x - \mathbf x_g$ 计算：`);
EQ(R`\alpha_g = \frac{w_g}{V},\quad \beta_g = \frac{v_g}{V},\quad q_g = -\frac{\partial w_g}{\partial x},\quad r_g = +\frac{\partial v_g}{\partial x}\qquad(\text{冻结湍流：}\partial/\partial x = \tfrac1V\,d/dt)`);
EQ(R`\begin{aligned} H_u &= \sigma_u\sqrt{\tfrac{2L_u}{\pi V}}\,\frac{1}{1 + \tfrac{L_u}{V}s}, & H_{v} &= \sigma_v\sqrt{\tfrac{L_v}{\pi V}}\,\frac{1 + \sqrt3\,\tfrac{L_v}{V}s}{\bigl(1 + \tfrac{L_v}{V}s\bigr)^{2}}, & H_{w} &= \sigma_w\sqrt{\tfrac{L_w}{\pi V}}\,\frac{1 + \sqrt3\,\tfrac{L_w}{V}s}{\bigl(1 + \tfrac{L_w}{V}s\bigr)^{2}} \\ H_p &= \sigma_w\sqrt{\tfrac{0.8}{V}}\,\frac{(\pi/4b)^{1/6}}{L_w^{1/3}\bigl(1 + \tfrac{4b}{\pi V}s\bigr)}, & H_q &= \frac{-s/V}{1 + \tfrac{4b}{\pi V}s}\,H_w, & H_r &= \frac{s/V}{1 + \tfrac{3b}{\pi V}s}\,H_v \end{aligned}`);
P(R`驱动为双边谱密度 π 的白噪声（离散方差 π/Δt），线速度分量方差解析等于 $\sigma^{2}$。中高空 $L_u=L_v=L_w=1750\,\mathrm{ft}$，σ 取轻 5、中 10、严重约 21 ft/s（MIL 图 262）；低空 $L_w = h$、$\sigma_w = 0.1\,u_{20}$（图 266、267）。MIL 正文以“等效飞机角速度”定义阵风角速度，符号与此处相反。GJB 表 A50（PDF p.406）另给一套模拟与试飞用的定义（中等 1.5 m/s、严重 5 m/s，尺度随高度），本项目取 MIL 图 262 的中高空值。`);
CITE("MIL-STD-1797A 4.9.2–4.9.3（PDF p.804–828）；GJB 5.8（PDF p.400–413）。");
H3("full-other", "4.5 油门与直接法向力");
EQ(R`\dot T = \frac{\delta_T - T}{\tau_e},\qquad \dot V \mathrel{+}= g\,T;\qquad \dot\alpha \mathrel{-}= \frac{g}{V}\,\delta_n,\qquad n_z \mathrel{+}= \delta_n`);
BOX("代入当前对象：惯性与湍流参数", renderFullBox);

H2("kinematics", "5 运动学：从动力学输出到派生输出");
P(R`xmind：“动力学输出经过运动学变换得到派生输出”。姿态以机体到地面的旋转矩阵 $C_{bn}$ 传播，每步按机体角速度精确旋转（Rodrigues），再读出欧拉角：`);
EQ(R`\dot C_{bn} = C_{bn}[\boldsymbol\omega]_\times,\qquad C_{k+1} = C_k\Bigl(I + \sin\vartheta\,K + (1-\cos\vartheta)\,K^{2}\Bigr),\quad \vartheta = |\boldsymbol\omega|\,\Delta t,\ K = [\boldsymbol\omega/|\boldsymbol\omega|]_\times`);
EQ(R`\phi = \operatorname{atan2}(C_{32}, C_{33}),\qquad \theta = -\arcsin C_{31},\qquad \psi = \operatorname{atan2}(C_{21}, C_{11})`);
P(R`它与 xmind 的欧拉角方程等价（测试中逐步差分核对到 1e-6），且旋转矩阵的传播本身没有 θ = 90° 的奇异（读出欧拉角时在 θ = ±90° 仍奇异，只影响显示）：`);
EQ(R`\dot\phi = p + (q\sin\phi + r\cos\phi)\tan\theta,\qquad \dot\theta = q\cos\phi - r\sin\phi,\qquad \dot\psi = \frac{q\sin\phi + r\cos\phi}{\cos\theta}`);
EQ(R`\begin{bmatrix} u\\ v\\ w\end{bmatrix} = V\begin{bmatrix}\cos\alpha\cos\beta\\ \sin\beta\\ \sin\alpha\cos\beta\end{bmatrix}\ \Bigl(\text{xmind：}V = \frac{u}{\cos\alpha\cos\beta}\Bigr),\qquad \begin{bmatrix}\dot X\\ \dot Y\\ \dot Z\end{bmatrix} = C_{bn}\begin{bmatrix} u\\ v\\ w\end{bmatrix},\qquad \gamma = -\arcsin\frac{\dot Z}{V}`);

H2("discrete", "6 离散实现");
EQ(R`\mathbf x_{k+1} = \Phi\,\mathbf x_k + \Gamma_u\,\mathbf u_{k} + \Gamma_a\,\mathbf a_k,\qquad \Phi = e^{A\Delta t},\quad [\Gamma_u\ \Gamma_a] = \int_0^{\Delta t} e^{A\tau}\,d\tau\;[B\ \ I]`);
UL([
  R`线性部分（导数矩阵 $A$、操纵矩阵 $B$）按零阶保持精确离散，$\Delta t = 1\,\mathrm{ms}$；$\mathbf a_k$ 为该步开始时算出的非线性加速度（重力、运动学、惯性、阵风、油门），在步内保持。`,
  R`纯延迟用线性插值的分数延迟（沿用原滚转通道）；相对精确连续延迟，误差中位数 1e-7 至 4e-6。`,
  R`输出约定与原 PChannel 相同：先给出当前采样的状态，再用当前输入推进一步。`,
]);

H2("criteria", "7 品质判据");
P(R`全部判据取 IV 类、A 种飞行阶段、中央杆。表中数值按 GJB 扫描件逐项核对（GJB 图 A184 印成了航向带宽图，侧滑偏离边界取图 A186–A189）。GJB 与 MIL-STD-1797A 不同之处：俯仰延迟 3 级 GJB 0.30 s、MIL 0.25 s，本项目取 GJB；首个极小值判据只在 MIL 中有。综合等级取下表各项最差；标准对各级都提出的要求（定常侧滑方向、表 A4 的 $1/T_{\theta1}>0$、长周期不得非周期发散）不满足时不属任何等级，归入 OOD。`);
P(R`仍未检查：滚转性能（GJB 5.4.9.1，需要杆力上限）、驾驶员位置侧向加速度（5.4.5）、正上反所需杆力上限（5.4.6，标准未给 A 种的侧滑范围）、表 A4 的 $\omega_{sp}T_{\theta2}$ 替代判据。`);
TABLE(["项目", "公式 / 计算方式", "1 / 2 / 3 级", "出处（PDF 页）"], [
  ["滚转模态", R`$T_R$`, "≤ 1.0 / 1.4 / 10 s", "GJB 表 A18 p.215"],
  ["螺旋模态", R`倍幅时间 $T_2 = \ln 2/\lambda_s$（$\lambda_s>0$ 时）`, "≥ 12 / 8 / 4 s", "GJB 表 A19 p.229"],
  ["滚转灵敏度", R`$S_{1s} = |\phi(1\,\mathrm s)|/F$`, "≤ 3.38 / 5.62 deg/N", "GJB 表 A31 p.276"],
  ["滚转延迟", R`$p/F_{as}$ 等效延迟 $\tau_p$（v3 未纳入，v4 起定级）`, "≤ 0.10 / 0.20 / 0.25 s", "GJB 5.4.1.1.5 表 A21 p.250；MIL 表 XXVII p.565"],
  ["荷兰滚", R`$\zeta_d,\ \zeta_d\omega_d,\ \omega_d$；当 $\omega_d^{2}|\phi/\beta|_d > 20$ 时 $(\zeta_d\omega_d)_{\min} \mathrel{+}= k_L(\omega_d^{2}|\phi/\beta|_d - 20)$，$k_L = 0.014/0.009/0.005$`, "0.19, 0.35, 1.0 / 0.02, 0.05, 0.4 / 0, —, 0.4", "GJB 表 A35 p.303；MIL 表 XL p.652"],
  ["滚转振荡", R`去螺旋的阶跃 $p$：$\zeta_d\le0.2$ 时 $\frac{p_1+p_3-2p_2}{p_1+p_3+2p_2}$，否则 $\frac{p_1-p_2}{p_1+p_2}$；另加 MIL 的首个极小值 $p_2/p_1$；$|\phi/\beta|_d < 1.5$ 不评（GJB p.245），无振荡计 1 级（本项目约定）`, R`随 $\psi_\beta$：平台 0.05/0.10，−200°…−270° 为 0.25/0.60；$p_2/p_1 \ge$ 60% / 5%（只在 MIL，v3 中从未起作用）`, "GJB 图 A116 p.240；MIL 4.5.1.4 p.550–551"],
  ["ψβ", R`$\beta_d \propto e^{-\zeta_d\omega_d t}\cos(\omega_d\sqrt{1-\zeta_d^{2}}\,t + \psi_\beta)$；取 $\beta/F_{as}\cdot\frac1s$ 在荷兰滚极点的留数相角，减 $360^\circ\tau_p/T_d$。标准按第 n 个局部极大的时刻 $t_{n\beta}$ 定义，与余弦相角差 $\arcsin\zeta_d$（本库中位数 26°），本项目取余弦相角`, R`只用于图 A116：$p$ 超前 $\beta$ 45°–225° 用上标尺，否则平移 180°；侧滑偏离图只有一个标尺`, "GJB p.13、241；MIL p.140–141"],
  ["侧滑偏离", R`$\Delta\beta$：$\max(2\,\mathrm s, T_d/2)$ 内 β 的总偏离；$k = \phi(t_{req})/\phi_{req}$（按速度段：VL/L 30° 1.1 s，M 90° 1.3 s，H 50° 1.1 s）`, R`$\Delta\beta/k$ 随 $\psi_\beta$：L1 2°→6°，L2 4°→15°；GJB 指出 $|\phi/\beta|_d > 5$ 时意义不大，v3 仍照评`, "GJB 图 A186–A189 p.334–336；MIL 图 235 p.686"],
  ["定常侧滑方向", R`$\begin{bmatrix}L_a & L'_{rp}\\ N_a & N'_{rp}\end{bmatrix}\!\begin{bmatrix}F_{as}\\ F_{rp}\end{bmatrix} = -\begin{bmatrix}L'_\beta\\ N'_\beta\end{bmatrix}\beta,\ \phi = -\frac{Y_\beta\beta + Y_{rp}F_{rp}}{g'}$`, R`右侧滑配右杆或零杆、右坡度或零；右蹬产生左侧滑。标准对各级都要求，不满足即不属任何等级（OOD；v3 曾按 3 级计）`, "GJB 5.4.6 p.33；MIL 4.5.5 p.161、4.6.1.2"],
  ["短周期频率", R`CAP $= \omega_{sp}^{2}/(n/\alpha)$`, R`0.28–3.6 且 $\omega_{sp}\ge1$ / 0.16–10 且 $\omega_{sp}\ge0.6$ / 其余`, "GJB 图 A3(a) p.78"],
  ["短周期阻尼", R`$\zeta_{sp}$`, "0.35–1.30 / 0.25–2.00 / 其余（本库上限 1.2）", "GJB 表 A6 p.77"],
  ["俯仰延迟", R`$\tau_\theta$`, "≤ 0.10 / 0.20 / 0.30 s", "GJB 表 A7 p.80"],
  ["每 g 杆力", R`$F_s/n$，$n_L = 7$：L1 上限 $\mathrm{clip}(1090/(n/\alpha), 41.7, 125)$，下限 16.7 N/g`, "按表 A10(a)", "GJB 表 A10(a) p.154"],
  ["长周期", R`$\zeta_p$；$T_2 = -0.693/(\zeta_p\omega_p)$`, R`$\zeta_p \ge 0.04$ / $\ge 0$ / $T_2 \ge 55\,\mathrm s$ 且 $|\zeta_p| < 1$；实根发散不属任何等级（OOD；v3 曾按 $T_2$ 计 3 级）`, "GJB 表 A5 p.77；MIL 4.2.1.1 p.297"],
  ["长周期零点", R`$\theta/F_s$ 的 $1/T_{\theta1}$`, R`各级都要求 $> 0$，否则 OOD（v4 起）`, "GJB 表 A4 p.77"],
]);
BOX("代入当前对象：各判据数值", renderCriteriaBox);

H2("library", "8 对象库（v4）的采样");
UL([
  R`模态按 GJB 等级采样（原 IV-A 库规则）：$T_R$、$\lambda_s$、$\zeta_d/\omega_d$、$S_{1s}$、滚转延迟 $\tau_p$、俯仰短周期；L2、L3 轮流把一项推到该级，其余各项保持 1 级（$\tau_p \le 0.10$ s）。`,
  R`耦合形状取自 CR-2144：$Y_\beta$、$N'_p/L'_p$、$\eta$ 按 5–95 百分位均匀抽取，$L'_r/N'_r$ 按真实分位数抽取；方向舵比例 $f$、$Q$ 按真实分位数相对上反与侧滑导数抽取。`,
  R`由第 3.3 节的四次方程解出导数，要求 7 个归一化量落在 94 个真实飞行状态的实际范围内，并要求 $N'_r < 0$、$M_q < 0$。`,
  R`配平迎角、惯量按 4.3、4.2 节；每个对象带飞行包线 $V_{\min}\in[50,70]$、$V_{\max}\in[260,400]\,\mathrm{m/s}$，按速度段确定侧滑偏离中的 $k$。`,
  R`第 7 节判据取最差为综合等级；1/2/3 级各 1,500，OOD 500；每级按定级来源轮流选取。OOD 包括“不属任何等级”和原滚转库的研究门限（$\omega_\phi/\omega_d$ 超出证据范围、$\zeta_\phi \notin [0.1, 0.7]$、$L_{F_a}$ 超出 IV 类证据、$\tau_p > 0.25$ s）。`,
  R`v3（同一构造、修正前的判据）保留在 library/six_channel_5000_v3，不覆盖。`,
]);
BOX("对象库统计", renderLibraryBox);

H2("verification", "9 验证");
UL([
  R`传递函数与动力学互证：导数模型的 $p/F_{as}$ 与冻结滚转通道的系数相对误差中位数 4e-14、最大 3e-11；纵向四状态的特征多项式除以 GJB 短周期因子，余项最大 5e-13（v4 每 10 个对象抽 1 个，共 500 个）；本页每个对象实时显示这两项误差。`,
  R`小幅仿真与线性模型相差 ≤ 2e-3；p 通道与冻结 PChannel 相差 6e-5。`,
  R`运动学：旋转矩阵读出的欧拉角满足 5 节三式（1e-6）；等迎角绕稳定轴滚转时航迹侧偏 < 前进距离的 1e-9。`,
  R`惯性项与直接解欧拉方程一致；无气动力矩时翻滚 2 s、迎角偏离超过 0.3 rad，惯性系角动量 $C_{bn}J\boldsymbol\omega_b$ 的相对变化 < 5e-3（修正前为 1.69）；配平点为精确平衡；下沉阵风使过载立即下降、航迹向下。`,
  R`判据在 CR-2144 真实飞机上运行：F-104A、F-4C 的侧滑偏离为 1 级；真实飞机的上反方向合格率 84%。`,
  R`v4 推导出的 $\omega_\phi/\omega_d$ 的 5–95 百分位为 0.73–1.10，真实飞机为 0.64–1.13。`,
  R`v4 库逐项检查：配额与综合等级、$\tau_p$ 等级、已定级对象满足各级通用要求、$N'_r<0$、$M_q<0$、配平迎角与主轴倾角在真实范围内、极点零点复现、等级可由存储参数重算。`,
  R`45 项 CPU 测试全部通过（v1 14、耦合模型 16、v3 库 6、v4 库 9）。`,
]);

H2("issues", "10 合理性检查（2026-10-31）与 v4 的处理");
P(R`本页公式经 sympy 逐项复推、对象库数值逐个复算，均无误。检查（含一次独立复核）发现的问题分两类。第一类在代码和页面中直接修正：第 4 节全量模型漏了坐标轴转动项；3.5 节对螺旋与上反的错误表述；观察页大机动（360° 滚转按 1 级滚转性能给杆，60° 压坡度按线性模型标定，迎角变化或侧滑超过 15° 时停止）；模态标签；其余表述与引用。第二类会改变对象的等级或导数，由 v4 重新采样解决：`);
TABLE(["v3 的问题", "依据", "v3 中受影响的对象", "v4 的处理"], [
  ["滚转等效延迟没有定级", "GJB 5.4.1.1.5 表 A21：0.10 / 0.20 / 0.25 s", R`1 级中 757 个 $\tau_p > 0.10$ s`, R`定级；1 级 $\tau_p \le 0.10$ s，L2、L3 各有以延迟定级的对象（131、163 个）`],
  ["定常侧滑方向不满足计 3 级", "GJB 5.4.6、5.5.1.1.2 对各级都要求", "3 级中 453 个", "OOD（128 个）"],
  ["长周期实根发散按 T₂ 计 3 级", "GJB 表 A5 3 级要求 |ζ_p| < 1；MIL 4.2.1.1", "3 级中 36 个", "OOD（50 个）"],
  [R`$1/T_{\theta1} \le 0$ 未检查`, "GJB 表 A4：至少大于零，不分等级", "157 个", "OOD（50 个）"],
  [R`$M_q > 0$、$N'_r > 0$`, "非飞机；真实飞机中没有", "93 个、11 个", "不入库（0 个）"],
  [R`$L'_r/N'_r$ 在 5–95 百分位内均匀抽取`, "偏向大 |L′r|", R`$L'_r$ 中位数 4.6`, R`按真实分位数抽取，中位数 3.6`],
  ["配平迎角固定 0.05 rad", R`应由 $n/\alpha$ 给出`, "全部", R`$1/(n/\alpha)$ + 零升偏差，中位数 4.2°`],
  [R`$I_{xz}/I_x$ 独立均匀抽取`, "主轴倾角偏大", "中位数 0.37", R`按主轴倾角 0.82°–2.86°，$I_{xz}/I_y \le 0.05$`],
]);
UL([
  R`等级以外仍存在的局限：导数是增稳后的等效值，$N'_r$ 约为裸机的 10 倍（3.5 节）；正上反所需杆力上限、滚转性能、驾驶员位置侧向加速度、$\omega_{sp}T_{\theta2}$ 替代判据未检查；$\Delta\beta/k$ 在 $|\phi/\beta|_d>5$ 时照评；OOD 含研究门限。`,
  R`大机动用的是线性、不随动压变化的气动导数，没有失速；迎角变化或侧滑超过约 15° 后的结果不可信，观察页在此截止。v4 每 50 个抽 1 个，360° 滚转中有 10 个（无惯性耦合时 4 个）超出该范围，迎角偏离中位数 3.1°（v3 为 6.6°）。`,
  R`v4 的定级来源：2 级 12 类、3 级 11 类，按来源轮流选取；OOD 11 类各约 45 个。新库与已训练的滚转教师不一一对应。`,
]);

// ---------------------------------------------------------------- object boxes
function renderLoesBox(box, m) {
  const l = m.loes, r = m.roll, c = m.channels;
  kv(box, [[R`$\omega_{sp}$ (rad/s)`, fmt(l.omega_sp)], [R`$\zeta_{sp}$`, fmt(l.zeta_sp)], [R`$n/\alpha$ (g/rad)`, fmt(l.n_alpha)],
    [R`$V_0$ (m/s)`, fmt(l.v0)], [R`$T_{\theta2}$ (s)`, fmt(l.t_theta2)], ["CAP", fmt(l.cap)], [R`$F_s/n$ (N/g)`, fmt(l.fs_per_g)],
    [R`$K_\theta$ (rad/s²/N)`, fmt(l.k_theta)], [R`$\tau_\theta$ (s)`, fmt(l.tau_theta)], [R`$T_R$ (s)`, fmt(r.t_r)], [R`$\lambda_s$ (1/s)`, fmt(r.lambda_s)],
    [R`$\zeta_d,\ \omega_d$`, `${fmt(r.zeta_d)}, ${fmt(r.omega_d)}`], [R`$\zeta_\phi,\ \omega_\phi$`, `${fmt(r.zeta_phi)}, ${fmt(r.omega_phi)}`],
    [R`$S_{1s}$ (deg/N)`, fmt(r.sensitivity)], [R`$L_{F_a}$`, fmt(r.l_fa)], [R`$\tau_p$ (s)`, fmt(r.tau_p)]]);
  eq(box, tfTeX(c.q_Fs_loes, "q", "F_s"));
  eq(box, tfTeX(c.alpha_Fs_loes, R`\alpha`, "F_s") + R`,\qquad ` + tfTeX(c.nz_Fs_loes, "n_z", "F_s"));
  eq(box, tfTeX(c.p_Fas_loes, "p", "F_{as}"));
  rich(box, "p", R`单位：角度 rad、角速度 rad/s、力 N、过载 g。`, { class: "note" });
}

function renderLongitudinalBox(box, m) {
  const d = m.longitudinal, c = m.channels;
  kv(box, [[R`$X_u$`, fmt(d.x_u)], [R`$X_\alpha$`, fmt(d.x_a)], [R`$Z_u$`, fmt(d.z_u)], [R`$Z_\alpha$`, fmt(d.z_a)],
    [R`$M_u$`, fmt(d.m_u)], [R`$M_\alpha$`, fmt(d.m_a)], [R`$M_q$`, fmt(d.m_q)], [R`$K_\theta$`, fmt(d.m_s)],
    [R`$1/T_{\theta1}$`, fmt(d.inv_t_theta1)], [R`$1/T_{\theta2}$`, fmt(1 / m.loes.t_theta2)],
    ["长周期", d.phugoid ? `ωp ${fmt(d.phugoid.omega_p)}，ζp ${fmt(d.phugoid.zeta_p)}` : `实根：s² + ${fmt(d.long_period[0])} s + ${fmt(d.long_period[1])}`],
    [R`$\sqrt2\,g/V_0$`, fmt(Math.SQRT2 * 9.80665 / m.loes.v0)]]);
  eq(box, R`A_{\mathrm{lon}} = ` + matTeX(d.A) + R`,\qquad B_{\mathrm{lon}} = ` + matTeX(d.B) + R`\quad \bigl(\mathbf x = [u,\alpha,q,\theta],\ \mathbf u = [F_s,\delta_T]\bigr)`);
  eq(box, R`\Delta_{\mathrm{lon}} = ${polyTeX(c.q_Fs.den)} = ${factorsTeX(c.q_Fs.poles)}`);
  eq(box, tfTeX(c.theta_Fs, R`\theta`, "F_s"));
  eq(box, tfTeX(c.nz_Fs, "n_z", "F_s"));
  eq(box, tfTeX(c.u_dT, "V - V_0", R`\delta_T`));
  rich(box, "p", `短周期因子校核：Δ_lon ÷ (s² + 2ζω s + ω²) 的余项相对值 ${fmt(m.checks.longitudinal_short_period_remainder)}。`, { class: "note" });
}

function renderLateralBox(box, m) {
  const d = m.lateral, c = m.channels;
  kv(box, [[R`$Y_\beta$`, fmt(d.y_b)], [R`$L'_\beta$`, fmt(d.l_b)], [R`$L'_p$`, fmt(d.l_p)], [R`$L'_r$`, fmt(d.l_r)],
    [R`$N'_\beta$`, fmt(d.n_b)], [R`$N'_p$`, fmt(d.n_p)], [R`$N'_r$`, fmt(d.n_r)], [R`$g'$`, fmt(d.g_over_v)],
    [R`$L_a,\ N_a$`, `${fmt(d.l_a)}, ${fmt(d.n_a)}`], [R`$\eta = N_a/L_a$`, fmt(d.eta)],
    [R`$Y_{rp},\ L'_{rp},\ N'_{rp}$`, `${fmt(d.y_rp)}, ${fmt(d.l_rp)}, ${fmt(d.n_rp)}`],
    [R`上反比 $-L'_\beta/N'_\beta$`, fmt(-d.l_b / d.n_b)], [R`$c_1 = N'_p/L'_p$`, fmt(d.n_p / d.l_p)], [R`$c_2 = L'_r/N'_r$`, fmt(d.l_r / d.n_r)]]);
  eq(box, R`A_{\mathrm{lat}} = ` + matTeX(d.A) + R`,\qquad B_{\mathrm{lat}} = ` + matTeX(d.B) + R`\quad \bigl(\mathbf x = [\beta,p,r,\phi],\ \mathbf u = [F_{as},F_{rp}]\bigr)`);
  eq(box, R`\Delta_{\mathrm{lat}} = ${polyTeX(c.p_Fas.den)} = ${factorsTeX(c.p_Fas.poles)}`);
  for (const [key, out, inp] of [["p_Fas", "p", "F_{as}"], ["beta_Fas", R`\beta`, "F_{as}"], ["r_Fas", "r", "F_{as}"],
    ["beta_Frp", R`\beta`, "F_{rp}"], ["r_Frp", "r", "F_{rp}"], ["p_Frp", "p", "F_{rp}"]]) eq(box, tfTeX(c[key], out, inp));
  rich(box, "p", `与冻结滚转通道的对照：导数模型 p/F_as 与 LOES 系数最大相对误差 ${fmt(m.checks.lateral_vs_roll_loes_max_rel)}。`, { class: "note" });
  htmlEl("div", { class: "box-title" }, "模态与特征向量（以最大分量归一，幅值∠相角）", box);
  const table = htmlEl("table", { class: "criteria" }, null, box);
  const head = htmlEl("tr", {}, null, table);
  for (const h of ["模态", "特征值", "ω / ζ 或时间常数", "β", "p", "r", "φ"]) htmlEl("th", {}, h, head);
  const rows = [...m.modes.lateral, ...m.modes.longitudinal.map((x) => ({ ...x, lon: true }))];
  for (const mode of rows) {
    const tr = htmlEl("tr", {}, null, table);
    htmlEl("td", {}, mode.mode, tr);
    htmlEl("td", { class: "num" }, `${fmt(mode.eigenvalue[0])}${mode.eigenvalue[1] ? ` ± ${fmt(mode.eigenvalue[1])}j` : ""}`, tr);
    htmlEl("td", { class: "num" }, mode.omega ? `${fmt(mode.omega)} rad/s / ${fmt(mode.zeta)}` : `${fmt(mode.time_constant)} s`, tr);
    const names = mode.lon ? ["u", "α", "q", "θ"] : ["β", "p", "r", "φ"];
    names.forEach((n) => { const v = mode.vector[n]; htmlEl("td", { class: "num" }, `${n} ${fmt(v[0], 2)}∠${fmt(v[1], 3)}°`, tr); });
  }
  rich(box, "p", `荷兰滚的 |φ/β|d = ${fmt(m.criteria.dutch_roll_shape.phi_over_beta)}，ψβ = ${fmt(m.criteria.psi_beta_deg)}°。纵向模态行的四列依次为 u、α、q、θ。`, { class: "note" });
}

const CHANNEL_NAMES = {
  q_Fs: "q/Fs", q_Fs_loes: "q/Fs（LOES）", theta_Fs: "θ/Fs", alpha_Fs: "α/Fs", nz_Fs: "nz/Fs", u_Fs: "V/Fs", u_dT: "V/δT",
  theta_dT: "θ/δT", p_Fas: "p/Fas", p_Fas_loes: "p/Fas（LOES）", beta_Fas: "β/Fas", r_Fas: "r/Fas", phi_Fas: "φ/Fas",
  beta_Frp: "β/Frp", r_Frp: "r/Frp", p_Frp: "p/Frp",
};
function renderFrequencyBox(box, m) {
  const row = htmlEl("div", { class: "chart-row" }, null, box);
  const c = m.channels, colors = [css("--series-1"), css("--series-2"), css("--series-3")];
  poleZeroChart(htmlEl("div", { class: "chart" }, null, row), { title: "横航向：Δlat 极点（×）与零点（○）", sets: [
    { name: "Δlat 极点", kind: "pole", roots: c.p_Fas.poles, color: css("--text-primary") },
    { name: "p/Fas 零点", kind: "zero", roots: c.p_Fas.zeros, color: colors[0] },
    { name: "β/Fas 零点", kind: "zero", roots: c.beta_Fas.zeros, color: colors[1] },
    { name: "β/Frp 零点", kind: "zero", roots: c.beta_Frp.zeros, color: colors[2] }] });
  poleZeroChart(htmlEl("div", { class: "chart" }, null, row), { title: "纵向：Δlon 极点（×）与零点（○）", sets: [
    { name: "Δlon 极点", kind: "pole", roots: c.q_Fs.poles, color: css("--text-primary") },
    { name: "θ/Fs 零点", kind: "zero", roots: c.theta_Fs.zeros, color: colors[0] },
    { name: "α/Fs 零点", kind: "zero", roots: c.alpha_Fs.zeros, color: colors[1] }] });
  legend(htmlEl("div", { class: "legend" }, null, box), [{ label: "极点", color: css("--text-primary"), kind: "cross" },
    { label: "零点：左图 p/Fas、β/Fas、β/Frp，右图 θ/Fs、α/Fs（颜色依次）", color: colors[0], kind: "ring" },
    { label: "虚线圈：超出画幅", color: css("--text-muted"), kind: "ring" }]);
  htmlEl("div", { class: "box-title" }, "频率响应（含纯延迟；最多选三个通道）", box);
  const picker = htmlEl("div", { class: "channel-picker" }, null, box);
  for (const key of Object.keys(CHANNEL_NAMES)) {
    const label = htmlEl("label", {}, null, picker);
    const input = htmlEl("input", { type: "checkbox" }, null, label);
    input.checked = page.bode.has(key);
    label.appendChild(document.createTextNode(CHANNEL_NAMES[key]));
    input.addEventListener("change", () => {
      if (input.checked) { page.bode.add(key); while (page.bode.size > 3) page.bode.delete(page.bode.values().next().value); }
      else page.bode.delete(key);
      renderBoxes();
    });
  }
  while (page.bode.size > 3) page.bode.delete(page.bode.values().next().value);
  const chosen = [...page.bode];
  const series = (field) => chosen.map((key, i) => ({ name: CHANNEL_NAMES[key], values: c[key].bode[field], color: colors[i], dash: DASHES[i] }));
  legend(htmlEl("div", { class: "legend" }, null, box), chosen.map((key, i) => ({ label: CHANNEL_NAMES[key], color: colors[i], dash: DASHES[i] })));
  const charts = htmlEl("div", { class: "chart-row" }, null, box);
  bodeChart(htmlEl("div", { class: "chart" }, null, charts), { title: "幅值 (dB)", freqs: m.frequencies, series: series("db"), field: "db" });
  bodeChart(htmlEl("div", { class: "chart" }, null, charts), { title: "相角 (deg)", freqs: m.frequencies, series: series("deg"), field: "deg" });
}

function renderFullBox(box, m) {
  const i = m.inertia, v = m.loes.v0, b = m.span_m, ft = 0.3048;
  kv(box, [[R`$I_y/I_x$`, fmt(i.iy)], [R`$I_z/I_x$`, fmt(i.iz)], [R`$I_{xz}/I_x$`, fmt(i.ixz)],
    [R`$(I_z - I_x)/I_y$（$pr$ 系数）`, fmt((i.iz - 1) / i.iy)], [R`$I_{xz}/I_y$（$r^2-p^2$ 系数）`, fmt(i.ixz / i.iy)],
    ["翼展 b (m)", fmt(b)], ["中度湍流 σ (m/s)", fmt(10 * ft)], ["L (m)", fmt(1750 * ft)],
    [R`$L/V$ (s)`, fmt(1750 * ft / v)], [R`$4b/(\pi V)$ (s)`, fmt(4 * b / (Math.PI * v))]]);
}

function renderCriteriaBox(box, m) {
  const c = m.criteria, lv = m.component_levels;
  const level = (k) => (lv[k] ? `L${lv[k]}` : "超出");
  const rows = [
    ["滚转模态 T_R", `${fmt(m.roll.t_r)} s`, level("roll_t_r")],
    ["螺旋 λs", `${fmt(m.roll.lambda_s)} 1/s${m.roll.lambda_s > 0 ? `（倍幅 ${fmt(Math.LN2 / m.roll.lambda_s)} s）` : ""}`, level("roll_spiral")],
    ["1 s 灵敏度", `${fmt(m.roll.sensitivity)} deg/N`, level("roll_sensitivity_1s")],
    ["滚转延迟", `τp ${fmt(m.roll.tau_p)} s（GJB 表 A21：0.10 / 0.20 / 0.25 s）`, level("roll_delay")],
    ["荷兰滚", `ζd ${fmt(m.roll.zeta_d)}，ζdωd ${fmt(m.roll.zeta_d * m.roll.omega_d)}，ωd²|φ/β| ${fmt(c.dutch_roll.omega_d2_phi_over_beta)}（L1 下限 ${fmt(c.dutch_roll.corrected_product_min_level_1)}）`, level("roll_dutch_roll")],
    ["滚转振荡", `p_osc/p_av ${fmt(c.roll_oscillation.p_osc_over_p_av)}（限 ${fmt(c.roll_oscillation.limit_level_1)} / ${fmt(c.roll_oscillation.limit_level_2)}）；p2/p1 ${fmt(c.roll_oscillation.first_minimum_ratio)}${c.roll_oscillation.applicable ? "" : "；|φ/β|<1.5，不适用"}`, level("roll_oscillation")],
    ["侧滑偏离", `Δβ/k ${fmt(c.sideslip_excursion.delta_beta_over_k_deg)}°（速度段 ${m.envelope.speed_range}，限 ${fmt(c.sideslip_excursion.limit_level_1)} / ${fmt(c.sideslip_excursion.limit_level_2)}），ψβ ${fmt(c.psi_beta_deg)}°`, level("sideslip_excursion")],
    ["定常侧滑方向", `每度侧滑：杆 ${fmt(c.steady_sideslip.roll_force_n_per_deg)} N、蹬 ${fmt(c.steady_sideslip.pedal_force_n_per_deg)} N、坡度 ${fmt(c.steady_sideslip.bank_deg_per_deg)}°`, level("steady_sideslip_sense")],
    ["短周期频率", `CAP ${fmt(m.loes.cap)}，ωsp ${fmt(m.loes.omega_sp)}`, level("pitch_frequency")],
    ["短周期阻尼", `ζsp ${fmt(m.loes.zeta_sp)}`, level("pitch_damping")],
    ["俯仰延迟", `τθ ${fmt(m.loes.tau_theta)} s`, level("pitch_delay")],
    ["每 g 杆力", `${fmt(m.loes.fs_per_g)} N/g`, level("pitch_stick_force")],
    ["长周期零点", `1/Tθ1 ${fmt(m.longitudinal.inv_t_theta1)} 1/s`, level("pitch_inv_t_theta1")],
    ["长周期", m.longitudinal.phugoid ? `ζp ${fmt(m.longitudinal.phugoid.zeta_p)}` : (c.long_period.time_to_double_s ? `发散，倍幅 ${fmt(c.long_period.time_to_double_s)} s` : "两个稳定实根"), level("long_period")],
  ];
  const table = htmlEl("table", { class: "criteria" }, null, box);
  const head = htmlEl("tr", {}, null, table);
  for (const h of ["判据", "本对象", "等级"]) htmlEl("th", {}, h, head);
  for (const [name, value, l] of rows) {
    const tr = htmlEl("tr", {}, null, table);
    htmlEl("td", {}, name, tr); htmlEl("td", {}, value, tr); htmlEl("td", { class: "num" }, l, tr);
  }
  rich(box, "p", `综合等级：${m.level === 1 ? "L1；全部判据 1 级" : m.level ? `L${m.level}；由「${LABELS[m.group] || m.group}」定级` : `OOD（${(m.ood_reasons || []).map((k) => LABELS[k] || k).join("、")}）`}。`, { class: "note" });
}

function renderLibraryBox(box) {
  const s = page.summary; if (!s) return;
  const counts = {}; for (const row of s.rows) counts[row[2]] = (counts[row[2]] || 0) + 1;
  kv(box, [["候选 / 能对应真实飞机", `${s.manifest.candidate_count.toLocaleString()} / ${s.manifest.valid_candidates.toLocaleString()}`],
    ["候选池 L1 / L2 / L3 / OOD", Object.values(s.manifest.pool_sizes).map((v) => v.toLocaleString()).join(" / ")],
    ["入库 L1 / L2 / L3 / OOD", [1, 2, 3, 0].map((l) => (counts[l] || 0).toLocaleString()).join(" / ")], ["随机种子", String(s.manifest.seed)]]);
  const p = htmlEl("p", { class: "note" }, null, box);
  p.appendChild(document.createTextNode("逐对象浏览、参数分布与真实飞机对照见 "));
  htmlEl("a", { href: `library#id=${encodeURIComponent(page.id || "")}` }, "对象库浏览", p);
  p.appendChild(document.createTextNode("。"));
}

// ---------------------------------------------------------------- assembly
const boxes = [];
function buildDocument() {
  const content = document.getElementById("content"), toc = document.getElementById("toc");
  for (const block of DOC) {
    const [kind] = block;
    if (kind === "h2" || kind === "h3") {
      const [, id, title] = block;
      htmlEl(kind, { id }, title, content);
      htmlEl("a", { href: `#${id}`, class: kind === "h3" ? "sub" : "" }, title, toc);
    } else if (kind === "p") rich(content, "p", block[1]);
    else if (kind === "eq") eq(content, block[1]);
    else if (kind === "cite") rich(content, "p", block[1], { class: "cite" });
    else if (kind === "ul") { const ul = htmlEl("ul", {}, null, content); block[1].forEach((item) => rich(ul, "li", item)); }
    else if (kind === "table") {
      const table = htmlEl("table", { class: "criteria" }, null, content);
      const head = htmlEl("tr", {}, null, table); block[1].forEach((h) => htmlEl("th", {}, h, head));
      block[2].forEach((row) => { const tr = htmlEl("tr", {}, null, table); row.forEach((cell) => rich(tr, "td", cell)); });
    } else if (kind === "box") {
      const box = htmlEl("div", { class: "object-box" }, null, content);
      htmlEl("div", { class: "box-title" }, block[1], box);
      const body = htmlEl("div", {}, null, box);
      boxes.push([body, block[2]]);
    } else if (kind === "flow") {
      const flow = htmlEl("div", { class: "flow" }, null, content);
      for (const [title, text] of [["杆力输入", "Fs、Fas、Frp（N）；油门 δT、直接法向力 δn"], ["六通道 LOES", "q、α、nz｜p｜β、r：GJB 等效系统，参数按品质等级采样"],
        ["小扰动动力学", "纵向 [u,α,q,θ] + 横航向 [β,p,r,φ]：精确对应 LOES，交叉通道与长周期随之确定"],
        ["全量动力学", "风轴重力与运动学、欧拉方程惯性耦合、阵风（只经气动项）"], ["运动学", "旋转矩阵 → φ、θ、ψ；速度 → X、Y、Z；γ"],
        ["判据与对象库", "GJB/MIL 全部判据取最差；5,000 个对象"]]) {
        const card = htmlEl("div", {}, null, flow);
        htmlEl("b", {}, title, card); htmlEl("span", {}, text, card);
      }
    }
  }
}

function renderBoxes() {
  for (const [body, render] of boxes) {
    body.replaceChildren();
    if (!page.model) { htmlEl("p", { class: "muted" }, "加载中…", body); continue; }
    try { render(body, page.model); } catch (error) { htmlEl("p", { class: "warn" }, `渲染失败：${error.message}`, body); }
  }
}

async function choose(id) {
  page.id = id;
  history.replaceState(null, "", `#id=${encodeURIComponent(id)}${location.hash.includes("&") ? location.hash.slice(location.hash.indexOf("&")) : ""}`);
  document.getElementById("object-input").value = id;
  document.querySelectorAll(".object-box").forEach((b) => b.classList.add("loading"));
  const response = await fetch(`api/model?id=${encodeURIComponent(id)}`, { cache: "no-store" });
  if (!response.ok) return;
  page.model = await response.json();
  const chip = document.getElementById("object-chip");
  chip.replaceChildren();
  const span = htmlEl("span", { class: "chip" }, null, chip);
  htmlEl("span", { class: "key" }, null, span).style.background = levelColor(page.model.level || 0);
  span.appendChild(document.createTextNode(page.model.level ? `综合 L${page.model.level}` : "OOD"));
  renderBoxes();
  document.querySelectorAll(".object-box").forEach((b) => b.classList.remove("loading"));
}

(async function main() {
  buildDocument();
  renderBoxes();
  const summary = await (await fetch("api/summary", { cache: "no-store" })).json();
  page.summary = summary;
  const ids = document.getElementById("object-ids");
  for (const row of summary.rows) htmlEl("option", { value: row[0] }, null, ids);
  const input = document.getElementById("object-input");
  input.addEventListener("change", () => { if (summary.rows.some((r) => r[0] === input.value)) choose(input.value); });
  const wanted = new URLSearchParams(location.hash.slice(1)).get("id");
  await choose(summary.rows.some((r) => r[0] === wanted) ? wanted : "train_core-0000");
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderBoxes);
})();
