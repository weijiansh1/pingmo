'use strict';
let data, index = 50, timer = null;
const el = id => document.getElementById(id);
const fmt = (x, digits = 3, signed = false) => {
  if (Math.abs(x) < .5 * 10 ** (-digits)) x = 0;
  return (signed && x > 0 ? '+' : '') + x.toFixed(digits);
};
function stop() {
  if (timer !== null) clearInterval(timer);
  timer = null;
  el('play').textContent = '播放';
}
function setIndex(value) {
  index = Math.max(0, Math.min(data.cycles.length - 1, value));
  el('timeline').value = index;
  render();
}
function render() {
  const r = data.cycles[index];
  el('clock').textContent = `${fmt(r.t0, 2)} → ${fmt(r.t1, 2)} s · 第 ${index + 1} 周期`;
  el('input-time').textContent = `${fmt(r.t0, 2)} s：做动作之前`;
  el('output-time').textContent = `${fmt(r.t1, 2)} s：20 ms 之后`;
  el('command').textContent = fmt(r.command_deg_s, 2) + ' deg/s';
  el('target').textContent = fmt(r.target_deg_s, 3) + ' deg/s';
  el('p-before').textContent = fmt(r.p_before_deg_s, 3) + ' deg/s';
  el('error').textContent = fmt(r.error_before_deg_s, 3, true) + ' deg/s';
  el('action').textContent = fmt(r.action, 4, true);
  el('action-marker').style.left = `${(r.action + 1) * 50}%`;
  const direction = r.requested_change_n > 1e-6 ? '增加' : r.requested_change_n < -1e-6 ? '减少' : '保持';
  el('action-explanation').textContent = `这个动作表示本步${direction}请求力。按当前接口：1.76 × 动作 = ${fmt(r.unclipped_change_n, 4, true)} N，再作力限幅。`;
  el('force-before').textContent = fmt(r.requested_before_n, 3) + ' N';
  el('force-change').textContent = fmt(r.requested_change_n, 4, true) + ' N';
  el('force-after').textContent = fmt(r.requested_after_n, 3) + ' N';
  el('applied').textContent = fmt(r.applied_after_n, 3) + ' N';
  el('p-after').innerHTML = fmt(r.p_after_deg_s, 3) + ' <small>deg/s</small>';
  el('p-change').textContent = '本周期角速度变化：' + fmt(r.p_after_deg_s - r.p_before_deg_s, 4, true) + ' deg/s';
  el('previous').disabled = index === 0;
  el('next').disabled = index === data.cycles.length - 1;
  document.querySelectorAll('[data-step]').forEach(button => button.classList.toggle('active', Number(button.dataset.step) === index));
  let explanation;
  if (r.t0 < 1) {
    explanation = '指令还没到，目标和实际角速度都为零，冻结学生输出零动作。这个对象的参数已经选定，后面不会随周期改变。';
  } else if (Math.abs(r.p_after_deg_s) < 1e-8 && r.action > .001) {
    explanation = '指令已经变为 +15 deg/s，学生开始加力；但这 20 ms 内角速度仍为零。对象有约 43 ms 运输时延，力的变化不会立即出现在角速度上。';
  } else if (r.action < -.0001 && r.p_after_deg_s > r.p_before_deg_s) {
    explanation = '这一步在减小请求力，角速度却仍继续上升。请求力仍可能为正，执行机构和飞机状态也有动态记忆；减力不等于角速度立即下降。';
  } else if (r.error_before_deg_s < 0 && r.action > .0001) {
    explanation = '实际角速度略高于旧参考目标，本步仍小幅加力。学生还读取目标变化和历史，动作不是仅按当前误差的正负计算。';
  } else {
    explanation = `本周期，学生根据当前观测与历史计算动作，将请求力${direction}；仿真飞机推进 20 ms，产生新角速度，再把新反馈交给下一周期。`;
  }
  el('explanation').textContent = explanation;
  const obs = r.observation_normalized;
  const rows = [
    ['外部指令', r.command_deg_s, 'deg/s'], ['旧参考目标', r.target_deg_s, 'deg/s'],
    ['实际角速度', r.p_before_deg_s, 'deg/s'], ['目标跟踪误差', r.error_before_deg_s, 'deg/s'],
    ['误差积分', obs[4] * 30, 'deg'], ['实际角速度导数', obs[5] * 5 * 180 / Math.PI, 'deg/s²'],
    ['旧目标导数', obs[6] * 5 * 180 / Math.PI, 'deg/s²'],
    ['请求力', r.requested_before_n, 'N'], ['执行机构输出', r.applied_before_n, 'N']
  ];
  el('input-table').innerHTML = rows.map((row, i) => `<tr><td>${row[0]}</td><td>${fmt(row[1], 4)} ${row[2]}</td><td>${fmt(obs[i], 5)}</td></tr>`).join('');
  drawCharts();
}
function drawChart(canvas, curves, unit, minimumSpan) {
  const width = canvas.clientWidth, height = canvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const left = 51, right = 13, top = 24, bottom = 31;
  const pw = width - left - right, ph = height - top - bottom;
  let lo = 0, hi = minimumSpan;
  for (const curve of curves) for (const point of curve.values) {
    lo = Math.min(lo, point[1]); hi = Math.max(hi, point[1]);
  }
  const pad = Math.max(minimumSpan, hi - lo) * .1;
  lo -= pad; hi += pad;
  const x = t => left + t / 8 * pw;
  const y = v => top + (hi - v) / (hi - lo) * ph;
  ctx.font = '11px DemoChinese, sans-serif';
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i <= 4; i++) {
    const v = lo + (hi - lo) * i / 4;
    ctx.strokeStyle = '#e1e8ee'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(left, y(v)); ctx.lineTo(width - right, y(v)); ctx.stroke();
    ctx.fillStyle = '#53677a'; ctx.fillText(fmt(v, hi - lo > 6 ? 1 : 2), left - 8, y(v));
  }
  ctx.textAlign = 'center';
  for (let t = 0; t <= 8; t += 2) {
    ctx.strokeStyle = '#eef2f6'; ctx.beginPath(); ctx.moveTo(x(t), top); ctx.lineTo(x(t), height - bottom); ctx.stroke();
    ctx.fillStyle = '#53677a'; ctx.fillText(t + ' s', x(t), height - 13);
  }
  ctx.save(); ctx.beginPath(); ctx.rect(left, top, pw, ph); ctx.clip();
  const r = data.cycles[index];
  ctx.fillStyle = '#0072b214'; ctx.fillRect(x(r.t0), top, Math.max(2, x(r.t1) - x(r.t0)), ph);
  for (const curve of curves) {
    ctx.strokeStyle = curve.color; ctx.lineWidth = curve.width || 1.8;
    ctx.setLineDash(curve.dash || []); ctx.beginPath();
    curve.values.forEach((point, i) => i === 0 ? ctx.moveTo(x(point[0]), y(point[1])) : ctx.lineTo(x(point[0]), y(point[1])));
    ctx.stroke();
  }
  ctx.setLineDash([]); ctx.strokeStyle = '#20435d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x(r.t0), top); ctx.lineTo(x(r.t0), top + ph); ctx.stroke();
  for (const curve of curves) {
    const point = curve.values[index];
    ctx.fillStyle = curve.color; ctx.beginPath(); ctx.arc(x(point[0]), y(point[1]), 3.3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore(); ctx.fillStyle = '#53677a'; ctx.textAlign = 'left'; ctx.fillText(unit, 5, 11);
}
function drawCharts() {
  if (!data) return;
  const last = data.cycles[data.cycles.length - 1];
  const series = (key, finalKey) => data.cycles.map(r => [r.t0, r[key]]).concat([[last.t1, last[finalKey || key]]]);
  drawChart(el('rate-chart'), [
    {color:'#172a3b', dash:[7,4], values:series('command_deg_s','next_command_deg_s')},
    {color:'#647889', dash:[3,3], values:series('target_deg_s','next_target_deg_s')},
    {color:'#0072b2', width:2, values:series('p_before_deg_s','p_after_deg_s')}
  ], '角速度 deg/s', 1);
  drawChart(el('force-chart'), [
    {color:'#bc4b00', values:series('requested_before_n','requested_after_n')},
    {color:'#007e62', dash:[2,3], values:series('applied_before_n','applied_after_n')}
  ], '力 N', .1);
}
async function start() {
  const response = await fetch('run.json');
  if (!response.ok) throw new Error('record');
  data = await response.json();
  el('timeline').max = data.cycles.length - 1;
  el('delay').textContent = '约 ' + Math.round(data.object_parameters.tau_p * 1000) + ' ms';
  const parameters = [
    ['l_fa','L_Fa · 控制效能','输入力作用的强弱'], ['lambda_s','λ_s · 螺旋模态特征根','慢动态衰减或发散'],
    ['t_r','T_R · 滚转时间常数','滚转模态响应快慢'], ['zeta_d','ζ_d · 振荡模态阻尼比','振荡的衰减'],
    ['omega_d','ω_d · 振荡模态自然频率','振荡的时间尺度'], ['r_omega','R_ω · 零点频率比','影响零点位置'],
    ['r_zeta','R_ζ · 零点阻尼比例','影响零点位置'], ['tau_p','τ_p · 输入运输时延','控制力何时到达对象']
  ];
  el('parameter-table').innerHTML = parameters.map(([key, label, description]) => `<tr><td>${label}</td><td>${fmt(data.object_parameters[key],6)}</td><td>${description}</td></tr>`).join('');
  el('timeline').addEventListener('input', event => { stop(); setIndex(Number(event.target.value)); });
  el('previous').addEventListener('click', () => { stop(); setIndex(index - 1); });
  el('next').addEventListener('click', () => { stop(); setIndex(index + 1); });
  document.querySelectorAll('[data-step]').forEach(button => button.addEventListener('click', () => { stop(); setIndex(Number(button.dataset.step)); }));
  el('play').addEventListener('click', () => {
    if (timer !== null) return stop();
    if (index === data.cycles.length - 1) setIndex(0);
    el('play').textContent = '暂停';
    timer = setInterval(() => {
      setIndex(index + 1);
      if (index === data.cycles.length - 1) stop();
    }, 120);
  });
  await document.fonts.ready;
  new ResizeObserver(drawCharts).observe(el('rate-chart'));
  setIndex(50);
}
start().catch(() => { el('explanation').textContent = '回放记录载入失败，请刷新页面。'; });
