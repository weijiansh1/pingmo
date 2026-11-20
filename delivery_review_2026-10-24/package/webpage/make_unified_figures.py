"""Figures for the 4th step: the unified DHC-6 conditional student.

Reads only ``dhc6_finetune/unified/`` (summaries, metrics, weights, theta) and
the pinned evaluation batch, so every plotted number is the archived value.
Trajectories are replayed from the saved weights with ``torch.no_grad`` — no
training is performed.

    python dhc6_blog_assets/make_unified_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
WS = HERE.parent
UNI = WS / "dhc6_finetune" / "unified"
sys.path.insert(0, str(WS / "pingmo"))

from src.transfer import fine_tune as ft  # noqa: E402

matplotlib.rcParams["axes.unicode_minus"] = False
for _c in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if _c in {f.name for f in font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _c
        break

BUDGETS = (15, 30, 60)
TOTAL = {15: 135, 30: 240, 60: 480}
ARMS = ("frozen", "full", "head_only", "scratch")
LABEL = {"frozen": "冻结 W0", "full": "全量微调", "head_only": "仅输出头", "scratch": "同预算从零"}
COLOR = {"frozen": "#64748b", "full": "#b91c1c", "head_only": "#0f766e", "scratch": "#d97706"}

SUMMARY = {b: json.loads((UNI / f"b{b}" / "summary.json").read_text(encoding="utf-8")) for b in BUDGETS}
METRICS = {b: json.loads((UNI / f"b{b}" / "metrics.json").read_text(encoding="utf-8")) for b in BUDGETS}
THETA = {b: json.loads((UNI / f"b{b}" / "theta.json").read_text(encoding="utf-8")) for b in BUDGETS}
EVAL = json.loads((UNI / "shared" / "eval_models.json").read_text(encoding="utf-8"))

# ------------------------------------------------------------- budget curves
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
for ax, group, title in zip(axes, ("adaptation", "heldout"),
                            ("8 个适配状态", "4 个跨状态留出条件")):
    values = {arm: [SUMMARY[b]["arms"][arm]["summary"][group]["mean_rmse_deg_s"]
                    for b in BUDGETS] for arm in ARMS}
    for arm in ARMS:
        ax.plot([TOTAL[b] for b in BUDGETS], values[arm], marker="o", ms=6, lw=2.0,
                color=COLOR[arm], label=LABEL[arm])
    ax.set_xticks([TOTAL[b] for b in BUDGETS])
    ax.set_xticklabels([f"{TOTAL[b]} s\n({b} s/状态)" for b in BUDGETS])
    ax.set_xlabel("真实目标数据总秒数（含重叠去重）")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.margins(x=0.16, y=0.18)
axes[0].set_ylabel("平均跟踪 RMSE (deg/s)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("三档预算下的统一条件学生：组均值 RMSE（单 seed，未选“最佳预算”）", fontsize=13)
# All twelve values as a table: per-point labels collided with the plotted
# lines, so the numbers live here where they are exactly readable.
table_rows = []
for arm in ARMS:
    row = [LABEL[arm]]
    for j, b in enumerate(BUDGETS):
        for group in ("adaptation", "heldout"):
            row.append(f"{SUMMARY[b]['arms'][arm]['summary'][group]['mean_rmse_deg_s']:.3f}")
    table_rows.append(row)
columns = ["arm"] + [f"{TOTAL[b]} s\n{'适配' if g == 'adaptation' else '留出'}"
                     for b in BUDGETS for g in ("adaptation", "heldout")]
tbl_ax = fig.add_axes([0.08, -0.30, 0.86, 0.18])
tbl_ax.axis("off")
table = tbl_ax.table(cellText=table_rows, colLabels=columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8.2)
table.scale(1.0, 1.35)
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#cbd5e1")
    if row == 0:
        cell.set_text_props(fontweight="bold")
fig.text(0.5, -0.34, "表中为 12 条指令上的平均 RMSE（deg/s），与图中各点一一对应；"
                    "B=30 s/状态 是沿用 pilot 的中间档，不因其数值较低而视为已选最佳预算。",
         ha="center", fontsize=8.2, color="#64748b")
fig.savefig(HERE / "unified_budget_curves.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote unified_budget_curves.png")

# --------------------------------------------------------------- heatmap
states = list(METRICS[15]["arms"]["frozen"]["conditions"])
rows = [s for s in states
        if METRICS[15]["arms"]["frozen"]["conditions"][s]["group"] == "adaptation"]
rows += [s for s in states
         if METRICS[15]["arms"]["frozen"]["conditions"][s]["group"] == "heldout"]
matrix = np.zeros((len(rows), len(BUDGETS)))
for j, b in enumerate(BUDGETS):
    for i, state in enumerate(rows):
        fz = np.mean([c["rmse_deg_s"] for c in METRICS[b]["arms"]["frozen"]["conditions"][state]["commands"]])
        fu = np.mean([c["rmse_deg_s"] for c in METRICS[b]["arms"]["full"]["conditions"][state]["commands"]])
        matrix[i, j] = (fz - fu) / fz * 100.0
labels = [f"{s} ★" if METRICS[15]["arms"]["frozen"]["conditions"][s]["group"] == "heldout" else s
          for s in rows]
fig, ax = plt.subplots(figsize=(7.4, 7.4))
im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=float(np.nanmax(matrix)))
ax.set_xticks(range(len(BUDGETS)))
ax.set_xticklabels([f"{TOTAL[b]} s" for b in BUDGETS])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels, fontsize=8.6)
for i in range(len(rows)):
    for j in range(len(BUDGETS)):
        ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center", fontsize=7.8,
                color="#1f2937")
ax.axhline(7.5, color="#111827", lw=1.4)
ax.set_title("full 相对 frozen 的 RMSE 改善（%），★ = 4 个跨状态留出条件（横线以下）", fontsize=11.0)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="改善百分比")
fig.tight_layout()
fig.savefig(HERE / "unified_state_heatmap.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote unified_state_heatmap.png")

# --------------------------------------------------------------- trade-off
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))
b = 30
for ax, group, title in zip(axes, ("adaptation", "heldout"), ("8 个适配状态", "4 个跨状态留出条件")):
    xs = []
    for arm in ARMS:
        sm = SUMMARY[b]["arms"][arm]["summary"][group]
        xs.append(sm["mean_tv_n_per_s"])
        ax.scatter(sm["mean_tv_n_per_s"], sm["mean_rmse_deg_s"], s=70, color=COLOR[arm],
                   label=LABEL[arm], zorder=3)
    lo, hi = min(xs), max(xs)
    for arm in ARMS:
        sm = SUMMARY[b]["arms"][arm]["summary"][group]
        right = sm["mean_tv_n_per_s"] <= lo + 0.55 * (hi - lo) if hi > lo else True
        ax.annotate(LABEL[arm], (sm["mean_tv_n_per_s"], sm["mean_rmse_deg_s"]),
                    textcoords="offset points", xytext=(8 if right else -8, 9),
                    ha="left" if right else "right", fontsize=8.0, color=COLOR[arm])
    ax.set_xlabel("平均控制活动 TV (N/s)")
    ax.set_ylabel("平均跟踪 RMSE (deg/s)")
    ax.set_title(f"{title} · B = 30 s/状态")
    ax.grid(alpha=0.3)
    ax.margins(x=0.30, y=0.22)
axes[0].legend(frameon=False, fontsize=8.0, loc="upper right")
fig.suptitle("误差与动作活动代价：训练降低误差，但 TV 上升（30 s 档，均值口径，不堆合计）", fontsize=12.4)
fig.tight_layout()
fig.savefig(HERE / "unified_tradeoff.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote unified_tradeoff.png")

# ------------------------------------------------------------- trajectories
def load_arm(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ft.DenseConditionalStudent(
        int(payload["observation_dim"]), int(payload["aircraft_parameter_dim"]),
        int(payload["action_dim"]), width=int(payload["network_width"]),
        residual_blocks=int(payload["residual_blocks"]),
        residual_scale=float(payload["residual_scale"]),
        enforce_odd_policy=bool(payload["enforce_odd_policy"]))
    model.load_state_dict(payload["model"])
    model.eval()
    return model

def one_command(batch, index):
    out = {k: (v[index:index + 1] if isinstance(v, np.ndarray) else v) for k, v in batch.items()}
    out["ids"] = [batch["ids"][index]]
    return out

def replay(budget: int, state: str, command_id: str):
    theta_by_state = {row["condition"]: row["normalized_theta"]
                      for row in THETA[budget]["adaptation"] + THETA[budget]["heldout"]}
    theta = torch.as_tensor(theta_by_state[state], dtype=torch.float64)
    board = EVAL["boards"][state]
    fit = ft.ArxFit(board["fit"]["na"], board["fit"]["nb"], board["fit"]["nk"],
                    tuple(board["fit"]["a"]), tuple(board["fit"]["b"]),
                    board["fit"]["vaf_percent"], board["fit"]["n_samples"])
    params = ft.arx_torch_parameters(fit.a, fit.b, fit.nk)
    batch = ft.command_batch_50hz(tuple(ft.regression_commands()) + tuple(ft.heldout_commands()))
    index = batch["ids"].index(command_id)
    single = one_command(batch, index)
    out = {}
    with torch.no_grad():
        for arm in ("frozen", "full"):
            model = load_arm(UNI / f"b{budget}" / "weights" / f"{arm}.pt")
            trace = ft.arx_rollout_torch(model, theta, single, params)
            p = np.degrees(trace["p"][0].numpy())
            ref = np.degrees(trace["ref"][0].numpy())
            out[arm] = {"p": p, "ref": ref,
                        "rmse": float(np.sqrt(np.mean((p - ref) ** 2)))}
    return out

cases = [("改善", 30, "V080_H2500"), ("退化", 15, "V080_H3500")]
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.5), sharey=True)
for ax, (kind, budget, state) in zip(axes, cases):
    traces = replay(budget, state, "eval-step-pos-15deg-s")
    ref = traces["frozen"]["ref"]
    t = (np.arange(ref.size) + 1) * ft.DT
    ax.plot(t, ref, "k--", lw=1.4, label="参考 $p_{ref}$")
    for arm in ("frozen", "full"):
        row = next(c for c in METRICS[budget]["arms"][arm]["conditions"][state]["commands"]
                   if c["command_id"] == "eval-step-pos-15deg-s")
        assert abs(row["rmse_deg_s"] - traces[arm]["rmse"]) < 1e-6, (budget, state, arm)
        ax.plot(t, traces[arm]["p"], lw=2.0, color=COLOR[arm],
                label=f"{LABEL[arm]}（RMSE {row['rmse_deg_s']:.3f}）")
    ax.set_title(f"B = {budget} s · {state} · +15 deg/s 阶跃（{kind}）")
    ax.set_xlabel("时间 (s)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8.2)
axes[0].set_ylabel("滚转角速度 p (deg/s)")
fig.suptitle("两例无梯度重放：一例改善（B30 / V080_H2500）、一例退化（B15 / V080_H3500）", fontsize=12.4)
fig.tight_layout()
fig.savefig(HERE / "unified_trajectories.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote unified_trajectories.png (success + regression, p_ref labelled)")
