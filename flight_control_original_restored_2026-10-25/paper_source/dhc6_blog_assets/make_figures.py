"""Persistent figure generator for the blog page (regenerate with:
    python dhc6_blog_assets/make_figures.py
).

Figures produced (all into this directory):
  four_steps.png            4-step research mainline
  mapping_scenarios.png     archived scenario outputs (mapping A / B)
  quality_bounds.png        exploratory comparison with digitised bounds
  stage3_frozen_summary.png stage-3 frozen holdout snapshot
  simo_exploration.png      single-input multi-output subspace exploration

Correctness notes encoded here:
  * quality *sensitivity* (A31) is deliberately NOT plotted: the archived
    quantity is a 0.1 Hz frequency-response magnitude scaled by 20/22, which is
    not the 1 s bank-angle/force sensitivity the A31 bound is defined for.
  * zeta_d (dimensionless) and zeta_d*omega_d (rad/s) get separate axes.
  * the SIMO figure is labelled as an exploratory subspace trial, not MIMO.
Reads existing result JSON; writes only PNGs in this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
WS = HERE.parent
matplotlib.rcParams["axes.unicode_minus"] = False
for _c in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
    if _c in {f.name for f in font_manager.fontManager.ttflist}:
        matplotlib.rcParams["font.family"] = _c
        break

STATES = json.loads((WS / "dhc6_states_results.json").read_text(encoding="utf-8"))
QUALITY = json.loads((WS / "dhc6_quality_results.json").read_text(encoding="utf-8"))
ARX = json.loads((WS / "dhc6_arx_all.json").read_text(encoding="utf-8"))
SIMO = json.loads((WS / "dhc6_mimo_models.json").read_text(encoding="utf-8"))
ROWS = STATES["rows"]
LABELS = [r["condition"].replace("V0", "V").replace("_H", "/H") for r in ROWS]

# ---------------------------------------------------------------- four steps
fig, ax = plt.subplots(figsize=(15.6, 6.4))
ax.set_xlim(0, 15.8)
ax.set_ylim(0, 6.5)
ax.axis("off")
steps = [
    ("① 随机取样", "9 维 Sobol 候选\n→ 目标品质区域\n→ 1 s 滚转角灵敏度\n标定 L_Fa\n"
                   "→ 重算品质标签\n→ 配额选入", "#dbeafe", "#1d4ed8"),
    ("② 教师模型", "输入：单架对象\n固定参数\n方法：纯奖励 TD3\n（Actor 不含 θ）\n"
                   "输出：32 个\n合格 Teacher", "#e0e7ff", "#4338ca"),
    ("③ 学生蒸馏", "输入：Teacher 动作\n+ 学生自访问状态\n方法：Teacher 驱动\n"
                   "+ Student 驱动 DAgger\n输出：540,417 参数\n策略 π(o, θ~)", "#dcfce7", "#15803d"),
    ("④ 小样本迁移（已执行）", "输入：W0 + DHC-6 适配窗\n8 状态 × 15/30/60 s\n方法：模型辅助权重微调\n"
                             "→ 统一学生 W_DHC6\n评测：4 留出状态 + 12 状态\n3 档预算 / 每臂 64 步", "#fef3c7", "#b45309"),
]
w, h, gap = 3.45, 3.35, 0.40
for i, (title, detail, face, edge) in enumerate(steps):
    x = 0.35 + i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 2.40), w, h,
                                boxstyle="round,pad=0.10,rounding_size=0.14",
                                facecolor=face, edgecolor=edge, linewidth=1.8))
    ax.text(x + w / 2, 5.25, title, ha="center", va="center", fontsize=14,
            fontweight="bold", color=edge)
    ax.text(x + w / 2, 3.68, detail, ha="center", va="center", fontsize=9.0,
            color="#26364f", linespacing=1.72)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + w + 0.02, 4.07), (x + w + gap - 0.02, 4.07),
                                     arrowstyle="-|>", mutation_scale=20,
                                     linewidth=2.0, color="#94a3b8"))
ax.text(0.35, 6.15, "品模控制器的构建与适配（离线）：四步训练与迁移主线", fontsize=17,
        fontweight="bold", color="#132a52")
ax.text(0.35, 1.45, "①②③ 已有冻结产物（对象库 / Teacher Bank / Student 权重）；④ 已执行：单个统一条件学生，"
                    "8 状态适配、4 状态跨条件留出、12 状态评测（真实目标 135/240/480 s，每臂 64 步，单 seed）。",
        fontsize=10.4, color="#61708c", va="center")
ax.text(0.35, 0.80, "每个对象上的教师是独立训练出来的专用策略；学生是读取对象参数条件的统一策略——"
                    "两阶段不是单一可微损失管线。",
        fontsize=10.5, color="#a16207", va="center")
fig.savefig(HERE / "four_steps.png", dpi=140, bbox_inches="tight",
            pad_inches=0.22, facecolor="white")
plt.close(fig)
print("wrote four_steps.png")

# ------------------------------------------------------- mapping scenarios
fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.4))
x = np.arange(len(ROWS))
width = 0.27
ax = axes[0]
ax.bar(x - width, [r["raw_rmse"] for r in ROWS], width, label="开环基线（映射 A）",
       color="#9ca3af")
ax.bar(x, [r["strict_rmse"] for r in ROWS], width,
       label="冻结 Student · 映射 A（22 N 对 10°，条件输入 L_Fa 由标定得到）",
       color="#2563eb")
ax.bar(x + width, [r["fair_rmse"] for r in ROWS], width,
       label="冻结 Student · 映射 B（22 N 对 20°，且条件输入 L_Fa × 2）",
       color="#16a34a")
ax.set_xticks(x, LABELS, rotation=90, fontsize=7)
ax.set_ylabel("平均 RMSE（deg/s，6 条留出指令）")
ax.set_title("(a) 归档输出：开环基线与两种映射配置", fontsize=11)
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=8, loc="upper right")
ax.set_ylim(0, max(r["raw_rmse"] for r in ROWS) * 1.30)

ax = axes[1]
auth = [r["authority_deg_s"] for r in ROWS]
ax.plot(auth, [r["strict_rmse"] for r in ROWS], "o-", color="#2563eb", label="映射 A")
ax.plot(auth, [r["fair_rmse"] for r in ROWS], "s--", color="#16a34a", label="映射 B")
ax.annotate("映射 B 最低点在 V70 附近，\nV80 又回升（非单调）",
            xy=(21.3, 0.55), xytext=(15.0, 1.55), fontsize=9, color="#15803d",
            arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.1))
ax.annotate("映射 A 随估计权限增加而下降",
            xy=(25.6, 1.58), xytext=(19.0, 3.15), fontsize=9, color="#1d4ed8",
            arrowprops=dict(arrowstyle="->", color="#1d4ed8", lw=1.1))
ax.set_xlabel("10° 副翼下的稳态滚转率（deg/s，由拟合低频增益推出）")
ax.set_ylabel("平均 RMSE（deg/s）")
ax.set_title("(b) 观测方向：A 单调下降，B 非单调", fontsize=11)
ax.grid(alpha=0.3)
ax.legend(fontsize=9)
fig.suptitle("映射配置的敏感性（两个配置改动的量不同，B 没有匹配的开环基线）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(HERE / "mapping_scenarios.png", dpi=140, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote mapping_scenarios.png")

# --------------------------------------------------------- quality bounds
fig, axes = plt.subplots(1, 3, figsize=(16, 4.9))
x = np.arange(len(QUALITY))
labels_q = [r["condition"].replace("V0", "V").replace("_H", "/H") for r in QUALITY]

ax = axes[0]
ax.bar(x, [r["T_R"] for r in QUALITY], color="#2563eb")
ax.axhline(1.0, color="#b91c1c", ls="--", lw=1.2, label="所选边界 L1 上限 1.0 s")
ax.set_ylabel("T_R（s）")
ax.set_title("(a) 滚转模态时间常数\n（p 通道拟合）", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.bar(x, [r["zeta_d"] for r in QUALITY], color="#7c3aed")
ax.axhline(0.08, color="#b91c1c", ls="--", lw=1.2, label="所选边界 L1 下限 0.08")
ax.set_ylabel("ζ_d（无量纲）")
ax.set_title("(b) 荷兰滚阻尼比\n（β 通道拟合，仅供该探索性比较）", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
ax.bar(x, [r["product"] for r in QUALITY], color="#f59e0b")
ax.axhline(0.15, color="#b91c1c", ls="--", lw=1.2, label="所选边界 L1 下限 0.15")
ax.set_ylabel("ζ_d·ω_d（rad/s）")
ax.set_title("(c) 荷兰滚阻尼×频率\n（β 通道拟合，仅供该探索性比较）", fontsize=10)
ax.legend(fontsize=8)

for a in axes:
    a.set_xticks(x, labels_q, rotation=90, fontsize=6.5)
    a.grid(alpha=0.3, axis="y")
fig.suptitle("与所选数字化品质边界的探索性比较（非型号符合性评定；不含 A31 灵敏度，"
             "因为归档量与 A31 定义不同）", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(HERE / "quality_bounds.png", dpi=140, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote quality_bounds.png")

# ------------------------------------------------- stage-3 frozen summary
versions = ["v3\nStudent-driven", "v4\n稳定性蒸馏", "v4 + v5\n11 N/s 限速"]
panels = [
    ([1.0788, 0.9457, 1.0357], "验证整机 平均 RMSE（deg/s）", "#2563eb", None, "{:.4f}"),
    ([22.9062, 4.7569, 7.9847], "验证整机 最大峰值误差（deg/s）", "#f59e0b", 5.0, "{:.2f}"),
    ([1.816, 2.133, 0.574], "Student / Teacher 请求力总变差比", "#16a34a", 1.25, "{:.3f}"),
]
fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
for ax, (values, title, color, gate, fmt) in zip(axes, panels):
    bars = ax.bar(versions, values, color=color)
    top = max(values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.045, fmt.format(v),
                ha="center", fontsize=10)
    ax.set_ylim(0, top * 1.32)
    if gate is not None:
        ax.axhline(gate, color="#b91c1c", ls="--", lw=1.2, label=f"门限 {gate:g}")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title(title, fontsize=10, pad=10)
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(labelsize=8.5)
    ax.margins(x=0.12)
fig.suptitle("第 ③ 步冻结结果快照（6 架验证整机，36 个飞机-指令对）", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.91])
fig.savefig(HERE / "stage3_frozen_summary.png", dpi=140, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote stage3_frozen_summary.png")

# --------------------------------------------------- SIMO subspace trial
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9))
conds = list(SIMO.keys())
siso_vaf = [ARX[c]["p"]["vaf"] for c in conds]
simo_vaf = [SIMO[c]["vaf"] for c in conds]
xc = np.arange(len(conds))
w2 = 0.38
ax = axes[0]
ax.bar(xc - w2 / 2, siso_vaf, w2, label="单通道 ARX（副翼 → p）", color="#2563eb")
ax.bar(xc + w2 / 2, simo_vaf, w2, label="三输出子空间模型（副翼 → p, r, β）",
       color="#f59e0b")
ax.set_ylabel("模拟误差 VAF（%）")
ax.set_ylim(0, 112)
ax.set_xticks(xc, [c.replace("V0", "V").replace("_H", "/H") for c in conds],
              rotation=90, fontsize=6.5)
ax.set_title("(a) 拟合优度：三输出模型明显更低", fontsize=10.5)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.3, axis="y")

ax = axes[1]
ts = [SIMO[c]["T_s"] for c in conds]
ax.bar(xc, ts, color="#7c3aed")
ax.set_ylabel("螺旋模态时间常数估计（s）")
ax.set_xticks(xc, [c.replace("V0", "V").replace("_H", "/H") for c in conds],
              rotation=90, fontsize=6.5)
ax.set_title(f"(b) 螺旋模态估计分散（{min(ts):.1f}–{max(ts):.1f} s）→ 不可靠",
             fontsize=10.5)
ax.grid(alpha=0.3, axis="y")
fig.suptitle("单输入多输出子空间探索（探索性；拟合度低，未用于闭环结论）", fontsize=12.5)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(HERE / "simo_exploration.png", dpi=140, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("wrote simo_exploration.png")
print("assets:", sorted(p.name for p in HERE.glob("*.png")))
