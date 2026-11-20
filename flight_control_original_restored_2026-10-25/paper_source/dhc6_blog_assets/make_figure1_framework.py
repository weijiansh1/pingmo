"""Figure 1: method-first framework schematic.

    python dhc6_blog_assets/make_figure1_framework.py

Each card is titled by the method it uses (sampling / PPO / DAgger / fine-tune),
with a one-line mechanism under the title and a real data thumbnail inside.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets" / "pipeline_framework"
LIB = WS / "pingmo/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#16233a"
MUTED = "#16233a"
ACC = "#2f6fd0"
GRADE = {1: "#a9c8e8", 2: "#5b8ec4", 3: "#254b74"}
RED = "#c9453b"

fig = plt.figure(figsize=(10.4, 4.2))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

rows = [json.loads(l) for l in LIB.read_text(encoding="utf-8").splitlines() if l.strip()]
lfa = np.array([r["parameters"]["l_fa"] for r in rows])
om = np.array([r["parameters"]["omega_d"] for r in rows])
lvl = np.array([r["gjb_level"] for r in rows])

cards = [
    ("①", "品质分级采样", "GJB 1/2/3 级 + 分布外", "5,000 架 · 8 参数", 0.020),
    ("②", "PPO 教师训练", "actor–critic · 614 万步/架", "49 架 · 46 架通过门限", 0.265),
    ("③", "DAgger 蒸馏", "教师标注 → 学生回归", "3,856 参数 · 无特权输入", 0.510),
    ("④", "小样本微调", "40 步 Adam · lr 3e-5", "12 状态 · 2.91 → 0.59", 0.755),
]
cw, cy, ch = 0.222, 0.135, 0.585

thumbs = []
for badge, method, mech, spec, cx in cards:
    ax.add_patch(FancyBboxPatch((cx * 100 + 0.6, cy * 100), cw * 100 - 1.2, ch * 100,
                                boxstyle="round,pad=0,rounding_size=1.6",
                                facecolor="#f5f8fc", edgecolor="#b9c7da", lw=1.3, zorder=2))
    ax.text(cx * 100 + 3.0, cy * 100 + ch * 100 - 5.0, badge, fontsize=12.5, color=ACC,
            fontweight="bold", va="center", zorder=4)
    ax.text(cx * 100 + 7.2, cy * 100 + ch * 100 - 5.0, method, fontsize=12.0, color=INK,
            fontweight="bold", va="center", zorder=4)
    ax.text(cx * 100 + 3.0, cy * 100 + ch * 100 - 10.8, mech, fontsize=8.8, color=MUTED,
            va="center", zorder=4)
    ax.text(cx * 100 + 3.0, cy * 100 + 3.8, spec, fontsize=8.8, color=MUTED,
            va="center", zorder=4)
    thumbs.append(fig.add_axes([cx + 0.024, cy + 0.085, cw - 0.048, ch - 0.245],
                               facecolor="none"))

# ---------------- 1: parameter family
t1 = thumbs[0]
for lv in (1, 2, 3):
    m = lvl == lv
    t1.scatter(lfa[m], om[m], s=1.1, c=GRADE[lv], alpha=0.6, linewidths=0)
t1.set_xscale("log")
t1.set_yscale("log")
t1.set_xticks([])
t1.set_yticks([])
t1.tick_params(which="both", length=0)
for sp in t1.spines.values():
    sp.set_visible(False)

# ---------------- 2: PPO actor-critic loop
t2 = thumbs[1]
t2.set_xlim(0, 10)
t2.set_ylim(0, 10)
t2.axis("off")
t2.add_patch(FancyBboxPatch((0.2, 6.6), 1.5, 1.5, boxstyle="round,pad=0,rounding_size=0.2",
                            facecolor="#f3eefb", edgecolor="#9a83c8", lw=0.8))
t2.text(0.95, 7.35, "θ, z", ha="center", va="center", fontsize=5.6, color="#6b52a3")
t2.add_patch(FancyArrowPatch((1.8, 7.35), (2.0, 7.35), arrowstyle="-|>", mutation_scale=6,
                             lw=0.8, color="#9a83c8"))
t2.add_patch(FancyBboxPatch((2.1, 6.4), 2.9, 2.6, boxstyle="round,pad=0,rounding_size=0.3",
                            facecolor="#e8f0fb", edgecolor="#8fb0d8", lw=0.9))
t2.text(3.55, 7.7, "策略 π", ha="center", va="center", fontsize=6.4, color=INK,
        fontweight="bold")
t2.add_patch(FancyBboxPatch((6.3, 6.4), 3.4, 2.6, boxstyle="round,pad=0,rounding_size=0.3",
                            facecolor="#eaf5ee", edgecolor="#7fae92", lw=0.9))
t2.text(8.0, 7.7, "环境 Gθ", ha="center", va="center", fontsize=6.4, color=INK,
        fontweight="bold")
t2.add_patch(FancyArrowPatch((5.1, 8.5), (6.2, 8.5), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color=INK))
t2.text(5.65, 8.85, "a", ha="center", fontsize=6.0, color=MUTED)
t2.add_patch(FancyArrowPatch((6.2, 7.1), (5.1, 7.1), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color=INK))
t2.text(5.65, 6.55, "o, r", ha="center", fontsize=6.0, color=MUTED)
t2.add_patch(FancyBboxPatch((0.6, 2.0), 3.0, 2.0, boxstyle="round,pad=0,rounding_size=0.25",
                            facecolor="#fdf1e3", edgecolor="#d8b46a", lw=0.9))
t2.text(2.1, 3.0, "critic V", ha="center", va="center", fontsize=6.2, color=INK)
t2.add_patch(FancyBboxPatch((5.2, 2.0), 3.6, 2.0, boxstyle="round,pad=0,rounding_size=0.25",
                            facecolor="#fdf1e3", edgecolor="#d8b46a", lw=0.9))
t2.text(7.0, 3.0, "GAE + clip", ha="center", va="center", fontsize=6.2, color=INK)
t2.add_patch(FancyArrowPatch((3.7, 3.0), (5.1, 3.0), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color=INK))
t2.add_patch(FancyArrowPatch((7.0, 4.1), (3.4, 6.3), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color="#5b6c85", connectionstyle="arc3,rad=0.15"))

# ---------------- 3: DAgger loop
t3 = thumbs[2]
t3.set_xlim(0, 10)
t3.set_ylim(0, 10)
t3.axis("off")
t3.add_patch(FancyBboxPatch((0.3, 6.3), 3.4, 2.7, boxstyle="round,pad=0,rounding_size=0.3",
                            facecolor="#f3eefb", edgecolor="#9a83c8", lw=0.9))
t3.text(2.0, 7.65, "教师 μT (π)", ha="center", va="center", fontsize=6.0, color="#6b52a3")
t3.add_patch(FancyBboxPatch((6.3, 6.3), 3.4, 2.7, boxstyle="round,pad=0,rounding_size=0.3",
                            facecolor="#e8f0fb", edgecolor="#8fb0d8", lw=0.9))
t3.text(8.0, 7.65, "学生 μS", ha="center", va="center", fontsize=6.0, color=INK)
t3.add_patch(FancyArrowPatch((3.8, 8.45), (6.2, 8.45), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color=INK))
t3.text(5.0, 8.7, "标注 a*", ha="center", fontsize=5.8, color=MUTED)
t3.add_patch(FancyArrowPatch((6.2, 6.9), (3.8, 6.9), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color=INK, ls=(0, (3, 2))))

t3.add_patch(FancyBboxPatch((2.7, 2.0), 4.6, 2.5, boxstyle="round,pad=0,rounding_size=0.25",
                            facecolor="white", edgecolor="#b9c7da", lw=0.9))
t3.text(5.0, 3.25, "D_k 聚合样本", ha="center", va="center", fontsize=6.0, color=INK)
t3.add_patch(FancyArrowPatch((2.6, 5.9), (3.6, 4.7), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color="#5b6c85"))
t3.add_patch(FancyArrowPatch((7.4, 5.9), (6.4, 4.7), arrowstyle="-|>", mutation_scale=7,
                             lw=0.9, color="#5b6c85"))
t3.text(5.0, 1.5, "×6 轮聚合", ha="center", fontsize=5.8, color=MUTED)

# ---------------- 4: before / after
t4 = thumbs[3]
t4.set_xlim(0, 10)
t4.set_ylim(0, 10)
t4.axis("off")
for i, (lab, v, col) in enumerate([("前", 2.91, "#c9d4e3"), ("后", 0.59, ACC)]):
    w = v / 2.91 * 5.0
    y = 6.6 - i * 3.4
    t4.add_patch(Rectangle((1.7, y - 1.0), w, 2.0, facecolor=col, edgecolor="none"))
    t4.text(1.5, y, lab, ha="right", va="center", fontsize=6.6, color=MUTED)
    t4.text(1.7 + w + 0.2, y, f"{v:.2f}", ha="left", va="center", fontsize=6.8,
            color=INK, fontweight="bold")
t4.add_patch(Polygon([(9.76, 8.6), (8.41, 9.094999999999999),
                      (8.63, 8.6), (8.41, 8.105)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=5))
t4.add_patch(Polygon([(8.63, 8.82), (7.69, 9.46),
                      (8.13, 8.6), (7.69, 7.74),
                      (8.63, 8.38)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=5))
t4.text(8.9, 7.10, "DHC-6", ha="center", fontsize=5.8, color=MUTED, zorder=5)

# ---------------- flow arrows
for lab, x0, x1 in [("采样", 21.9, 26.9), ("蒸馏", 48.0, 53.0), ("适配", 74.1, 79.1)]:
    ax.add_patch(FancyArrowPatch((x0, 42.6), (x1, 42.6), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.6, color=INK, zorder=6))
    ax.text((x0 + x1) / 2, 46.8, lab, ha="center", va="bottom", fontsize=8.4,
            color=MUTED, zorder=6)

# ---------------- headline
ax.text(50, 93.2, "从对象族到目标机型：四步方法", ha="center",
        va="center", fontsize=13.0, color=INK, fontweight="bold")
cascade = [("5,000", "对象"), ("49", "教师"), ("1", "学生"), ("12", "配平状态")]
cx0 = 30.0
for i, (num, lab) in enumerate(cascade):
    x = cx0 + i * 13.5
    ax.text(x, 86.6, num, ha="center", va="center", fontsize=17, color=ACC if i == 2 else INK,
            fontweight="bold")
    ax.text(x, 82.4, lab, ha="center", va="center", fontsize=8.2, color=MUTED)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + 3.4, 85.4), (x + 9.6, 85.4), arrowstyle="-|>",
                                     mutation_scale=10, lw=1.1, color="#5b6c85"))

fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
print("wrote", OUT.with_suffix(".svg"))
