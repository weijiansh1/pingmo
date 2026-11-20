"""Hero pipeline figure (graphical-abstract style).

    python dhc6_blog_assets/make_pipeline_hero.py

Four icons, thick flow arrows, big numerals, almost no prose.  Vector output.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets" / "pipeline_framework"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#16233a"
MUTED = "#16233a"
L1, L2, L3 = "#7aa7d9", "#3f6fa8", "#1f3f66"
ACcent = "#e0761f"
RED = "#d8453a"
GRID = "#e7ecf4"

fig, ax = plt.subplots(figsize=(10.2, 3.05))
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.axis("off")

centers = [12, 37.5, 63, 88]
y_icon = 17.5

# ---------------- 1. object family: coloured dots
rng = np.random.default_rng(7)
ax.add_patch(FancyBboxPatch((centers[0] - 7.6, y_icon - 5.6), 15.2, 11.2,
                            boxstyle="round,pad=0,rounding_size=0.7",
                            facecolor="#f7f9fd", edgecolor=GRID, lw=1.0))
for i in range(64):
    cx = centers[0] - 6.4 + rng.uniform(0, 12.8)
    cy = y_icon - 4.6 + rng.uniform(0, 9.2)
    ax.add_patch(Circle((cx, cy), 0.34, color=rng.choice([L1, L2, L3]),
                        ec="white", lw=0.5, zorder=3))

# ---------------- 2. teachers: 7x7 grid, three failed
ax.add_patch(FancyBboxPatch((centers[1] - 6.4, y_icon - 5.6), 12.8, 11.2,
                            boxstyle="round,pad=0,rounding_size=0.7",
                            facecolor="#f7f9fd", edgecolor=GRID, lw=1.0))
k = 0
for r in range(7):
    for c in range(7):
        if k >= 49:
            break
        col = RED if k in (17, 23, 46) else "#2f5d8f"
        ax.add_patch(Rectangle((centers[1] - 5.1 + c * 1.55, y_icon + 4.0 - r * 1.55),
                               1.05, 1.05, facecolor=col, edgecolor="none", zorder=3))
        k += 1

# ---------------- 3. one student: a single cell
ax.add_patch(FancyBboxPatch((centers[2] - 6.4, y_icon - 5.6), 12.8, 11.2,
                            boxstyle="round,pad=0,rounding_size=0.7",
                            facecolor="#f7f9fd", edgecolor=GRID, lw=1.0))
ax.add_patch(Rectangle((centers[2] - 2.1, y_icon - 2.1), 4.2, 4.2,
                       facecolor=INK, edgecolor="none", zorder=3))

# ---------------- 4. aircraft: DHC-6 plan view (straight wing, two nacelles)
ac_x, ac_y = centers[3], y_icon
ax.add_patch(FancyBboxPatch((ac_x - 0.62, ac_y - 4.9), 1.24, 8.3,
                            boxstyle="round,pad=0,rounding_size=0.6",
                            facecolor=INK, edgecolor="none", zorder=4))
ax.add_patch(Polygon([(ac_x - 0.55, ac_y + 1.55), (ac_x - 7.2, ac_y + 1.55),
                      (ac_x - 7.2, ac_y + 2.55), (ac_x - 0.55, ac_y + 2.35)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=3))
ax.add_patch(Polygon([(ac_x + 0.55, ac_y + 1.55), (ac_x + 7.2, ac_y + 1.55),
                      (ac_x + 7.2, ac_y + 2.55), (ac_x + 0.55, ac_y + 2.35)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=3))
for sx in (-3.4, 2.3):
    ax.add_patch(FancyBboxPatch((ac_x + sx, ac_y + 0.85), 1.05, 1.55,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                facecolor="white", edgecolor=INK, lw=1.1, zorder=5))
ax.add_patch(Polygon([(ac_x - 0.45, ac_y + 1.2), (ac_x - 2.6, ac_y - 3.4),
                      (ac_x - 2.6, ac_y - 2.5), (ac_x - 0.45, ac_y + 0.5)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=3))
ax.add_patch(Polygon([(ac_x + 0.45, ac_y + 1.2), (ac_x + 2.6, ac_y - 3.4),
                      (ac_x + 2.6, ac_y - 2.5), (ac_x + 0.45, ac_y + 0.5)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=3))
ax.add_patch(Polygon([(ac_x - 0.2, ac_y - 3.5), (ac_x - 1.5, ac_y - 4.6),
                      (ac_x - 1.5, ac_y - 4.0), (ac_x - 0.2, ac_y - 2.9)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=3))
ax.add_patch(Polygon([(ac_x + 0.2, ac_y - 3.5), (ac_x + 1.5, ac_y - 4.6),
                      (ac_x + 1.5, ac_y - 4.0), (ac_x + 0.2, ac_y - 2.9)],
                     closed=True, facecolor=INK, edgecolor="none", zorder=3))

# ---------------- flow arrows
for a, b in zip(centers, centers[1:]):
    ax.add_patch(FancyArrowPatch((a + 8.4, y_icon), (b - 8.0, y_icon),
                                 arrowstyle="-|>", mutation_scale=22, lw=2.6,
                                 color=INK, shrinkA=0, shrinkB=0, zorder=1))

# ---------------- numerals + labels
heads = ["5,000", "49 → 46", "3,856", "12 状态"]
subs = ["品质分级对象", "逐对象教师", "一个统一学生", "DHC-6 适配"]
notes = ["GJB 1/2/3 级 + OOD\n8 参数低阶等效模型",
         "PPO，4.26 M 参数\n46 架通过门限",
         "CPU 单核可推理\n同分布 1.28×",
         "40 步微调\n误差 2.91 → 0.59"]
for i, (c, h, s_, n) in enumerate(zip(centers, heads, subs, notes)):
    ax.text(c, 28.0, h, ha="center", va="center", fontsize=17 if i != 1 else 15.5,
            color=ACcent if i == 3 else INK, fontweight="bold")
    ax.text(c, 7.2, s_, ha="center", va="center", fontsize=10.2, color=INK)
    ax.text(c, 2.6, n, ha="center", va="center", fontsize=7.4, color=MUTED,
            linespacing=1.5)

ax.text(50, -1.6, "", fontsize=1)
fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
print("wrote", OUT.with_suffix(".svg"))
