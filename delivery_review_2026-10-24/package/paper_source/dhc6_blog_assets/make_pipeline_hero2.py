"""Graphical abstract: many quality-graded aircraft -> one 3,856-parameter chip -> one aircraft.

    python dhc6_blog_assets/make_pipeline_hero2.py

Iconography + one sparkline, no result panels, no axes, no legends.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets" / "pipeline_framework"
TRACES = WS / "_report_review/remote/student_small_traces.json"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#16233a"
MUTED = "#16233a"
GRADE = ["#a9c8e8", "#5b8ec4", "#254b74"]
ACC = "#e0761f"


def plane(ax, x, y, s, color, z=3, alpha=1.0):
    """Tiny plan-view aircraft pointing right."""
    ax.add_patch(Polygon([(x + 0.95 * s, y), (x - 0.15 * s, y + 0.30 * s),
                          (x - 0.05 * s, y), (x - 0.15 * s, y - 0.30 * s)],
                         closed=True, facecolor=color, edgecolor="none", zorder=z,
                         alpha=alpha))
    ax.add_patch(Polygon([(x - 0.05 * s, y + 0.12 * s), (x - 0.60 * s, y + 0.62 * s),
                          (x - 0.35 * s, y + 0.10 * s)],
                         closed=True, facecolor=color, edgecolor="none", zorder=z,
                         alpha=alpha))
    ax.add_patch(Polygon([(x - 0.05 * s, y - 0.12 * s), (x - 0.60 * s, y - 0.62 * s),
                          (x - 0.35 * s, y - 0.10 * s)],
                         closed=True, facecolor=color, edgecolor="none", zorder=z,
                         alpha=alpha))
    ax.add_patch(Polygon([(x - 0.55 * s, y), (x - 0.85 * s, y + 0.28 * s),
                          (x - 0.72 * s, y), (x - 0.85 * s, y - 0.28 * s)],
                         closed=True, facecolor=color, edgecolor="none", zorder=z,
                         alpha=alpha))


fig, ax = plt.subplots(figsize=(10.4, 3.2))
ax.set_xlim(0, 104)
ax.set_ylim(0, 32)
ax.axis("off")

# ---------------- left: the graded family
rng = np.random.default_rng(3)
xs = [7.0, 11.2, 15.4, 19.6, 23.8]
for ci, cx in enumerate(xs):
    for r in range(9):
        cy = 7.0 + r * 2.15 + rng.uniform(-0.3, 0.3)
        col = GRADE[(r + ci) % 3]
        plane(ax, cx + rng.uniform(-0.5, 0.5), cy, 1.5, col, alpha=0.95)

# faint label
ax.text(17.5, 30.2, "5,000", ha="center", va="center", fontsize=22, color=INK,
        fontweight="bold")
ax.text(17.5, 27.4, "品质分级对象", ha="center", va="center", fontsize=9.6, color=MUTED)

# ---------------- converging flow
chip_x, chip_w = 42.0, 15.0
for cy in (11.0, 14.5, 18.0, 21.5, 25.0):
    ax.plot([26.6, chip_x], [cy, 16.0], color="#c2cfe0", lw=1.0, zorder=1)
ax.add_patch(FancyArrowPatch((30.0, 16.0), (chip_x - 0.8, 16.0), arrowstyle="-|>",
                             mutation_scale=14, lw=1.6, color="#5b6c85", zorder=2))

# ---------------- middle: the chip
chip_y, chip_h = 10.6, 10.8
ax.add_patch(FancyBboxPatch((chip_x, chip_y), chip_w, chip_h,
                            boxstyle="round,pad=0,rounding_size=1.6",
                            facecolor=INK, edgecolor="none", zorder=4))
ax.text(chip_x + chip_w / 2, chip_y + 6.7, "3,856", ha="center", va="center",
        fontsize=19, color="white", fontweight="bold", zorder=5)
ax.text(chip_x + chip_w / 2, chip_y + 3.3, "参数 · 单核 CPU", ha="center", va="center",
        fontsize=8.6, color="#c9d4e3", zorder=5)
ax.text(chip_x + chip_w / 2, chip_y + 1.5, "一个策略", ha="center", va="center",
        fontsize=8.0, color="#5b6c85", zorder=5)

# ---------------- right: one aircraft + sparkline
ax.add_patch(FancyArrowPatch((chip_x + chip_w + 1.0, 16.0), (chip_x + chip_w + 5.4, 16.0),
                             arrowstyle="-|>", mutation_scale=18, lw=2.4, color=INK,
                             shrinkA=0, shrinkB=0, zorder=3))
ax.text((chip_x + chip_w + 1.0 + chip_x + chip_w + 5.4) / 2, 13.2, "迁移", ha="center",
        fontsize=8.0, color=MUTED)

plane(ax, chip_x + chip_w + 15.6, 12.2, 5.0, INK, z=4)
ax.text(chip_x + chip_w + 17.4, 8.0, "DHC-6 仿真", ha="center", fontsize=8.6,
        color=MUTED)

# sparkline: reference vs response for a 21 deg/s step
d = json.loads(TRACES.read_text(encoding="utf-8"))["series"]["±21"]
t = np.array(d["t"]); ref = np.array(d["ref"]); stu = np.array(d["p"])
sel = (t >= 1.0) & (t <= 4.6)
tx = chip_x + chip_w + 6.4 + (t[sel] - 1.0) / 3.6 * 19.0
def ymap(v):
    return 17.8 + v / 26.0 * 4.4
ax.plot(tx, ymap(ref[sel]), "--", color="#5b6c85", lw=1.4, zorder=5)
ax.plot(tx, ymap(stu[sel]), "-", color=ACC, lw=2.0, zorder=6)

ax.text(chip_x + chip_w + 14.0, 22.9, "阶跃跟踪（橙）与参考（灰）", ha="center", fontsize=7.2, color=MUTED)
ax.text(52.0, 5.0, "同一权重，换一架飞机仍能跟踪", ha="center", fontsize=10.4,
        color=INK, fontweight="bold")
ax.text(52.0, 2.2, "适配状态中位误差 2.91 → 0.59 deg/s", ha="center", fontsize=8.6,
        color=MUTED)

fig.subplots_adjust(left=0.004, right=0.996, top=0.995, bottom=0.005)
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
print("wrote", OUT.with_suffix(".svg"))
