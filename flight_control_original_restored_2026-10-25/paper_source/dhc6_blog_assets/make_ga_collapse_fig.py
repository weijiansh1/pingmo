"""Render the collapse graphical abstract from ga_collapse.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

WS = Path(__file__).resolve().parents[1]
D = json.loads((WS / "_report_review/remote/ga_collapse.json").read_text(encoding="utf-8"))
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
GRID = "#e7ecf4"
GRADE = {1: "#8fb6de", 2: "#3f6fa8", 3: "#1b3a5c"}
ACC = "#e0761f"

t = np.array(D["t"])
fig = plt.figure(figsize=(10.4, 3.5))
ax1 = fig.add_axes([0.055, 0.18, 0.30, 0.66])
ax2 = fig.add_axes([0.655, 0.18, 0.325, 0.66])

TITLES = {"open": "(a) 同一力输入：响应分散",
          "closed": "(b) 同一策略：全部收到指令附近"}
for ax, key in ((ax1, "open"), (ax2, "closed")):
    title = TITLES[key]
    for pl in D["plants"]:
        y = np.array(pl[key])
        ax.plot(t, y, lw=0.8, color=GRADE[int(pl["level"])], alpha=0.55)
    ax.set_xlim(0, 8)
    ax.set_ylim(-5, 200)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#5b6c85")
    ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
    ax.set_title(title, fontsize=9.6, color=INK, fontweight="bold", loc="left", pad=5)
    ax.set_xlabel("时间 (s)", fontsize=7.6, color=MUTED, labelpad=1)

ax1.set_ylabel("滚转角速度 (deg/s)", fontsize=7.6, color=MUTED, labelpad=2)
ax2.set_ylabel("")

ref = np.array(D["ref"])
for ax in (ax1, ax2):
    ax.axhline(21, color="#5b6c85", lw=1.1, ls="--", zorder=6)
ax1.text(4.2, 28, "指令 21 deg/s", fontsize=7.0, color="#6b7a90")
ax2.text(4.2, 28, "指令 21 deg/s", fontsize=7.0, color="#6b7a90")

# middle: the compression arrow + number
axm = fig.add_axes([0.355, 0.18, 0.27, 0.66])
axm.set_xlim(0, 10)
axm.set_ylim(0, 10)
axm.axis("off")
axm.add_patch(FancyArrowPatch((0.6, 5.0), (9.4, 5.0), arrowstyle="-|>", mutation_scale=26,
                              lw=3.0, color=INK, shrinkA=0, shrinkB=0))
axm.text(5.0, 6.4, "3,856", ha="center", fontsize=25, color=INK, fontweight="bold")
axm.text(5.0, 3.2, "参数 · 单核 CPU", ha="center", fontsize=9.0, color=MUTED)
axm.text(5.0, 1.2, "一个策略覆盖整族", ha="center", fontsize=10.4, color=INK,
         fontweight="bold")

fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
print("wrote", OUT.with_suffix(".svg"))
