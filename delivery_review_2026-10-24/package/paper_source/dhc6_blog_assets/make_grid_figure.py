"""One student, three plants, three commands (3x3 grid).

    python dhc6_blog_assets/make_grid_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

WS = Path(__file__).resolve().parents[1]
D = json.loads((WS / "_report_review/remote/student_grid.json").read_text(encoding="utf-8"))
OUT = WS / "dhc6_blog_assets"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#1b2a41"
MUTED = "#16233a"
GRID = "#e6ebf3"
REF = "#5b6c85"
STUD = "#e08a1e"

CMD_IDX = [2, 5, 27]
plants = D["plants"]
t = D["t"]

fig, axes = plt.subplots(3, 3, figsize=(8.6, 5.4), sharex=True, sharey="row")
for row, plant in enumerate(plants):
    for col, idx in enumerate(CMD_IDX):
        ax = axes[row, col]
        d = D["grid"][f"{plant['id']}|{idx}"]
        ax.plot(t, d["ref"], "--", color=REF, lw=1.0, label="参考")
        ax.plot(t, d["p"], color=STUD, lw=1.2, label="学生")
        if row == 0:
            ax.set_title(D["commands"][col], fontsize=8.6, color=INK,
                         fontweight="bold", pad=4)
        if col == 0:
            ax.set_ylabel(f"{plant['label']}\np (deg/s)", fontsize=7.6, color=MUTED)
        ax.set_xlim(0, 8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#5b6c85")
        ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
for ax in axes[2]:
    ax.set_xlabel("时间 (s)", fontsize=7.4, color=MUTED)
axes[0, 0].legend(fontsize=6.6, frameon=False, loc="lower right")
fig.tight_layout(pad=0.5, h_pad=0.9, w_pad=0.6)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"student_grid.{ext}", facecolor="white",
                dpi=150 if ext == "png" else None)
print("wrote student_grid.svg")
