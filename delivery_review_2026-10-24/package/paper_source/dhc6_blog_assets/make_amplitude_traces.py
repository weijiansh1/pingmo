"""Student response across amplitudes (incl. small steps) — 2x4 grid."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

WS = Path(__file__).resolve().parents[1]
D = json.loads((WS / "_report_review/remote/student_small_traces.json").read_text(encoding="utf-8"))
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

labels = ["±1", "±2", "±5", "±10", "±21", "±31", "双脉冲", "多正弦"]
fig, axes = plt.subplots(2, 4, figsize=(8.6, 3.5))
for ax, label in zip(axes.ravel(), labels):
    d = D["series"][label]
    ax.plot(d["t"], d["ref"], "--", color=REF, lw=1.0, label="参考")
    ax.plot(d["t"], d["p"], color=STUD, lw=1.2, label="学生")
    ax.set_title(label, fontsize=8.2, color=INK, fontweight="bold", pad=3)
    ax.set_xlim(0, 8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#5b6c85")
    ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
for ax in axes[1]:
    ax.set_xlabel("时间 (s)", fontsize=7.2, color=MUTED)
for ax in axes[:, 0]:
    ax.set_ylabel("p (deg/s)", fontsize=7.2, color=MUTED)
axes[0, 0].legend(fontsize=6.4, frameon=False, loc="lower right")
fig.tight_layout(pad=0.5, h_pad=0.9, w_pad=0.6)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"student_amplitudes_traces.{ext}", facecolor="white",
                dpi=150 if ext == "png" else None)
print("wrote student_amplitudes_traces.svg")
