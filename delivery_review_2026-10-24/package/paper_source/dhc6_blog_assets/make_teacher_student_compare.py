"""Teacher vs student: same plant, same commands, one figure.

    python dhc6_blog_assets/make_teacher_student_compare.py

Reads _report_review/remote/traces_multi.json (plant train_core-0421, 30 commands).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets"
D = json.loads((WS / "_report_review/remote/traces_multi.json").read_text(encoding="utf-8"))

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#16233a"
GRID = "#e3e9f2"
REF = "#7a8798"
TEACH = "#2f6fd0"
STUD = "#e0761f"

PICKS = [(0, "+10.5"), (1, "+20.9"), (2, "+31.4"),
         (3, "−10.5"), (4, "−20.9"), (5, "−31.4"),
         (24, "双脉冲"), (27, "多正弦")]

fig, axes = plt.subplots(2, 4, figsize=(9.2, 4.0))
ratio = []
for ax, (idx, label) in zip(axes.ravel(), PICKS):
    c = D["commands"][str(idx)]
    t = np.array(c["t"]); ref = np.array(c["ref"])
    tr = np.array(c["teacher"]); st = np.array(c["student"])
    m = t >= 2.0
    e_t = float(np.sqrt(np.mean((ref[m] - tr[m]) ** 2)))
    e_s = float(np.sqrt(np.mean((ref[m] - st[m]) ** 2)))
    ratio.append(e_s / max(e_t, 1e-9))

    ax.plot(t, ref, "--", color=REF, lw=0.9, zorder=4)
    ax.plot(t, tr, "-", color=TEACH, lw=1.3, zorder=5)
    ax.plot(t, st, "-", color=STUD, lw=1.3, zorder=6)
    ax.set_title(label, fontsize=9.0, color=INK, fontweight="bold", pad=3)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#a9b6c9")
    ax.tick_params(labelsize=7.0, colors=INK, length=2, pad=1)
    ax.set_xlim(0, 8)
    box = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85)
    ax.text(0.04, 0.90, f"教师 {e_t:.2f}", transform=ax.transAxes, fontsize=7.2,
            color=TEACH, fontweight="bold", va="top", bbox=box, zorder=7)
    ax.text(0.04, 0.76, f"学生 {e_s:.2f} ({e_s / max(e_t, 1e-9):.1f}×)", transform=ax.transAxes,
            fontsize=7.2, color=STUD, fontweight="bold", va="top", bbox=box, zorder=7)

for ax in axes[1]:
    ax.set_xlabel("时间 (s)", fontsize=7.6, color=INK, labelpad=1)
for ax in axes[:, 0]:
    ax.set_ylabel("p (deg/s)", fontsize=7.6, color=INK, labelpad=1)

handles = [plt.Line2D([], [], ls="--", color=REF, lw=1.0, label="参考"),
           plt.Line2D([], [], color=TEACH, lw=1.6, label="教师"),
           plt.Line2D([], [], color=STUD, lw=1.6, label="学生")]
fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=8.4,
           bbox_to_anchor=(0.5, 0.995), handlelength=1.6, columnspacing=1.6)
fig.text(0.5, 0.012, "对象 train_core-0421（2 级品质）；RMSE 为 2 s 后的跟踪误差，全库 44 架中位比 1.28×",
         ha="center", fontsize=7.4, color=INK)
fig.tight_layout(rect=(0, 0.03, 1, 0.93), pad=0.5, h_pad=0.9, w_pad=0.5)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"teacher_student_compare.{ext}", facecolor="white",
                dpi=200 if ext == "png" else None)
print("median ratio", round(float(np.median(ratio)), 2))
print("wrote teacher_student_compare.svg")
