"""Student response versus command amplitude, in-distribution vs OOD.

    python dhc6_blog_assets/make_amplitude_figure.py

Reads _report_review/remote/student_big_cmd.json (800 plants x 14 commands).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

WS = Path(__file__).resolve().parents[1]
D = json.loads((WS / "_report_review/remote/student_big_cmd.json").read_text(encoding="utf-8"))
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
C_ID = "#2f6fd0"
C_OOD = "#d8382f"

AMPS = [1, 2, 5, 10, 21, 31]
agg = {split: {"rmse": defaultdict(list), "pass": defaultdict(list)}
       for split in ("id", "ood")}
for d in D:
    for c in d["per_command"]:
        if not c["cmd"].startswith("step_"):
            continue
        amp = int(abs(c["amplitude"]))
        ok = c["rmse"] <= 1.5 and c["peak"] <= 5.0 and c["tv"] <= 12.0
        agg[d["split"]]["rmse"][amp].append(c["rmse"])
        agg[d["split"]]["pass"][amp].append(ok)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9))
for split, color, label in (("id", C_ID, "同分布（300 架）"), ("ood", C_OOD, "分布外（500 架）")):
    med = [np.median(agg[split]["rmse"][a]) for a in AMPS]
    lo = [np.percentile(agg[split]["rmse"][a], 25) for a in AMPS]
    hi = [np.percentile(agg[split]["rmse"][a], 75) for a in AMPS]
    ax1.plot(AMPS, med, marker="o", ms=4, lw=1.6, color=color, label=label)
    ax1.fill_between(AMPS, lo, hi, color=color, alpha=0.15, lw=0)
    rate = [100 * np.mean(agg[split]["pass"][a]) for a in AMPS]
    ax2.plot(AMPS, rate, marker="o", ms=4, lw=1.6, color=color, label=label)

ax1.axhline(1.5, color="#d8382f", lw=0.9, ls="--", alpha=0.7)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xticks(AMPS)
ax1.set_xticklabels([f"±{a}" for a in AMPS])
ax1.set_xlabel("指令幅值（deg/s）", fontsize=8.0, color=MUTED)
ax1.set_ylabel("跟踪 RMSE（deg/s）", fontsize=8.0, color=MUTED)
ax1.set_title("(a) 中位误差与四分位带", fontsize=8.6, color=INK, fontweight="bold",
              loc="left", pad=4)
ax1.legend(fontsize=6.8, frameon=False, loc="upper left")

ax2.set_xscale("log")
ax2.set_xticks(AMPS)
ax2.set_xticklabels([f"±{a}" for a in AMPS])
ax2.set_ylim(0, 105)
ax2.set_xlabel("指令幅值（deg/s）", fontsize=8.0, color=MUTED)
ax2.set_ylabel("三门槛通过率（%）", fontsize=8.0, color=MUTED)
ax2.set_title("(b) 通过率", fontsize=8.6, color=INK, fontweight="bold", loc="left", pad=4)
ax2.legend(fontsize=6.8, frameon=False, loc="lower left")

for ax in (ax1, ax2):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#5b6c85")
    ax.tick_params(labelsize=7.0, colors=MUTED)

fig.tight_layout(pad=0.4, w_pad=1.4)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"student_amplitude.{ext}", facecolor="white",
                dpi=150 if ext == "png" else None)
print("wrote student_amplitude.svg")
