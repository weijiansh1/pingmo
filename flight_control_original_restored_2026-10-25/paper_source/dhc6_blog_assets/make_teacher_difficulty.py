"""Teacher difficulty vs plant parameters: RMSE against roll effectiveness.

    python dhc6_blog_assets/make_teacher_difficulty.py

Reads the teacher metrics from results_summary.json and the plant parameters from
the library, plots RMSE against L_Fa (log-log) and marks the three failures.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

WS = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((WS / "_report_review/results_summary.json").read_text(encoding="utf-8"))
LIB = WS / "pingmo/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"
rows = [json.loads(l) for l in LIB.read_text(encoding="utf-8").splitlines() if l.strip()]
by_id = {r["plant_id"]: r for r in rows}
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
COLORS = {1: "#2f6fd0", 2: "#e08a1e", 3: "#2f8f5b"}
BAD = "#d8382f"

pts = []
for t in SUMMARY["teacher"]["plants"]:
    p = by_id[t["plant_id"]]
    pts.append({"id": t["plant_id"], "rmse": t["rmse"], "peak": t["peak"],
                "level": p["gjb_level"], "l_fa": p["parameters"]["l_fa"],
                "pass": t["pass"]})

x = np.array([p["l_fa"] for p in pts])
y = np.array([p["rmse"] for p in pts])
ok = np.array([p["pass"] for p in pts])

fig, ax = plt.subplots(figsize=(6.4, 3.5))
for lv, color in COLORS.items():
    m = np.array([p["level"] == lv and p["pass"] for p in pts])
    ax.scatter(x[m], y[m], s=26, color=color, alpha=0.9, linewidths=0, label=f"{lv} 级")
bad = ~ok
ax.scatter(x[bad], y[bad], s=48, color=BAD, marker="X", linewidths=0, label="未通过")

# trend on the passing subset (log-log)
lx, ly = np.log10(x[ok]), np.log10(np.maximum(y[ok], 1e-2))
k, b = np.polyfit(lx, ly, 1)
grid = np.linspace(lx.min(), lx.max(), 50)
ax.plot(10 ** grid, 10 ** (k * grid + b), color=MUTED, lw=1.1, ls="--", alpha=0.9)
r = np.corrcoef(lx, ly)[0, 1]

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("L_Fa（滚转操纵效能，1/s²）", fontsize=8.6, color=MUTED)
ax.set_ylabel("教师跟踪 RMSE（deg/s）", fontsize=8.6, color=MUTED)
ax.grid(True, color=GRID, lw=0.6)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#5b6c85")
ax.tick_params(labelsize=7.2, colors=MUTED)
ax.legend(fontsize=7.0, frameon=False, loc="upper right")
ax.text(0.03, 0.93, f"通过样本 r = {r:+.2f}（log-log）", transform=ax.transAxes,
        fontsize=7.4, color=MUTED, va="top")
for p in pts:
    if not p["pass"]:
        ax.annotate(p["id"].split("-")[-1], (p["l_fa"], p["rmse"]),
                    textcoords="offset points", xytext=(-22, 6), fontsize=6.6, color=BAD)
fig.tight_layout(pad=0.4)
for ext in ("svg", "png"):
    fig.savefig(OUT / f"teacher_difficulty.{ext}", facecolor="white",
                dpi=150 if ext == "png" else None)
print("wrote teacher_difficulty.svg  (r=%.2f)" % r)
