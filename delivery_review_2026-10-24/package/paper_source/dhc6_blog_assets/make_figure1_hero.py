"""Figure 1: graphical-abstract grammar — context -> method -> outcome.

    python dhc6_blog_assets/make_figure1_hero.py

Left  : the 5,000-plant quality-graded family (real data, real axes)
Middle: the controller, drawn as data shapes (history window -> net -> force rate)
Right : one real result — pass rate vs step amplitude, student (ID/OOD) vs PID
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
GRID = "#e7ecf4"
GRADE = {1: "#9dc0e6", 2: "#3f6fa8", 3: "#1b3a5c"}
ACC = "#e0761f"
GREY = "#5b6c85"
RED = "#c9453b"

rows = [json.loads(l) for l in LIB.read_text(encoding="utf-8").splitlines() if l.strip()]
lfa = np.array([r["parameters"]["l_fa"] for r in rows])
om = np.array([r["parameters"]["omega_d"] for r in rows])
lvl = np.array([r["gjb_level"] for r in rows])

summary = json.loads((WS / "_report_review/results_summary.json").read_text(encoding="utf-8"))
teach = [p["plant_id"] for p in summary["teacher"]["plants"]]
tpos = {r["plant_id"]: (r["parameters"]["l_fa"], r["parameters"]["omega_d"]) for r in rows
        if r["plant_id"] in set(teach)}

cmp_rows = json.loads((WS / "_report_review/pid_vs_student.json").read_text(encoding="utf-8"))
amps = [1, 2, 5, 10, 21, 31]
def series(split, key):
    out = []
    for a in amps:
        v = next(r for r in cmp_rows if r["split"] == split and r["cmd"] == str(a))
        out.append(v[key])
    return out

fig = plt.figure(figsize=(10.4, 4.1))
gs = GridSpec(1, 3, width_ratios=[1.0, 1.05, 1.35], wspace=0.30,
              left=0.045, right=0.985, top=0.86, bottom=0.16)

# ---------------- context
ax1 = fig.add_subplot(gs[0, 0])
for lv in (1, 2, 3):
    m = lvl == lv
    ax1.scatter(lfa[m], om[m], s=4.5, c=GRADE[lv], alpha=0.55, linewidths=0,
                label=f"{lv} 级")
tx = [tpos[k][0] for k in teach]
ty = [tpos[k][1] for k in teach]
ax1.scatter(tx, ty, s=26, facecolors="none", edgecolors=INK, linewidths=0.9, zorder=5)
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_xlabel("滚转操纵效能 $L_{Fa}$", fontsize=8.4, color=MUTED, labelpad=2)
ax1.set_ylabel("滚转模态频率 $\\omega_d$ (rad/s)", fontsize=8.4, color=MUTED, labelpad=2)
ax1.set_title("① 5,000 个品质分级对象", fontsize=9.4, color=INK, fontweight="bold",
              loc="left", pad=6)
leg = ax1.legend(fontsize=6.8, frameon=False, loc="lower right", handletextpad=0.4,
                 borderaxespad=0.2, labelspacing=0.3, markerscale=2.4)
ax1.tick_params(labelsize=6.8, colors=MUTED, length=2)
ax1.grid(True, color=GRID, lw=0.5)
ax1.set_axisbelow(True)
for sp in ("top", "right"):
    ax1.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax1.spines[sp].set_color("#5b6c85")
ax1.text(0.02, 0.02, "○ 49 个教师对象", transform=ax1.transAxes, fontsize=6.6,
         color=INK, va="bottom")

# ---------------- method
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.axis("off")
ax2.set_title("② 一个策略，跨对象复用", fontsize=9.4, color=INK, fontweight="bold",
              loc="left", pad=6)

# history window
for i in range(5, -1, -1):
    ax2.add_patch(Rectangle((0.5 + 0.16 * i, 5.4 + 0.16 * i), 1.5, 2.2,
                            facecolor="#f3d9a4", edgecolor="#c9a45f", lw=0.6, zorder=2 + i))
ax2.text(1.62, 4.55, "32 步历史", ha="center", fontsize=7.0, color=MUTED)

# net glyph
for i, (w, c) in enumerate([(2.0, "#b9d2ee"), (2.0, "#8fb3d9"), (2.0, "#4f739c")]):
    ax2.add_patch(FancyBboxPatch((3.4 + 0.12 * i, 4.6 + 0.12 * i), w, 3.6,
                                 boxstyle="round,pad=0,rounding_size=0.25",
                                 facecolor=c, edgecolor="none", zorder=2 + i))
ax2.text(4.5, 8.78, "卷积 + 主干", ha="center", fontsize=7.0, color=MUTED)
ax2.text(4.5, 3.55, "3,856 参数", ha="center", fontsize=8.8, color=INK, fontweight="bold")

# output + plant
ax2.add_patch(FancyBboxPatch((6.9, 5.25), 1.15, 1.5,
                             boxstyle="round,pad=0,rounding_size=0.2",
                             facecolor="white", edgecolor=INK, lw=1.0, zorder=4))
ax2.text(7.48, 6.0, "ΔF", ha="center", va="center", fontsize=8.0, color=INK,
         fontweight="bold", zorder=5)
ax2.add_patch(FancyBboxPatch((6.2, 1.1), 2.9, 1.35,
                             boxstyle="round,pad=0,rounding_size=0.25",
                             facecolor="#eaf5ee", edgecolor="#7fae92", lw=0.9, zorder=3))
ax2.text(7.65, 1.78, "品质对象 $G_\\theta(s)$", ha="center", va="center", fontsize=7.2,
         color=INK, zorder=4)

for p, q in [((2.3, 6.5), (3.3, 6.5)), ((5.6, 6.5), (6.8, 6.5))]:
    ax2.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                  color=INK, shrinkA=0, shrinkB=0))
ax2.add_patch(FancyArrowPatch((7.48, 5.15), (7.48, 2.55), arrowstyle="-|>",
                              mutation_scale=10, lw=1.2, color=INK,
                              connectionstyle="arc3,rad=0.0"))
ax2.add_patch(FancyArrowPatch((6.1, 1.78), (2.6, 1.78), arrowstyle="-|>",
                              mutation_scale=10, lw=1.2, color=INK))
ax2.add_patch(FancyArrowPatch((2.6, 1.78), (0.56, 5.85), arrowstyle="-|>",
                              mutation_scale=10, lw=1.2, color=INK,
                              connectionstyle="arc3,rad=-0.28"))
ax2.text(4.35, 1.32, "滚转率反馈", ha="center", fontsize=6.8, color=MUTED)

# ---------------- outcome
ax3 = fig.add_subplot(gs[0, 2])
xs = np.arange(len(amps))
ax3.plot(xs, series("id", "stu_pass"), "-o", ms=4.5, lw=2.0, color="#2f6fd0",
         label="学生 · 同分布", zorder=5)
ax3.plot(xs, series("ood", "stu_pass"), "-o", ms=4.5, lw=2.0, color=RED,
         label="学生 · 分布外", zorder=5)
ax3.plot(xs, series("id", "pid_pass"), "--s", ms=3.6, lw=1.2, color=GREY,
         label="PID 基线 · 同分布", zorder=4)
ax3.set_xticks(xs)
ax3.set_xticklabels([f"±{a}" for a in amps])
ax3.set_ylim(0, 105)
ax3.set_xlabel("阶跃指令幅值 (deg/s)", fontsize=8.4, color=MUTED, labelpad=2)
ax3.set_ylabel("门限通过率 (%)", fontsize=8.4, color=MUTED, labelpad=2)
ax3.set_title("③ 同一权重，800 架对象 × 14 条指令", fontsize=9.4, color=INK,
              fontweight="bold", loc="left", pad=6)
ax3.legend(fontsize=6.8, frameon=False, loc="lower left", handletextpad=0.5,
           borderaxespad=0.2, labelspacing=0.3)
ax3.grid(True, color=GRID, lw=0.5)
ax3.set_axisbelow(True)
for sp in ("top", "right"):
    ax3.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax3.spines[sp].set_color("#5b6c85")
ax3.tick_params(labelsize=6.8, colors=MUTED, length=2)
ax3.annotate("99.7%", (0, series("id", "stu_pass")[0]), textcoords="offset points",
             xytext=(4, 5), fontsize=7.2, color="#2f6fd0", fontweight="bold")
ax3.annotate("60.3%", (5, series("id", "stu_pass")[5]), textcoords="offset points",
             xytext=(-2, 7), fontsize=7.2, color="#2f6fd0", fontweight="bold")

fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT.with_suffix(".png"), dpi=200, facecolor="white")
print("wrote", OUT.with_suffix(".svg"))
