"""Library-wide student evaluation figure (120 stratified plants).

    python dhc6_blog_assets/make_library_eval.py

(a) RMSE against roll effectiveness with the gate; (b) pass rate by quality level.
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
DATA = json.loads((WS / "_report_review/remote/student_library_eval.json").read_text(encoding="utf-8"))
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


def main() -> None:
    rmse = np.array([d["rmse"] for d in DATA])
    peak = np.array([d["peak"] for d in DATA])
    tv = np.array([d["tv"] for d in DATA])
    lv = np.array([d["level"] for d in DATA])
    lfa = np.array([d["params"]["l_fa"] for d in DATA])
    ok = (rmse <= 1.5) & (peak <= 5.0) & (tv <= 12.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 2.9),
                                   gridspec_kw={"width_ratios": [2.1, 1]})

    for level, color in COLORS.items():
        m = lv == level
        ax1.scatter(lfa[m], rmse[m], s=11, color=color, alpha=0.85, linewidths=0,
                    label=f"{level} 级")
    ax1.axhline(1.5, color="#d8382f", lw=1.0, ls="--")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("L_Fa（滚转操纵效能）", fontsize=8.0, color=MUTED)
    ax1.set_ylabel("学生 RMSE（deg/s）", fontsize=8.0, color=MUTED)
    ax1.set_title(f"(a) {len(DATA)} 架随机抽样：通过 {ok.sum()}/{len(DATA)}", fontsize=8.6,
                  color=INK, fontweight="bold", loc="left", pad=4)
    ax1.legend(fontsize=6.6, frameon=False, loc="upper right")
    ax1.grid(True, color=GRID, lw=0.6)
    ax1.set_axisbelow(True)

    rates = []
    for level in (1, 2, 3):
        m = lv == level
        rates.append(ok[m].mean() * 100)
    ax2.bar(["1 级", "2 级", "3 级"], rates, color=[COLORS[1], COLORS[2], COLORS[3]],
            width=0.62)
    for i, r in enumerate(rates):
        ax2.text(i, r + 2, f"{r:.0f}%", ha="center", fontsize=7.4, color=INK)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("通过率（%）", fontsize=8.0, color=MUTED)
    ax2.set_title("(b) 按品质等级", fontsize=8.6, color=INK, fontweight="bold",
                  loc="left", pad=4)
    ax2.grid(True, axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#5b6c85")
        ax.tick_params(labelsize=7.0, colors=MUTED)

    fig.tight_layout(pad=0.4, w_pad=1.4)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"student_library_eval.{ext}", facecolor="white",
                    dpi=150 if ext == "png" else None)
    print(f"wrote student_library_eval.svg  pass {ok.sum()}/{len(DATA)}")


if __name__ == "__main__":
    main()
