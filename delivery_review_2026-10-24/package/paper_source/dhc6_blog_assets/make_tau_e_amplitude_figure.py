"""Amplitude-robustness figure for the GJB tau_e evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets"
DATA = WS / "pingmo/experiments/gjb_tau_e_20261023/results/amplitude_summary.json"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#1b2a41"
MUTED = "#5a6a80"
GRID = "#dfe6f0"
BLUE = "#4f7fb5"
ORANGE = "#d9824b"
RED = "#b5473c"
GRAY = "#9aa8bd"


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    amps = ["0.5", "1.0", "2.0"]
    x = np.array([0.5, 1.0, 2.0])
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.84, bottom=0.17, wspace=0.34)
    fig.suptitle("τe 评价的激励幅值稳健性（120 架 × 3 策略）", fontsize=11.8,
                 color=INK, fontweight="bold", x=0.02, ha="left")

    # (a) common-case tau_e
    ax = axes[0]
    for policy, color, label in (("raw_ft", BLUE, "原始历史"),
                                 ("aux_topk4", ORANGE, "SAE-aux")):
        vals = [data[policy]["common_tau_e_median"][a] for a in amps]
        ax.plot(x, vals, "o-", color=color, lw=1.5, ms=4.5, label=label)
        for xx, vv in zip(x, vals):
            ax.text(xx, vv + 0.004, f"{vv:.3f}", ha="center", va="bottom",
                    fontsize=6.2, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["0.5", "1.0", "2.0"])
    ax.set_xlim(0.35, 2.15)
    ax.set_xlabel("每分量幅值 (deg/s)", fontsize=8)
    ax.set_ylabel("共同可评集 τe 中位 (s)", fontsize=8)
    ax.set_ylim(0, 0.16)
    ax.set_title(f"(a) 共同可评集：raw n={data['raw_ft']['common_n']}，"
                 f"aux n={data['aux_topk4']['common_n']}", fontsize=8.6,
                 loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # (b) assessable count
    ax = axes[1]
    for policy, color, label in (("raw_ft", BLUE, "原始历史"),
                                 ("aux_topk4", ORANGE, "SAE-aux"),
                                 ("topk4", RED, "Top-k SAE")):
        vals = [data[policy]["by_amplitude"][a]["assessable"] for a in amps]
        ax.plot(x, vals, "o-", color=color, lw=1.5, ms=4.5, label=label)
        for xx, vv in zip(x, vals):
            ax.text(xx, vv + 3, str(vv), ha="center", va="bottom", fontsize=6.2,
                    color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(["0.5", "1.0", "2.0"])
    ax.set_xlim(0.35, 2.15)
    ax.set_xlabel("每分量幅值 (deg/s)", fontsize=8)
    ax.set_ylabel("可评对象数 / 120", fontsize=8)
    ax.set_ylim(0, 125)
    ax.set_title("(b) 过拟合门比例随幅值变化", fontsize=8.6, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="center left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # (c) topk4 residual vs amplitude
    ax = axes[2]
    vaf = [data["topk4"]["by_amplitude"][a]["vaf_median"] for a in amps]
    phase = [data["topk4"]["by_amplitude"][a]["phase_median"] for a in amps]
    ax.bar(x, vaf, 0.32, color=RED, alpha=0.75, edgecolor="white", linewidth=0.6,
           label="VAF 中位")
    ax.axhline(0.95, color=GRAY, lw=0.8, ls=(0, (4, 3)))
    ax.text(2.08, 0.96, "VAF 门 0.95", fontsize=5.6, color=MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["0.5", "1.0", "2.0"])
    ax.set_xlim(0.35, 2.15)
    ax.set_xlabel("每分量幅值 (deg/s)", fontsize=8)
    ax.set_ylabel("VAF 中位", fontsize=8, color=RED)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelcolor=RED, labelsize=7)
    ax.set_title("(c) Top-k SAE：三种幅值均不过门", fontsize=8.6, loc="left",
                 color=INK)
    ax2 = ax.twinx()
    ax2.plot(x, phase, "s--", color=INK, lw=1.2, ms=4, label="相位 RMSE")
    ax2.set_ylabel("相位 RMSE (deg)", fontsize=8, color=INK)
    ax2.tick_params(axis="y", labelcolor=INK, labelsize=7)
    for xx, vv in zip(x, phase):
        ax2.text(xx, vv + 1.5, f"{vv:.0f}°", ha="center", va="bottom",
                 fontsize=6.0, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    path = OUT / "tau_e_amplitude.svg"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
