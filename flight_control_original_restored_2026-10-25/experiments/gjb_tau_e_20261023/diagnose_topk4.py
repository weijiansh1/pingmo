"""Time-domain diagnostic for the Top-k SAE closed loop."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tau_e  # noqa: E402

WS = HERE.parents[2]
OUT = WS / "dhc6_blog_assets"
EVID = WS / "_report_site/evidence/gjb_tau_e"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#1b2a41"
MUTED = "#5a6a80"
BLUE = "#4f7fb5"
RED = "#b5473c"
GRID = "#dfe6f0"


def main() -> None:
    rows = tau_e.common.load_library()
    row = [r for r in rows if r["plant_id"] == "id_test-3458"][0]
    signals = {}
    for policy in ("raw_ft", "topk4"):
        result = tau_e.run_case(policy, row, return_signals=True)
        signals[policy] = (np.asarray(result["pc"]), np.asarray(result["p"]))
        print(policy, "p_rms", result["p_rms"])
    EVID.mkdir(parents=True, exist_ok=True)
    np.savez(EVID / "topk4_timeseries.npz",
             **{f"{k}_{i}": v for k, pair in signals.items() for i, v in enumerate(pair)})

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.8))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.17, wspace=0.28)
    fig.suptitle("Top-k SAE 与原始历史的 180 s 闭环响应（id_test-3458）", fontsize=10.8,
                 color=INK, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    t = np.arange(len(signals["raw_ft"][0])) / tau_e.FS
    ax.plot(t, signals["raw_ft"][0], color="#9aa8bd", lw=0.8, label="指令 p_c")
    ax.plot(t, signals["raw_ft"][1], color=BLUE, lw=0.8, label="raw p")
    ax.plot(t, signals["topk4"][1], color=RED, lw=0.8, alpha=0.8, label="Top-k SAE p")
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("滚转角速度 (rad/s)", fontsize=8)
    ax.set_xlim(0, 180)
    ax.set_title("(a) 时域响应", fontsize=9.0, loc="left", color=INK)
    ax.legend(fontsize=6.4, frameon=False, ncol=3, loc="upper right")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    for policy, color, label in (("raw_ft", BLUE, "raw"), ("topk4", RED, "Top-k SAE")):
        p = signals[policy][1]
        f, psd = __import__("scipy.signal", fromlist=["welch"]).welch(
            p, fs=tau_e.FS, nperseg=2048, noverlap=1024)
        ax.semilogy(f, psd + 1e-12, color=color, lw=1.2, label=label)
    ax.set_xlim(0, 5)
    ax.set_xlabel("频率 (Hz)", fontsize=8)
    ax.set_ylabel("PSD (rad²/s²/Hz)", fontsize=8)
    ax.set_title("(b) 功率谱", fontsize=9.0, loc="left", color=INK)
    ax.legend(fontsize=6.8, frameon=False)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    for path in (OUT / "topk4_diagnostic.svg", EVID / "topk4_diagnostic.png"):
        fig.savefig(path, facecolor="white", bbox_inches="tight")
    fig.savefig(OUT / "topk4_diagnostic.png", dpi=170, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "topk4_diagnostic.svg")


if __name__ == "__main__":
    main()
