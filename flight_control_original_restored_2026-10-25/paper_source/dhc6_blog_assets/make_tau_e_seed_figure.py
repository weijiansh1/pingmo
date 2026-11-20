"""Seed-robustness figure for the GJB tau_e evaluation."""
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
DATA = WS / "pingmo/experiments/gjb_tau_e_20261023/results/seed_summary.json"

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
LEVEL_COLORS = ["#9fc3bf", "#f0c987", "#e08a5b", "#b5473c"]


def load_fits(path: Path, gate_key: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, value in data.items():
        ok = bool(value.get(gate_key, value.get("gate", False)))
        tau = value.get("tau_e")
        if tau is None and value.get("fit", {}).get("ok"):
            tau = value["fit"]["params"]["tau_e"]
        out[key] = {"policy": value["policy"], "ok": ok, "tau_e": tau}
    return out


def main() -> None:
    data = json.loads(DATA.read_text(encoding="utf-8"))
    base = DATA.parent
    seed1 = load_fits(base / "batch/fits_gate10.json", "gate10")
    seed2 = load_fits(base / "batch_seed2/fits.json", "gate")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.18, wspace=0.30)
    fig.suptitle("τe 评价的策略种子稳健性（120 架，raw / SAE-aux）", fontsize=11.6,
                 color=INK, fontweight="bold", x=0.02, ha="left")

    # (a) common-set scatter
    ax = axes[0]
    ax.plot([0, 0.45], [0, 0.45], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)))
    for policy, color, label in (("raw_ft", BLUE, "原始历史"),
                                 ("aux_topk4", ORANGE, "SAE-aux")):
        keys = [k for k in seed1 if k in seed2 and seed1[k]["policy"] == policy
                and seed1[k]["ok"] and seed2[k]["ok"]]
        xs = [seed1[k]["tau_e"] for k in keys]
        ys = [seed2[k]["tau_e"] for k in keys]
        ax.scatter(xs, ys, s=13, color=color, alpha=0.65, edgecolor="white",
                   linewidth=0.3, label=f"{label}（n={len(keys)}）", zorder=3)
        med1 = data[policy]["common_median"]["seed1"]
        med2 = data[policy]["common_median"]["seed2"]
        ax.scatter([med1], [med2], s=58, color=color, edgecolor=INK,
                   linewidth=0.6, marker="D", zorder=4)
        ax.annotate(f"({med1:.3f}, {med2:.3f})", (med1, med2),
                    textcoords="offset points", xytext=(6, -10), fontsize=6.0,
                    color=color)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.45)
    ax.set_xlabel("种子 1 的 τe (s)", fontsize=8)
    ax.set_ylabel("种子 2 的 τe (s)", fontsize=8)
    ax.set_title("(a) 共同可评集：逐对象 τe", fontsize=9.0, loc="left", color=INK)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.text(0.97, 0.05, "虚线为 y=x；菱形为中位", transform=ax.transAxes, ha="right",
            fontsize=5.8, color=MUTED)

    # (b) A21 distribution by seed
    ax = axes[1]
    groups = [("raw_ft", "原始历史"), ("aux_topk4", "SAE-aux")]
    labels = [f"{label}\n种子{s}" for _, label in groups for s in (1, 2)]
    bottoms = np.zeros(4)
    for li, (key, label) in enumerate((("level1", "1 级"), ("level2", "2 级"),
                                       ("level3", "3 级"), ("exceeds_level3", ">3 级"))):
        vals = []
        for policy, _ in groups:
            vals += [data[policy]["seed1"]["levels"][key],
                     data[policy]["seed2"]["levels"][key]]
        vals = np.asarray(vals, dtype=float)
        ax.bar(range(4), vals, 0.62, bottom=bottoms, color=LEVEL_COLORS[li],
               edgecolor="white", linewidth=0.6, label=label)
        bottoms += vals
    totals = bottoms.copy()
    for i, policy in enumerate(("raw_ft", "aux_topk4")):
        for s in (0, 1):
            ax.text(i * 2 + s, totals[i * 2 + s] + 2,
                    f"{int(totals[i * 2 + s])}", ha="center", va="bottom",
                    fontsize=6.4, color=MUTED)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylabel("可评对象数", fontsize=8)
    ax.set_ylim(0, 125)
    ax.set_title("(b) A21 等级分布", fontsize=9.0, loc="left", color=INK)
    ax.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    path = OUT / "tau_e_seed.svg"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
