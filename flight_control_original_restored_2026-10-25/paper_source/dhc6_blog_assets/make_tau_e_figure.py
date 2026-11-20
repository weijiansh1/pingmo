"""Figure for the first GJB tau_e evaluation (120 plants x 3 policies)."""
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
DATA = WS / "pingmo/experiments/gjb_tau_e_20261023/results/batch"

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
GRAY = "#9aa8bd"
LEVEL_COLORS = ["#9fc3bf", "#f0c987", "#e08a5b", "#b5473c"]


def loes_H(p: dict, f: np.ndarray) -> np.ndarray:
    s = 1j * 2 * np.pi * f
    k = p["K"]
    zphi, wphi = p["zeta_phi"], p["omega_phi"]
    lam, tr = p["lambda_s"], p["T_R"]
    zd, wd, tau = p["zeta_d"], p["omega_d"], p["tau_e"]
    num = k * s * (s ** 2 + 2 * zphi * wphi * s + wphi ** 2)
    den = (s - lam) * (s + 1.0 / tr) * (s ** 2 + 2 * zd * wd * s + wd ** 2)
    return (num / den) * np.exp(-tau * s)


def main() -> None:
    fits = json.loads((DATA / "fits_gate10.json").read_text(encoding="utf-8"))
    summary = json.loads((DATA / "summary_gate10.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    fig.subplots_adjust(left=0.05, right=0.89, top=0.86, bottom=0.16, wspace=0.50)
    fig.suptitle("GJB τe 首版评价：120 架 × 3 策略", fontsize=12.2, color=INK,
                 fontweight="bold", x=0.02, ha="left")

    # (a) tau_e vs tau_p
    ax = axes[0]
    for policy, color, label in (("raw_ft", BLUE, "原始历史"),
                                 ("aux_topk4", ORANGE, "SAE-aux")):
        pts = [v for v in fits.values() if v["policy"] == policy and v.get("gate10")]
        ax.scatter([v["tau_p"] for v in pts], [v["tau_e"] for v in pts],
                   s=18, color=color, alpha=0.75, edgecolor="white", linewidth=0.4,
                   label=label)
    for y, text in ((0.10, "L1 上限"), (0.20, "L2"), (0.25, "L3")):
        ax.axhline(y, color=GRAY, lw=0.8, ls=(0, (4, 3)))
        ax.text(0.202, y + 0.006, text, fontsize=5.8, color=MUTED, ha="right")
    ax.set_xlim(0, 0.205)
    ax.set_ylim(0, 0.48)
    ax.set_xlabel("对象纯延迟 τp (s)", fontsize=8)
    ax.set_ylabel("拟合等效延迟 τe (s)", fontsize=8)
    ax.set_title("(a) τe 随 τp 的变化（仅过拟合门）", fontsize=9.0, loc="left", color=INK)
    ax.legend(fontsize=6.8, frameon=False, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # (b) A21 distribution
    ax = axes[1]
    policies = [("raw_ft", "原始历史"), ("aux_topk4", "SAE-aux"), ("topk4", "Top-k SAE")]
    x = np.arange(len(policies))
    bottoms = np.zeros(len(policies))
    level_keys = ["level1", "level2", "level3", "exceeds_level3"]
    level_labels = ["1 级", "2 级", "3 级", "超过 3 级"]
    for key, label, color in zip(level_keys, level_labels, LEVEL_COLORS):
        vals = np.array([summary[p]["levels"][key] for p, _ in policies])
        ax.bar(x, vals, 0.55, bottom=bottoms, color=color, edgecolor="white",
               linewidth=0.6, label=label)
        bottoms += vals
    na_vals = np.array([summary[p]["not_assessable"] for p, _ in policies])
    ax.bar(x, na_vals, 0.55, bottom=bottoms, color="#e8ecf2", edgecolor="white",
           linewidth=0.6, hatch="//", label="未过门")
    for i, total in enumerate(bottoms + na_vals):
        ax.text(i, total + 2, f"{int(total)}", ha="center", va="bottom", fontsize=6.6,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in policies], fontsize=7.4)
    ax.set_ylabel("对象数", fontsize=8)
    ax.set_ylim(0, 135)
    ax.set_title("(b) A21 等级分布（未过门单列）", fontsize=9.0, loc="left", color=INK)
    ax.legend(fontsize=6.2, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # (c) Bode example: raw case closest to median tau_e
    ax = axes[2]
    raw = [v for v in fits.values() if v["policy"] == "raw_ft" and v.get("gate10")]
    med = summary["raw_ft"]["tau_e_median"]
    example = min(raw, key=lambda v: abs(v["tau_e"] - med)
                  + 0.002 * v["fit"]["phase_rmse_deg"])
    data = json.loads((DATA / f"raw_ft__{example['plant_id']}.json").read_text(encoding="utf-8"))
    f = np.asarray(data["freqs_hz"])
    H = np.asarray(data["H_real"]) + 1j * np.asarray(data["H_imag"])
    fd = np.geomspace(0.015, 1.7, 300)
    Hd = loes_H(example["fit"]["params"], fd)
    ax.semilogx(f, 20 * np.log10(np.abs(H)), "o", ms=3.2, color=BLUE, label="测量")
    ax.semilogx(fd, 20 * np.log10(np.abs(Hd)), "-", lw=1.1, color=BLUE, alpha=0.8)
    ax.set_xlabel("频率 (Hz)", fontsize=8)
    ax.set_ylabel("幅值 (dB)", fontsize=8, color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE, labelsize=7)
    ax.set_title("(c) 拟合示例（原始历史）", fontsize=9.0, loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.semilogx(f, np.degrees(np.unwrap(np.angle(H))), "s", ms=2.8, color=ORANGE,
                 label="测量相位")
    ax2.semilogx(fd, np.degrees(np.unwrap(np.angle(Hd))), "-", lw=1.1, color=ORANGE,
                 alpha=0.8)
    ax2.set_ylabel("相位 (deg)", fontsize=8, color=ORANGE, labelpad=2)
    ax2.tick_params(axis="y", labelcolor=ORANGE, labelsize=7)
    ax2.text(0.03, 0.05,
             f"{example['plant_id']}  τp={example['tau_p']:.3f}s  τe={example['tau_e']:.3f}s\n"
             f"VAF={example['fit']['vaf_mag']:.3f}  相位RMSE={example['fit']['phase_rmse_deg']:.1f}°",
             transform=ax2.transAxes, fontsize=5.8, color=MUTED, va="bottom")

    path = OUT / "tau_e_eval.svg"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
