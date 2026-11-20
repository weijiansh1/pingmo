"""采样图：参数边际分布、二维散点与划分配额（对象库 IV-A，5000 架）。

    python dhc6_blog_assets/make_sampling_overview.py

读 data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl，
输出 sampling_marginals.png / sampling_scatter.png / sampling_quotas.png。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

WS = Path(__file__).resolve().parents[1]
LIB = (WS / "pingmo/data/aircraft/generated/p_channel_library_iv_a_5000_v1"
       / "plants.jsonl")
OUT = WS / "dhc6_blog_assets"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(
            fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

KEYS = ["l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p"]
LABEL = {"l_fa": "L_Fa（操纵效能系数）", "lambda_s": "λ_s（1/s）", "t_r": "T_R（s）",
         "zeta_d": "ζ_d", "omega_d": "ω_d（rad/s）", "r_omega": "R_ω = ω_φ/ω_d",
         "r_zeta": "R_ζ = ζ_φ/ζ_d", "tau_p": "τ_p（s）"}
LOGX = {"l_fa", "t_r", "omega_d", "r_omega", "r_zeta"}
BINS = {"l_fa": np.geomspace(0.01, 1.7, 46),
        "lambda_s": np.linspace(-0.16, 0.30, 46),
        "t_r": np.geomspace(0.17, 16, 46),
        "zeta_d": np.linspace(0, 0.71, 46),
        "omega_d": np.geomspace(0.17, 6.1, 46),
        "r_omega": np.geomspace(0.28, 3.1, 46),
        "r_zeta": np.geomspace(0.14, 140, 46),
        "tau_p": np.linspace(0, 0.50, 46)}
COLOR = {1: "#2f6fd0", 2: "#e08a1e", 3: "#c8322b", None: "#8b8b8b"}
NAME = {1: "1 级", 2: "2 级", 3: "3 级", None: "OOD（分布外）"}
BUCKET_TO_LEVEL = {"L1": 1, "L2": 2, "L3": 3, "OOD": None}


def load():
    rows = [json.loads(line) for line in LIB.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    values = {k: np.array([r["parameters"][k] for r in rows]) for k in KEYS}
    level = np.array([BUCKET_TO_LEVEL[r["sampling_bucket"]] for r in rows], dtype=object)
    split = np.array([r["split"] for r in rows])
    return values, level, split


def hist_axis(ax, values, level, key, bins, title):
    for lv in (1, 2, 3, None):
        mask = level == lv
        if not mask.any():
            continue
        ax.hist(values[key][mask], bins=bins, color=COLOR[lv], alpha=0.95,
                label=NAME[lv], histtype="step", linewidth=1.5)
    if key in LOGX:
        ax.set_xscale("log")
    ax.set_xlabel(LABEL[key], fontsize=9.5)
    ax.set_title(title, fontsize=10.5)
    ax.tick_params(labelsize=8.5)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_ylim(bottom=0)


def vline(ax, x, style=":", label=None):
    ax.axvline(x, color="#333333", lw=0.9, ls=style, alpha=0.75)
    if label:
        ax.annotate(label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(2, -2), textcoords="offset points", fontsize=7.6,
                    color="#333333", ha="left", va="top")


def marginals() -> None:
    values, level, _ = load()
    fig, axes = plt.subplots(2, 4, figsize=(11.6, 4.6))
    flat = axes.ravel()

    hist_axis(flat[0], values, level, "l_fa", BINS["l_fa"], "滚转操纵效能（由灵敏度反标）")
    flat[0].set_ylabel("架数", fontsize=9.5)

    hist_axis(flat[1], values, level, "lambda_s", BINS["lambda_s"], "螺旋模态根")
    vline(flat[1], 0.0, "-", "0")

    hist_axis(flat[2], values, level, "t_r", BINS["t_r"], "滚转模态时间常数")
    for x, text in ((1.0, "1.0"), (1.4, "1.4"), (10.0, "10")):
        vline(flat[2], x, ":", text)

    hist_axis(flat[3], values, level, "zeta_d", BINS["zeta_d"], "荷兰滚阻尼比")
    for x, text in ((0.02, "0.02"), (0.19, "0.19")):
        vline(flat[3], x, ":", text)

    hist_axis(flat[4], values, level, "omega_d", BINS["omega_d"], "荷兰滚频率")
    flat[4].set_ylabel("架数", fontsize=9.5)
    for x, text in ((0.4, "0.4"), (1.0, "1.0")):
        vline(flat[4], x, ":", text)

    hist_axis(flat[5], values, level, "r_omega", BINS["r_omega"], "零点频率比")
    for x, text in ((0.65, "0.65"), (1.35, "1.35")):
        vline(flat[5], x, ":", text)

    hist_axis(flat[6], values, level, "r_zeta", BINS["r_zeta"], "零点阻尼比")

    hist_axis(flat[7], values, level, "tau_p", BINS["tau_p"], "纯运输延迟")
    vline(flat[7], 0.25, "--", "0.25")

    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("5000 架对象的参数边际分布（颜色为选入等级，竖线为等级边界）", fontsize=12)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    path = OUT / "sampling_marginals.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def scatter() -> None:
    values, level, _ = load()
    ink = "#16233c"
    panel_bg = "#f4f7fc"
    edge = "#b7c2d4"
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), sharey=True)

    def panel(ax, xkey, title, xlabel, letter, xlog=False):
        ax.set_facecolor(panel_bg)
        for lv in (None, 1, 2, 3):
            mask = level == lv
            if not mask.any():
                continue
            ax.scatter(values[xkey][mask], values["omega_d"][mask], s=8,
                       alpha=0.78, color=COLOR[lv], label=NAME[lv],
                       linewidths=0, rasterized=True)
        if xlog:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10.5, labelpad=6, color=ink)
        ax.set_title(title, fontsize=11.5, pad=10, color=ink, fontweight="bold")
        ax.text(-0.075, 1.045, letter, transform=ax.transAxes, fontsize=13,
                fontweight="bold", color=ink, ha="left", va="bottom")
        ax.set_axisbelow(True)
        ax.grid(True, color="white", lw=1.1)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(edge)
            ax.spines[side].set_linewidth(0.9)
        ax.tick_params(labelsize=9, colors="#3d4a63", length=3, width=0.8)

    ax = axes[0]
    panel(ax, "t_r", "滚转模态时间常数与荷兰滚频率", LABEL["t_r"], "(a)", xlog=True)
    for x in (1.0, 1.4, 10.0):
        ax.axvline(x, color="#5a6b85", lw=1.0, ls=(0, (5, 4)), alpha=0.55)
    for y in (1.0, 0.4):
        ax.axhline(y, color="#5a6b85", lw=1.0, ls=(0, (5, 4)), alpha=0.55)

    ax = axes[1]
    panel(ax, "zeta_d", "荷兰滚平面", LABEL["zeta_d"], "(b)")
    for y in (1.0, 0.4):
        ax.axhline(y, color="#5a6b85", lw=1.0, ls=(0, (5, 4)), alpha=0.55)
    ax.axvline(0.19, color="#5a6b85", lw=1.0, ls=(0, (5, 4)), alpha=0.55)
    ax.annotate("3 级零阻尼锚点", xy=(0.028, 1.35), xytext=(0.075, 2.45),
                fontsize=9, color="#6d2422", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#6d2422", lw=1.0,
                                shrinkA=2, shrinkB=3))

    axes[1].tick_params(axis="y", left=False)
    axes[0].set_ylabel(LABEL["omega_d"], fontsize=10.5, labelpad=6, color=ink)
    axes[0].set_ylim(0.1, 6.3)

    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(NAME[lv]) for lv in (1, 2, 3, None)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
               loc="lower center", ncol=4, fontsize=9.5, markerscale=2.6,
               frameon=True, facecolor="white", edgecolor="#dfe4ee",
               borderpad=0.6, handletextpad=0.5, columnspacing=2.4,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    path = OUT / "sampling_scatter.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def quotas() -> None:
    _, level, split = load()
    orders = ["train_core", "train_boundary", "validation", "id_test", "ood_test"]
    tick = {"train_core": "训练主集\ntrain_core", "train_boundary": "边界集\ntrain_boundary",
            "validation": "验证集\nvalidation", "id_test": "同分布测试\nid_test",
            "ood_test": "分布外研究集\nood_test"}
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    bottom = np.zeros(len(orders))
    for lv in (1, 2, 3, None):
        counts = np.array([np.sum((split == s) & (level == lv)) for s in orders])
        if not counts.any():
            continue
        ax.bar(orders, counts, bottom=bottom, color=COLOR[lv], label=NAME[lv])
        bottom += counts
    for index, total in enumerate(bottom):
        ax.text(index, total + 40, f"{int(total)}", ha="center", fontsize=9.5)
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([tick[s] for s in orders], fontsize=9)
    ax.set_ylabel("架数", fontsize=10)
    ax.set_title("五个划分的等级配额（合计 5000 架）", fontsize=12)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper right")
    ax.set_ylim(0, 2450)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    path = OUT / "sampling_quotas.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    marginals()
    scatter()
    quotas()




