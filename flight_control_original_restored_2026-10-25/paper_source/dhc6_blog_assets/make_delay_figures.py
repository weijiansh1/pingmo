"""Delay-representation figures for the paper page.

    python dhc6_blog_assets/make_delay_figures.py

Writes
  delay_probe.svg       four-panel probe summary (R2, MAE, time, sparsity)
  delay_network.svg     deployed student vs the frozen-SAE experiment branch

Numbers are read from
  pingmo/experiments/sae_delay_20261023/results_large/probe_episode.json
  pingmo/experiments/sae_delay_20261023/results_large/sae_training.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets"
DATA = WS / "pingmo/experiments/sae_delay_20261023/results_large"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#1b2a41"
MUTED = "#16233a"
RULE = "#4a5a72"
GRID = "#dfe6f0"

C_XCORR = "#2f6f6b"
C_DENSE = "#4f7fb5"
C_RAW = "#c99a3d"
C_PCA = "#8f76bd"
C_L1 = "#d9824b"
C_TOPK = "#b5473c"

# label, key, color
SERIES = [
    ("互相关", "handcrafted_xcorr", C_XCORR),
    ("稠密 AE", "dense16", C_DENSE),
    ("原始历史", "raw_history", C_RAW),
    ("PCA", "pca16", C_PCA),
    ("L1-SAE (L0≈7)", "l1_3e-03", C_L1),
    ("Top-k SAE (k=4)", "topk4", C_TOPK),
]

SHORT = {
    "handcrafted_xcorr": "互相关",
    "dense16": "稠密 AE",
    "raw_history": "原始历史",
    "pca16": "PCA",
    "l1_3e-03": "L1-SAE (L0≈7)",
    "l1_3e-02": "L1-SAE (L0≈3.5)",
    "l1_3e-04": "L1-SAE (L0≈12)",
    "topk4": "Top-k SAE (k=4)",
    "topk8": "Top-k SAE (k=8)",
}


def load() -> tuple[dict, dict]:
    probe = json.loads((DATA / "probe_episode.json").read_text(encoding="utf-8"))
    sae = json.loads((DATA / "sae_training.json").read_text(encoding="utf-8"))
    return probe, sae


def probe_figure(probe: dict, sae: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 6.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.90, bottom=0.09,
                        hspace=0.42, wspace=0.28)
    fig.suptitle("延迟表征实验：可解码性、误差与稀疏代价（留出 120 架）",
                 fontsize=12.2, color=INK, fontweight="bold", x=0.02, ha="left")

    # (a) R2 at 1.2 s and 2.0 s
    ax = axes[0, 0]
    times = ("1.2s", "2.0s")
    x = np.arange(len(SERIES))
    width = 0.36
    colors = ["#9fc3bf", C_XCORR]
    for k, t in enumerate(times):
        vals = [probe["probes"][t][key]["test_r2"] for _, key, _ in SERIES]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=t,
                      color=colors[k], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.4, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SERIES], rotation=18, ha="right", fontsize=7.6)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("测试 R²", fontsize=8)
    ax.set_title("(a) 因果历史前缀的延迟可解码性", fontsize=9.2, loc="left", color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # (b) MAE
    ax = axes[0, 1]
    for k, t in enumerate(times):
        vals = [probe["probes"][t][key]["test_mae_ms"] for _, key, _ in SERIES]
        bars = ax.bar(x + (k - 0.5) * width, vals, width, label=t,
                      color=colors[k], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in SERIES], rotation=18, ha="right", fontsize=7.6)
    ax.set_ylabel("MAE (ms)", fontsize=8)
    ax.set_title("(b) 延迟估计误差（越低越好）", fontsize=9.2, loc="left", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    # (c) R2 vs causal time
    ax = axes[1, 0]
    tkeys = [f"{t:.1f}s" for t in probe["times_s"]]
    xs = probe["times_s"]
    for label, key, color in SERIES:
        if key not in probe["probes"][tkeys[0]]:
            continue
        vals = [probe["probes"][t][key]["test_r2"] for t in tkeys]
        ax.plot(xs, vals, marker="o", ms=3.2, lw=1.4, color=color, label=label)
    ax.set_xlabel("因果前缀截止时间 (s)", fontsize=8)
    ax.set_ylabel("测试 R²", fontsize=8)
    ax.set_ylim(0, 1.04)
    ax.set_title("(c) 随观测时间的变化", fontsize=9.2, loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.3, frameon=False, ncol=2, loc="lower left")

    # (d) sparsity tradeoff
    ax = axes[1, 1]
    rows = []
    for name, stats in sae["models"].items():
        if name == "dense16" or name in ("l1_3e-03", "l1_3e-02", "l1_3e-04", "topk4", "topk8"):
            r2 = probe["probes"]["1.2s"][name]["test_r2"]
            rows.append((stats["l0"], r2, name))
    for l0, r2, name in rows:
        color = C_DENSE if name == "dense16" else (C_TOPK if "topk" in name else C_L1)
        ax.scatter([l0], [r2], s=46, color=color, edgecolor="white", linewidth=0.7, zorder=3)
        offset = {"dense16": (-8, 4), "topk8": (5, 8), "l1_3e-04": (5, -10),
                  "l1_3e-03": (-6, -12), "topk4": (5, 6), "l1_3e-02": (5, -12)}.get(name, (5, 4))
        ax.annotate(SHORT.get(name, name), (l0, r2), textcoords="offset points",
                    xytext=offset, fontsize=6.4, color=MUTED,
                    ha="right" if name in ("dense16", "l1_3e-03") else "left")
    ax.axhline(probe["probes"]["1.2s"]["raw_history"]["test_r2"], color=C_RAW, lw=1.0,
               ls=(0, (4, 3)), label="原始历史 (160 维)")
    ax.axhline(probe["probes"]["1.2s"]["pca16"]["test_r2"], color=C_PCA, lw=1.0,
               ls=(0, (1, 2)), label="PCA (16 维)")
    ax.set_xlabel("平均激活单元数 L0", fontsize=8)
    ax.set_ylabel("1.2 s 测试 R²", fontsize=8)
    ax.set_ylim(0, 0.72)
    ax.set_xlim(0, 17.5)
    ax.set_title("(d) 稀疏度—延迟信息权衡", fontsize=9.2, loc="left", color=INK)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.6, frameon=False, loc="lower right")

    path = OUT / "delay_probe.svg"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", path)


# ------------------------------------------------------------------ network
def arrow(ax, p, q, color=RULE, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=color, ls=ls, zorder=7))


def box(ax, x, y, w, h, title, sub="", face="#eef3fa", edge="#7796bd",
        title_size=7.4, sub_size=6.2, text_color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=face, edgecolor=edge, lw=0.8, zorder=3))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), title, ha="center", va="center",
            fontsize=title_size, color=text_color, fontweight="bold", zorder=4)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=sub_size, color=MUTED, zorder=4)


def grid_block(ax, x, y, w, h, rows, cols, face="#f3d9a4", edge="#c9a45f"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=0.8, zorder=3))
    for r in range(1, rows):
        ax.plot([x, x + w], [y + h * r / rows] * 2, color="white", lw=0.3,
                alpha=0.75, zorder=4)
    for c in range(1, cols):
        ax.plot([x + w * c / cols] * 2, [y, y + h], color="white", lw=0.3,
                alpha=0.75, zorder=4)


def z_tensor(ax, x, y, w=0.55, h=0.34):
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#e3daf6", edgecolor="#8f76bd",
                           lw=0.8, zorder=3))
    # 16 cells, only four filled: a visual stand-in for a k=4 sparse code.
    for i in range(16):
        cx = x + (i % 8) * w / 8
        cy = y + (i // 8) * h / 2
        if i in (2, 6, 9, 13):
            ax.add_patch(Rectangle((cx + 0.012, cy + 0.015), w / 8 - 0.024, h / 2 - 0.03,
                                   facecolor="#8f76bd", edgecolor="none", zorder=4))
    return x + w


def draw_network() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 5.0)

    ax.text(0.05, 4.86, "延迟表征实验：原始历史与冻结 SAE 两条前端的对照",
            fontsize=11.6, color=INK, fontweight="bold", ha="left", va="top")
    ax.text(0.05, 4.56, "SAE 只做表征；策略头/探针在冻结潜码上训练。对象按 τp 分层，训练与测试不重叠。",
            fontsize=7.2, color=MUTED, ha="left", va="top")

    # shared input
    grid_block(ax, 0.28, 1.60, 0.62, 1.55, rows=16, cols=5)
    ax.text(0.59, 1.42, "32×5 历史", ha="center", va="top", fontsize=7.0, color=INK)
    ax.text(0.59, 1.24, "cmd p_ref p e F", ha="center", va="top", fontsize=5.8, color=MUTED)

    # top path: deployed student
    ty = 3.30
    ax.text(1.12, 3.98, "部署学生（图 3，原始历史）", fontsize=7.6, color=MUTED,
            fontweight="bold")
    arrow(ax, (0.94, 2.85), (1.14, ty + 0.30), color="#5b6c85")
    box(ax, 1.16, ty, 1.38, 0.58, "Conv1D ×3", "5→8→8→8", face="#eef3fa", edge="#7796bd")
    arrow(ax, (2.56, ty + 0.29), (2.78, ty + 0.29), color="#5b6c85")
    box(ax, 2.80, ty, 0.80, 0.58, "c (16)", "时序特征", face="#dce8f7", edge="#7796bd")
    arrow(ax, (3.62, ty + 0.29), (3.84, ty + 0.29), color="#5b6c85")
    box(ax, 3.86, ty, 1.56, 0.58, "拼接 + 1 残差块", "16+9+7 → 32", face="#eef3fa",
        edge="#7796bd")
    arrow(ax, (5.44, ty + 0.29), (5.66, ty + 0.29), color="#5b6c85")
    box(ax, 5.68, ty, 1.26, 0.58, "均值头", "32 → 1", face="#fbe7d6", edge="#c07f4d")
    arrow(ax, (6.96, ty + 0.29), (7.20, ty + 0.29), color="#5b6c85")
    ax.text(7.28, ty + 0.29, "ΔF", fontsize=8.2, color=INK, va="center")

    # bottom path: SAE
    by = 1.45
    ax.text(1.12, 2.20, "SAE 延迟实验（冻结编码器）", fontsize=7.6, color=C_TOPK,
            fontweight="bold")
    arrow(ax, (0.94, 2.45), (1.14, by + 0.34), color=C_TOPK)
    box(ax, 1.16, by, 1.62, 0.66, "SAE 编码器", "160→64→16  ReLU", face="#f6e3d5",
        edge=C_L1, title_size=7.4)
    ax.text(1.97, by - 0.10, "L1 或 top-k；冻结", ha="center", va="top", fontsize=5.8,
            color=MUTED)
    arrow(ax, (2.80, by + 0.33), (3.02, by + 0.33), color=C_L1)
    zx = z_tensor(ax, 3.04, by + 0.14)
    ax.text(3.32, by - 0.10, "z (16)，平均 L0≈4–7", ha="center", va="top", fontsize=5.8,
            color=MUTED)
    arrow(ax, (zx + 0.04, by + 0.33), (zx + 0.30, by + 0.33), color=C_L1)

    # policy branch from z
    px = zx + 0.34
    ax.text(px + 0.02, by + 0.78, "策略分支", fontsize=6.6, color=C_L1, fontweight="bold")
    box(ax, px, by + 0.52, 1.42, 0.48, "拼接 + 1 残差块", "16+9+7 → 32", face="#f6e3d5",
        edge=C_L1, title_size=6.6, sub_size=5.8)
    arrow(ax, (px + 1.44, by + 0.76), (px + 1.66, by + 0.76), color=C_L1)
    box(ax, px + 1.68, by + 0.52, 1.12, 0.48, "均值头", "32 → 1", face="#fbe7d6",
        edge="#c07f4d", title_size=6.6, sub_size=5.8)
    arrow(ax, (px + 2.82, by + 0.76), (px + 3.04, by + 0.76), color=C_L1)
    ax.text(px + 3.12, by + 0.76, "ΔF", fontsize=7.6, color=INK, va="center")

    # probe branch from z
    qx = zx + 0.34
    ax.text(qx + 0.02, by - 0.02, "延迟读出", fontsize=6.6, color=C_XCORR,
            fontweight="bold")
    box(ax, qx, by - 0.34, 1.20, 0.42, "线性探针", "16 → 1", face="#dff0ed",
        edge=C_XCORR, title_size=6.6, sub_size=5.8)
    arrow(ax, (qx + 0.20, by + 0.14), (qx + 0.20, by - 0.00), color=C_XCORR)
    arrow(ax, (qx + 1.22, by - 0.13), (qx + 1.52, by - 0.13), color=C_XCORR)
    ax.text(qx + 1.60, by - 0.13, "估计 τp (ms)", fontsize=7.4, color=INK, va="center")

    # baseline note
    nx, ny = 7.55, 1.40
    ax.add_patch(Rectangle((nx, ny), 2.55, 1.55, facecolor="#fbfcfe",
                           edgecolor="#cfd8e6", lw=0.7, zorder=2))
    ax.text(nx + 0.12, ny + 1.36, "对照表征", fontsize=6.8, color=MUTED, fontweight="bold")
    lines = ["原始历史 160", "PCA-16", "稠密 AE-16", "互相关-36"]
    for i, line in enumerate(lines):
        ax.text(nx + 0.12, ny + 1.10 - i * 0.24, "· " + line, fontsize=6.4, color=INK)

    ax.text(0.05, 0.18,
            "结论：延迟信息主要存在于“请求力—响应”的时移；16 维稀疏潜码明显弱于互相关基线。",
            fontsize=7.0, color=MUTED, ha="left", va="bottom")

    path = OUT / "delay_network.svg"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def closed_loop_figure(policy: dict, cross: dict | None) -> None:
    variants = [("raw_ft", "原始历史"), ("pca16", "PCA-16"),
                ("dense16", "稠密 AE-16"), ("topk4", "Top-k SAE-16"),
                ("aux_topk4", "SAE-aux")]
    variants = [(k, label) for k, label in variants if k in policy]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.9))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.86, bottom=0.14, wspace=0.24)
    fig.suptitle("闭环对照：延迟分层与跨时延迁移（120 架留出）", fontsize=11.6,
                 color=INK, fontweight="bold", x=0.02, ha="left")

    ax = axes[0]
    bins = [("short", "短延迟", "#9fc3bf"), ("medium", "中延迟", "#4f7fb5"),
            ("long", "长延迟", "#b5473c")]
    x = np.arange(len(variants))
    width = 0.26
    for k, (key, label, color) in enumerate(bins):
        vals = [policy[v]["eval"][key]["pass_rate"] * 100 for v, _ in variants]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=6.2, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in variants], rotation=15, ha="right",
                       fontsize=7.4)
    ax.set_ylim(0, 108)
    ax.set_ylabel("通过率 (%)", fontsize=8)
    ax.set_title("(a) 按对象时延分层", fontsize=9.2, loc="left", color=INK)
    ax.legend(fontsize=6.8, frameon=False, ncol=3, loc="lower left")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    ax = axes[1]
    if cross:
        cvar = [(k, label) for k, label in variants if k in cross]
        x = np.arange(len(cvar))
        all_vals = [policy[k]["eval"]["long"]["pass_rate"] * 100 for k, _ in cvar]
        cross_vals = [cross[k]["eval"]["long"]["pass_rate"] * 100 for k, _ in cvar]
        width = 0.34
        b1 = ax.bar(x - width / 2, all_vals, width, label="全延迟训练",
                    color="#8fb3d9", edgecolor="white", linewidth=0.6)
        b2 = ax.bar(x + width / 2, cross_vals, width, label="仅短/中延迟训练",
                    color="#d9824b", edgecolor="white", linewidth=0.6)
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.2,
                        f"{b.get_height():.0f}", ha="center", va="bottom",
                        fontsize=6.2, color=MUTED)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in cvar], rotation=15, ha="right",
                           fontsize=7.4)
        ax.set_ylim(0, 108)
        ax.set_ylabel("长延迟通过率 (%)", fontsize=8)
        ax.set_title("(b) 跨时延：训练分布对长延迟留出的影响", fontsize=9.2,
                     loc="left", color=INK)
        ax.legend(fontsize=6.8, frameon=False, loc="lower left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "跨时延结果待补", ha="center", va="center",
                fontsize=9, color=MUTED)

    path = OUT / "delay_closed_loop.svg"
    fig.savefig(path, facecolor="white")
    fig.savefig(path.with_suffix(".png"), dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    probe, sae = load()
    probe_figure(probe, sae)
    # The network diagram lives in its own module so it can reuse the exact
    # primitives of make_network_tensor.py (same palette / tensor shapes).
    import make_delay_network
    make_delay_network.draw()
    policy_path = DATA / "policies_all_dagger2/policy_results.json"
    cross_path = DATA / "policies_cross/policy_results.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        cross = (json.loads(cross_path.read_text(encoding="utf-8"))
                 if cross_path.exists() else None)
        closed_loop_figure(policy, cross)


if __name__ == "__main__":
    main()
