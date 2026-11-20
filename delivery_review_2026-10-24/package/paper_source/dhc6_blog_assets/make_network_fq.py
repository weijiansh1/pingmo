"""Flight-quality-aware network diagram: policy row + closed-loop strip.

    python dhc6_blog_assets/make_network_fq.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

WS = Path(__file__).resolve().parents[1]
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
EDGE = "#3d4f6b"
BLUE, BLUE_E = "#dce8f7", "#8fb0d8"
GOLD, GOLD_E = "#fbeccb", "#d8b46a"
LILAC, LILAC_E = "#e3daf6", "#a48fd0"
GREEN, GREEN_E = "#dcefe2", "#7fae92"
PANEL = "#f7f9fc"


def box(ax, x, y, w, h, title, sub="", face=BLUE, edge=BLUE_E, fs=8.0, subfs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=face, edgecolor=edge, lw=0.9, zorder=3))
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold", zorder=4)
    if sub:
        for i, line in enumerate(sub.split("\n")):
            ax.text(x + w / 2, y + h * 0.30 - i * 0.20, line, ha="center", va="center",
                    fontsize=subfs, color=MUTED, zorder=4)
    return x + w


def arrow(ax, p, q, color=EDGE, lw=0.9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=color, zorder=6))


def draw(kind: str) -> None:
    teacher = kind == "teacher"
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    ax.axis("off")

    # ---------------- top row: policy
    y = 3.10
    ax.text(0.20, 4.32, "① 本体感受输入：32 步历史 × 5 通道  +  当前 9  +  旧动作 7",
            fontsize=7.4, color=MUTED)
    for i in range(5, -1, -1):
        ax.add_patch(Rectangle((0.35 + 0.085 * i, y - 0.24 + 0.085 * i), 0.44, 0.42,
                               facecolor=GOLD, edgecolor=GOLD_E, lw=0.7, zorder=2 + i))
    ax.text(0.35, y - 0.44, "h[t−31] … h[t]", fontsize=6.2, color=MUTED)
    arrow(ax, (1.10, y), (1.32, y))
    cv = box(ax, 1.38, y - 0.44, 1.40, 0.88, "Conv1D ×3", "5→24→24→24 · D 1/4/8")
    arrow(ax, (cv + 0.06, y), (cv + 0.28, y))
    cc = box(ax, cv + 0.34, y - 0.24, 0.66, 0.48, "‖", "concat", face="white", edge=EDGE,
             fs=9.5, subfs=6.0)
    arrow(ax, (cc + 0.06, y), (cc + 0.28, y))
    tr = box(ax, cc + 0.34, y - 0.44, 1.36, 0.88, "MLP trunk", "512 × 8" if teacher else "32 × 1")
    arrow(ax, (tr + 0.06, y), (tr + 0.28, y))
    hd = box(ax, tr + 0.34, y - 0.44, 1.40, 0.88, "Mean head", "odd symmetry",
             face=GOLD, edge=GOLD_E, fs=7.8, subfs=5.8)
    arrow(ax, (hd + 0.06, y), (hd + 0.28, y))
    df = box(ax, hd + 0.34, y - 0.26, 0.86, 0.52, "ΔF", "力速率 88 N/s", face="white",
             edge=EDGE, fs=7.8, subfs=5.8)
    df_mid = hd + 0.34 + 0.43

    if teacher:
        cc_mid = cc - 0.33          # concat centre (cc is its right edge)
        px = cc_mid - 0.875
        py = y - 1.18
        box(ax, px, py, 1.75, 0.52, "π = (θ, 对象状态)", "8 + 4 维 · 仅教师",
            face=LILAC, edge=LILAC_E, fs=7.2, subfs=6.0)
        arrow(ax, (cc_mid, py + 0.54), (cc_mid, y - 0.25), color=LILAC_E)

    # ---------------- closed loop as a labelled return path (no boxes)
    y_loop = 1.62
    ax.add_patch(FancyArrowPatch((df_mid, y - 0.28), (df_mid, y_loop + 0.02),
                                 arrowstyle="-", lw=0.9, color=EDGE, zorder=6))
    ax.plot([df_mid, 1.02], [y_loop, y_loop], color=EDGE, lw=0.9, zorder=6)
    ax.plot([1.02, 1.02], [y_loop, y - 0.62], color=EDGE, lw=0.9, zorder=6)
    ax.add_patch(FancyArrowPatch((1.02, y - 0.62), (1.24, y - 0.62),
                                 arrowstyle="-|>", mutation_scale=8, lw=0.9, color=EDGE,
                                 zorder=6))
    stages = [(7.30, "∫ dt", "F_req"), (5.95, "sat", "±22 N"),
              (4.55, "actuator", "τ = 20 ms"), (2.85, "品模 G_θ(s)", "θ 8 参数 · e^(−τ_p s)")]
    for x, name, sub in stages:
        ax.plot([x, x], [y_loop - 0.10, y_loop + 0.10], color=EDGE, lw=0.9, zorder=7)
        ax.text(x, y_loop + 0.16, name, ha="center", va="bottom", fontsize=7.2,
                color=INK, fontweight="bold")
        ax.text(x, y_loop - 0.16, sub, ha="center", va="top", fontsize=5.8, color=MUTED)
    ax.text(1.30, y - 0.72, "滚转率 p 回灌 32 步历史窗口", fontsize=6.4, color=MUTED)
    ax.text(0.34, y_loop + 0.16, "回路：", fontsize=7.2, color=MUTED, ha="left")

    title = ("教师：品模飞控闭环（π 只进教师）" if teacher
             else "学生：品模飞控闭环（无 π，仅本体感受）")
    ax.set_title(title, fontsize=10.4, color=INK, fontweight="bold", pad=6)
    ax.set_xlim(0.10, df + 0.12)
    ax.set_ylim(1.10, 4.55)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02)
    path = OUT / f"{kind}_network_fq.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path.with_suffix(".svg"))


if __name__ == "__main__":
    draw("teacher")
    draw("student")
