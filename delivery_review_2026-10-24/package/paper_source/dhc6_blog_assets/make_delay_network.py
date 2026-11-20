"""Style-matched network diagram for the delay-representation experiment.

    python dhc6_blog_assets/make_delay_network.py

Uses the same primitives, palette and typography as
``make_network_tensor.py`` so Figure 16 does not introduce a second visual
language next to the teacher/student diagrams.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

import make_network_tensor as NT

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets"

INK = NT.INK
MUTED = NT.MUTED
EDGE = NT.EDGE


def concat(ax, x, y, segments, width=0.26, label=""):
    total = sum(h for h, _, _, _ in segments)
    yy = y - total / 2
    for h, face, edge, text in segments:
        ax.add_patch(Rectangle((x, yy), width, h, facecolor=face, edgecolor=edge,
                               lw=0.7, zorder=4))
        if text:
            ax.text(x + width + 0.04, yy + h / 2, text, ha="left", va="center",
                    fontsize=5.4, color=MUTED, zorder=5)
        yy += h
    if label:
        ax.text(x + width / 2, y - total / 2 - 0.13, label, ha="center", va="top",
                fontsize=5.8, color=INK)
    return x + width


def head(ax, x, y, w=0.44, h=0.70, face=NT.HEAD, edge=NT.HEAD_E):
    ax.add_patch(Polygon([(x, y - h / 2), (x + w, y - 0.09),
                          (x + w, y + 0.09), (x, y + h / 2)], closed=True,
                         facecolor=face, edgecolor=edge, lw=0.8, zorder=3))


def output(ax, x, y):
    ax.add_patch(Rectangle((x, y - 0.09), 0.17, 0.18, facecolor="white",
                           edgecolor=EDGE, lw=0.8, zorder=4))
    ax.text(x + 0.25, y, "ΔF\n88 N/s", ha="left", va="center", fontsize=5.8, color=INK)


def sparse_grid(ax, x, y, w=0.40, h=0.60, filled=(2, 6, 9, 13)):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=NT.VEC, edgecolor=NT.VEC_E,
                           lw=0.8, zorder=3))
    cw, ch = w / 8, h / 2
    for i in range(16):
        cx = x + (i % 8) * cw
        cy = y + (i // 8) * ch
        if i in filled:
            ax.add_patch(Rectangle((cx + 0.012, cy + 0.015), cw - 0.024, ch - 0.03,
                                   facecolor=NT.VEC_E, edgecolor="none", zorder=4))
        else:
            ax.plot([cx, cx + cw], [cy, cy], color="white", lw=0.3, alpha=0.7, zorder=4)
            ax.plot([cx, cx], [cy, cy + ch], color="white", lw=0.3, alpha=0.7, zorder=4)


def draw() -> None:
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 11.0)
    ax.set_ylim(0, 5.8)

    # Shared input.
    NT.grid_block(ax, 0.34, 2.35, 0.56, 1.15, rows=16, cols=5,
                  face=NT.HIST, edge=NT.HIST_E)
    ax.text(0.62, 2.20, "32×5 历史", ha="center", va="top", fontsize=6.6, color=INK)
    ax.text(0.62, 2.06, "cmd p_ref p e F", ha="center", va="top", fontsize=5.4,
            color=MUTED)

    # ---------------------------------------------------------------- raw path
    ty = 4.55
    ax.text(1.12, 5.45, "对照：部署学生（原始历史，图 3）", fontsize=6.8,
            color=MUTED, ha="left", va="bottom")
    NT.arrow(ax, (0.90, 3.50), (1.12, 4.38), color="#9aa8bd", lw=0.8)
    NT.slabs(ax, 1.14, 4.13, 0.26, 0.84, 6, "#e6eaf0", "#9aa8bd",
             dx=0.045, dy=0.045)
    ax.text(1.68, 4.92, "Conv1D ×3", ha="left", va="center", fontsize=5.9,
            color=MUTED)
    ax.text(1.68, 4.78, "5→8→8→8", ha="left", va="center", fontsize=5.4,
            color=MUTED)
    ax.add_patch(Rectangle((2.30, ty - 0.23), 0.16, 0.46, facecolor="#e6eaf0",
                           edgecolor="#9aa8bd", lw=0.7, zorder=3))
    ax.text(2.38, ty - 0.31, "c (16)", ha="center", va="top", fontsize=5.3, color=MUTED)
    NT.arrow(ax, (2.50, ty), (2.72, ty), color="#9aa8bd")
    concat(ax, 2.74, ty,
           [(0.54, "#e6eaf0", "#9aa8bd", "16"),
            (0.17, "#e6eaf0", "#9aa8bd", "9"),
            (0.13, "#e6eaf0", "#9aa8bd", "7")],
           width=0.24, label="concat 32")
    NT.arrow(ax, (3.04, ty), (3.34, ty), color="#9aa8bd")
    NT.slabs(ax, 3.36, 4.13, 0.36, 0.84, 1, "#e6eaf0", "#9aa8bd")
    ax.text(3.54, 4.02, "1 残差块", ha="center", va="top", fontsize=5.6, color=MUTED)
    NT.arrow(ax, (3.78, ty), (4.00, ty), color="#9aa8bd")
    head(ax, 4.02, ty, face="#e6eaf0", edge="#9aa8bd")
    ax.text(4.24, 4.02, "均值头", ha="center", va="top", fontsize=5.6, color=MUTED)
    NT.arrow(ax, (4.50, ty), (4.72, ty), color="#9aa8bd")
    output(ax, 4.74, ty)

    # ---------------------------------------------------------------- SAE path
    sy = 1.95
    ax.text(1.12, 2.63, "SAE 延迟实验（冻结编码器）", fontsize=6.8, color=INK,
            ha="left", va="bottom")
    NT.arrow(ax, (0.90, 2.35), (1.12, 2.12), color=EDGE, lw=0.9)
    NT.slabs(ax, 1.14, 1.51, 0.28, 0.88, 6, NT.CONV, NT.CONV_E,
             dx=0.050, dy=0.050)
    ax.text(1.14, 1.39, "SAE 160→64→16", ha="left", va="top", fontsize=5.9, color=INK)
    ax.text(1.14, 1.25, "L1 / top-k · 冻结", ha="left", va="top", fontsize=5.3,
            color=MUTED)
    NT.arrow(ax, (2.40, sy), (2.62, sy))
    sparse_grid(ax, 2.64, sy - 0.30)
    ax.text(2.84, 2.34, "z (16) · L0 ≈ 4–7", ha="center", va="bottom",
            fontsize=5.7, color=INK)
    NT.arrow(ax, (3.10, sy), (3.32, sy))
    concat(ax, 3.34, sy,
           [(0.54, NT.VEC, NT.VEC_E, "16"),
            (0.17, NT.HEAD, NT.HEAD_E, "9"),
            (0.13, NT.HEAD, NT.HEAD_E, "7")],
           width=0.26, label="concat 32")
    NT.arrow(ax, (3.66, sy), (3.94, sy))
    NT.slabs(ax, 3.96, 1.53, 0.38, 0.84, 1, NT.CORE, NT.CORE_E)
    ax.text(4.15, 1.42, "1 残差块", ha="center", va="top", fontsize=5.6, color=INK)
    NT.arrow(ax, (4.40, sy), (4.64, sy))
    head(ax, 4.66, sy)
    ax.text(4.88, 1.42, "均值头", ha="center", va="top", fontsize=5.6, color=INK)
    NT.arrow(ax, (5.14, sy), (5.38, sy))
    output(ax, 5.40, sy)

    # Probe branch, drawn downward from z.
    ax.text(3.06, 0.94, "延迟读出", fontsize=5.6, color=MUTED, ha="left", va="bottom")
    ax.plot([2.84, 2.84], [sy - 0.30, 1.00], color=EDGE, lw=0.8, zorder=2)
    NT.arrow(ax, (2.84, 1.02), (2.84, 0.86), lw=0.8)
    ax.add_patch(Rectangle((2.52, 0.58), 0.64, 0.26, facecolor=NT.PRIV,
                           edgecolor=NT.PRIV_E, lw=0.7, zorder=3))
    ax.text(2.84, 0.71, "线性探针", ha="center", va="center", fontsize=5.5, color=INK)
    ax.text(2.84, 0.50, "16 → 1", ha="center", va="top", fontsize=5.1, color=MUTED)
    NT.arrow(ax, (3.18, 0.71), (3.42, 0.71), lw=0.8)
    ax.text(3.48, 0.71, "估计 τp", ha="left", va="center", fontsize=5.9, color=INK)

    # Baseline note, no box; same muted typography as the existing diagrams.
    ax.text(6.10, 3.72, "对照表征", fontsize=6.2, color=MUTED, fontweight="bold")
    for i, line in enumerate(("原始历史 160", "PCA-16", "稠密 AE-16", "互相关-36")):
        ax.text(6.10, 3.50 - i * 0.21, "· " + line, fontsize=5.8, color=MUTED)
    ax.text(6.10, 2.52, "SAE 只做表征；策略头/探针在冻结潜码上训练。",
            fontsize=5.6, color=MUTED)

    ax.set_title("延迟表征实验：原始历史（上）与冻结 SAE（下）两条前端",
                 fontsize=9.6, color=INK, fontweight="bold", pad=8)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    path = OUT / "delay_network.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path.with_suffix(".svg"))


if __name__ == "__main__":
    draw()
