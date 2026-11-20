"""Tensor-shaped architecture diagram (feature maps + stacked layers).

    python dhc6_blog_assets/make_network_tensor.py

Instead of rounded boxes the figure draws what actually flows: a 32x5 history
grid, 24-channel convolution maps, the 76-D concat vector (48+9+7+12 segments)
and the 8-block trunk.  The closed loop is a faint dashed return path.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

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
EDGE = "#33445e"
HIST = "#f3d9a4"
HIST_E = "#c9a45f"
CONV = "#b9d2ee"
CONV_E = "#7796bd"
VEC = "#c9b8e8"
VEC_E = "#8f76bd"
CORE = "#8fb3d9"
CORE_E = "#4f739c"
HEAD = "#f2b98a"
HEAD_E = "#c07f4d"
PRIV = "#d6c7f0"
PRIV_E = "#9a83c8"


def arrow(ax, p, q, color=EDGE, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=lw,
                                 color=color, ls=ls, zorder=7))


def grid_block(ax, x, y, w, h, rows, cols, face, edge):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, lw=0.9, zorder=3))
    for r in range(1, rows):
        ax.plot([x, x + w], [y + h * r / rows] * 2, color="white", lw=0.35, alpha=0.75, zorder=4)
    for c in range(1, cols):
        ax.plot([x + w * c / cols] * 2, [y, y + h], color="white", lw=0.35, alpha=0.75, zorder=4)


def slabs(ax, x, y, w, h, n, face, edge, dx=0.055, dy=0.055):
    """n stacked layers (channels / blocks) seen at a slight angle."""
    for i in range(n - 1, -1, -1):
        ax.add_patch(Rectangle((x + dx * i, y + dy * i), w, h, facecolor=face,
                               edgecolor=edge, lw=0.55, alpha=0.95, zorder=2 + i))
    return x + w + dx * (n - 1)


def draw(kind: str) -> None:
    teacher = kind == "teacher"
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    ax.axis("off")
    y0 = 2.05

    # ---- 1. history window 32 x 5
    grid_block(ax, 0.35, y0 - 0.55, 0.62, 1.10, rows=16, cols=5, face=HIST, edge=HIST_E)
    ax.text(0.66, y0 - 0.70, "32×5 历史", ha="center", va="top", fontsize=6.8, color=INK)
    ax.text(0.66, y0 - 0.90, "cmd p_ref p e F", ha="center", va="top", fontsize=5.6, color=MUTED)
    arrow(ax, (1.02, y0), (1.24, y0))

    # ---- 2. conv feature maps 24 x 32
    x = 1.28
    for i, n in enumerate((6, 6, 6)):
        r = slabs(ax, x, y0 - 0.50, 0.30, 1.00, n, CONV, CONV_E)
        if i == 0:
            ax.text(x + 0.15, y0 - 0.62, "24×32", ha="center", va="top",
                    fontsize=5.6, color=MUTED)
        x = r + 0.16
    ax.text(1.28, y0 + 0.86, "Conv1D ×3　D 1/4/8", ha="left", va="bottom", fontsize=6.6,
            color=INK)
    arrow(ax, (x + 0.02, y0), (x + 0.22, y0))

    # ---- 3. concat vector 76 = 48 + 9 + 7 (+12)
    cv_x = x + 0.26
    segs = [(0.52, CONV, CONV_E, "48"), (0.11, HEAD, HEAD_E, "9"),
            (0.09, HEAD, HEAD_E, "7")]
    if teacher:
        segs.append((0.15, PRIV, PRIV_E, "12"))
    total = sum(s[0] for s in segs)
    yy = y0 - total / 2
    for h, face, edge, lab in segs:
        ax.add_patch(Rectangle((cv_x, yy), 0.30, h, facecolor=face, edgecolor=edge,
                               lw=0.7, zorder=4))
        ax.text(cv_x + 0.40, yy + h / 2, lab, ha="left", va="center", fontsize=5.4,
                color=MUTED, zorder=5)
        yy += h
    ax.text(cv_x + 0.15, y0 - total / 2 - 0.14, "concat 76", ha="center", va="top",
            fontsize=6.4, color=INK)

    if teacher:
        ax.text(cv_x + 0.15, y0 + total / 2 + 0.42, "π 12 维",
                ha="center", va="bottom", fontsize=6.0, color=PRIV_E, fontweight="bold")
        ax.text(cv_x + 0.15, y0 + total / 2 + 0.24, "仅教师",
                ha="center", va="bottom", fontsize=5.4, color=PRIV_E)
        arrow(ax, (cv_x + 0.15, y0 + total / 2 + 0.21), (cv_x + 0.15, y0 + total / 2 + 0.01),
              color=PRIV_E, lw=1.0)

    arrow(ax, (cv_x + 0.55, y0), (cv_x + 0.78, y0))

    # ---- 4. trunk: 8 residual blocks
    tx = cv_x + 0.84
    r = slabs(ax, tx, y0 - 0.55, 0.42, 1.10, 8, CORE, CORE_E, dx=0.075, dy=0.06)
    ax.text(tx + 0.21, y0 - 0.68, "8 × 残差块", ha="center", va="top", fontsize=6.4, color=INK)
    ax.text(tx + 0.21, y0 + 0.90, "512 宽", ha="center", va="bottom", fontsize=6.4, color=INK)
    arrow(ax, (r + 0.04, y0), (r + 0.24, y0))

    # ---- 5. head -> scalar output
    hx = r + 0.28
    ax.add_patch(Polygon([(hx, y0 - 0.55), (hx + 0.55, y0 - 0.10),
                          (hx + 0.55, y0 + 0.10), (hx, y0 + 0.55)], closed=True,
                         facecolor=HEAD, edgecolor=HEAD_E, lw=0.8, zorder=3))
    ax.text(hx + 0.27, y0 - 0.72, "均值头 512→1", ha="center", va="top", fontsize=6.2, color=INK)
    ax.text(hx + 0.27, y0 + 0.62, "奇对称投影", ha="center", va="bottom", fontsize=5.6,
            color=MUTED)
    arrow(ax, (hx + 0.60, y0), (hx + 0.82, y0))
    ax.add_patch(Rectangle((hx + 0.86, y0 - 0.09), 0.20, 0.18, facecolor="white",
                           edgecolor=EDGE, lw=0.8, zorder=4))
    ax.text(hx + 1.16, y0, "ΔF\n88 N/s", ha="left", va="center", fontsize=5.8, color=INK)

    # ---- 6. faint closed loop with the flight-quality plant
    loop_y = 0.62
    ax.plot([hx + 0.96, hx + 0.96], [y0 - 0.12, loop_y], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot([hx + 0.96, 0.66], [loop_y, loop_y], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot([0.66, 0.66], [loop_y, y0 - 0.62], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for x, name, sub in ((hx + 0.10, "∫dt → sat ±22 N", "20 ms 作动器"),
                         (3.6, "品模 G_θ(s)", "8 参数 · e^(−τ_p s)")):
        ax.text(x, loop_y - 0.06, name, ha="center", va="top", fontsize=5.6, color="#6f7f96")
        ax.text(x, loop_y - 0.24, sub, ha="center", va="top", fontsize=5.2, color="#5b6c85")

    title = ("教师：32 步历史 → 卷积 → 拼接(含 π) → 8 残差块 → 力速率"
             if teacher else "学生：32 步历史 → 卷积 → 拼接(无 π) → 1 残差块 → 力速率")
    ax.set_title(title, fontsize=9.6, color=INK, fontweight="bold", pad=6)
    ax.set_xlim(0.20, hx + 1.75)
    ax.set_ylim(0.05, 3.30)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)
    path = OUT / f"{kind}_network_tensor.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path.with_suffix(".svg"))


if __name__ == "__main__":
    draw("teacher")
    draw("student")
