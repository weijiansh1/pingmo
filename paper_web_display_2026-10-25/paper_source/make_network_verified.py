"""Draw verified teacher and tiny-student diagrams from their model dimensions.

Run ``python paper_source/make_network_verified.py`` from the overlay directory.
The teacher uses 24-channel convolutions and a 512-wide, eight-block trunk;
the tiny student uses eight-channel convolutions and one 32-wide block.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

WS = Path(__file__).resolve().parents[1]
OUT = WS / "webpage" / "paper"

FONT = Path(__file__).with_name("network_verified_font.ttf")
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["svg.hashsalt"] = "verified-flight-control-network"
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
    conv_width = 24 if teacher else 8
    trunk_width = 512 if teacher else 32
    block_count = 8 if teacher else 1
    history_features = 2 * conv_width
    concat_dim = history_features + 9 + 7 + (12 if teacher else 0)
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    ax.axis("off")
    y0 = 2.05

    # ---- 1. history window 32 x 5
    grid_block(ax, 0.35, y0 - 0.55, 0.62, 1.10, rows=32, cols=5, face=HIST, edge=HIST_E)
    ax.text(0.66, y0 - 0.70, "32×5 历史", ha="center", va="top", fontsize=7.5, color=INK)
    ax.text(0.66, y0 - 0.90, r"$p_{\rm cmd}\;p_{\rm ref}\;p\;e\;F$", ha="center", va="top", fontsize=5.6, color=MUTED)
    arrow(ax, (1.02, y0), (1.24, y0))

    # ---- 2. conv feature maps 24 x 32
    x = 1.28
    for i, n in enumerate((conv_width, conv_width, conv_width)):
        r = slabs(ax, x, y0 - 0.50, 0.30, 1.00, n, CONV, CONV_E, dx=0.275 / (n - 1), dy=0.275 / (n - 1))
        if i == 0:
            ax.text(x + 0.15, y0 - 0.62, f"{conv_width}×32", ha="center", va="top",
                    fontsize=5.6, color=MUTED)
        x = r + 0.16
    ax.text(1.28, y0 + 0.86, "Conv1D ×3　D 1/4/8", ha="left", va="bottom", fontsize=7.5,
            color=INK)
    arrow(ax, (x + 0.02, y0), (x + 0.22, y0))
    ax.text(x + 0.14, y0 - 0.98, "last + mean", ha="center", va="top", fontsize=5.3, color=MUTED)

    # ---- 3. concat vector 76 = 48 + 9 + 7 (+12)
    cv_x = x + 0.26
    segs = [(1.10 * history_features / concat_dim, CONV, CONV_E, str(history_features)),
            (1.10 * 9 / concat_dim, HEAD, HEAD_E, "9"),
            (1.10 * 7 / concat_dim, HEAD, HEAD_E, "7")]
    if teacher:
        segs.append((1.10 * 12 / concat_dim, PRIV, PRIV_E, "12"))
    total = sum(s[0] for s in segs)
    yy = y0 - total / 2
    for h, face, edge, lab in segs:
        ax.add_patch(Rectangle((cv_x, yy), 0.30, h, facecolor=face, edgecolor=edge,
                               lw=0.7, zorder=4))
        ax.text(cv_x + 0.40, yy + h / 2, lab, ha="left", va="center", fontsize=5.4,
                color=MUTED, zorder=5)
        yy += h
    ax.text(cv_x + 0.15, y0 - total / 2 - 0.14, f"concat {concat_dim}", ha="center", va="top",
            fontsize=7.3, color=INK)

    if teacher:
        ax.text(cv_x + 0.15, y0 + total / 2 + 0.42, "特权信息 z · 12 维",
                ha="center", va="bottom", fontsize=6.0, color=PRIV_E, fontweight="bold")
        ax.text(cv_x + 0.15, y0 + total / 2 + 0.24, "8 参数 + 4 维状态",
                ha="center", va="bottom", fontsize=5.4, color=PRIV_E)
        arrow(ax, (cv_x + 0.15, y0 + total / 2 + 0.21), (cv_x + 0.15, y0 + total / 2 + 0.01),
              color=PRIV_E, lw=1.0)

    arrow(ax, (cv_x + 0.55, y0), (cv_x + 0.78, y0))

    # ---- 4. trunk: 8 residual blocks
    tx = cv_x + 0.84
    r = slabs(ax, tx, y0 - 0.55, 0.42, 1.10, block_count, CORE, CORE_E, dx=0.075, dy=0.06)
    ax.text(tx + 0.21, y0 - 0.68, f"{block_count} × 残差块", ha="center", va="top", fontsize=7.3, color=INK)
    ax.text(tx + 0.21, y0 + 0.90, f"{trunk_width} 宽", ha="center", va="bottom", fontsize=7.3, color=INK)
    arrow(ax, (r + 0.04, y0), (r + 0.24, y0))

    # ---- 5. head -> scalar output
    hx = r + 0.28
    ax.add_patch(Polygon([(hx, y0 - 0.55), (hx + 0.55, y0 - 0.10),
                          (hx + 0.55, y0 + 0.10), (hx, y0 + 0.55)], closed=True,
                         facecolor=HEAD, edgecolor=HEAD_E, lw=0.8, zorder=3))
    ax.text(hx + 0.27, y0 - 0.72, f"输出头 {trunk_width}→1", ha="center", va="top", fontsize=7.1, color=INK)
    ax.text(hx + 0.27, y0 + 0.62, "奇对称投影 + tanh", ha="center", va="bottom", fontsize=5.6,
            color=MUTED)
    arrow(ax, (hx + 0.60, y0), (hx + 0.82, y0))
    ax.add_patch(Rectangle((hx + 0.86, y0 - 0.09), 0.20, 0.18, facecolor="white",
                           edgecolor=EDGE, lw=0.8, zorder=4))
    ax.text(hx + 1.16, y0, r"$a_t$" + "\n[-1, 1]", ha="left", va="center", fontsize=5.8, color=INK)

    # ---- 6. faint closed loop with the flight-quality plant
    loop_y = 0.62
    ax.plot([hx + 0.96, hx + 0.96], [y0 - 0.12, loop_y], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot([hx + 0.96, 0.66], [loop_y, loop_y], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.plot([0.66, 0.66], [loop_y, y0 - 0.62], color="#9aa8bd", lw=0.8, ls=(0, (4, 3)), zorder=1)
    for x, name, sub in ((hx + 0.10, "88 N/s → 积分、限幅 ±22 N", "20 ms 执行机构"),
                         (3.6, r"对象 $G_\theta(s)$", r"8 参数 · $e^{-\tau_p s}$")):
        ax.text(x, loop_y - 0.06, name, ha="center", va="top", fontsize=5.6, color="#6f7f96")
        ax.text(x, loop_y - 0.24, sub, ha="center", va="top", fontsize=5.2, color="#5b6c85")

    title = ("特权教师 Actor · 4.26 M 参数" if teacher else "tiny 统一学生 · 3,856 参数")
    ax.set_title(title, fontsize=9.6, color=INK, fontweight="bold", pad=6)
    ax.set_xlim(0.20, hx + 1.75)
    ax.set_ylim(0.05, 3.30)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)
    path = OUT / f"{kind}_network_tensor.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white", metadata={"Date": None})
    fig.savefig(path.with_suffix(".pdf"), facecolor="white", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path.with_suffix(".svg"))


if __name__ == "__main__":
    draw("teacher")
    draw("student")
