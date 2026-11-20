"""Flat (tiled) network diagrams — compact, centred, gridded.

    python dhc6_blog_assets/make_network_flat.py

Each layer is a rectangle carrying a feature-map style grid; convolution stages
are three offset slices.  The axes limits are computed from the content so the
row is centred with no dead space.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#16233c"
MUTED = "#16233a"
EDGE = "#42506b"
GRID = "#ffffff"

FILL = {
    "in":   ("#cdd9ef", "#9fb4d6"),
    "conv": ("#9dc0ea", "#5f8cc4"),
    "cat":  ("#bfb2e2", "#8d7bc0"),
    "core": ("#6f9fd8", "#3f6ba3"),
    "head": ("#f2bd84", "#cf8e4e"),
    "out":  ("#e89b6a", "#bf6a3c"),
    "priv": ("#a98fd6", "#7d63b0"),
}


def flat(ax, x, w, h, kind, y0, slices=1, step=(0.16, 0.16), cells=(4, 5)):
    """Rectangle with an inner grid; ``slices`` stacks offset copies."""
    face, deep = FILL[kind]
    for i in range(slices - 1, -1, -1):
        bx, by = x + step[0] * i, y0 + step[1] * i
        ax.add_patch(Rectangle((bx, by), w, h, facecolor=face, edgecolor=deep,
                               lw=0.9, alpha=1.0, zorder=3 + i))
        nx, ny = cells
        for k in range(1, nx):
            ax.plot([bx + w * k / nx] * 2, [by, by + h], color=GRID, lw=0.6,
                    alpha=0.85, zorder=4 + i)
        for k in range(1, ny):
            ax.plot([bx, bx + w], [by + h * k / ny] * 2, color=GRID, lw=0.6,
                    alpha=0.85, zorder=4 + i)
        ax.add_patch(Rectangle((bx, by), w, h, facecolor="none", edgecolor=deep,
                               lw=0.9, zorder=5 + i))
    return x + w + step[0] * (slices - 1)


def caption(ax, x, w, name, shape, y0):
    ax.text(x + w / 2, y0 - 0.30, name, ha="center", va="top", fontsize=9.4,
            color=INK, fontweight="bold")
    ax.text(x + w / 2, y0 - 1.02, shape, ha="center", va="top", fontsize=7.6,
            color=MUTED)


def arrow(ax, x0, x1, y, color="#5c6678"):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.0, color=color, zorder=9))


def shrink(path: Path, colors: int = 96) -> None:
    Image.open(path).convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT).save(
        path, optimize=True)


def draw(network: str) -> None:
    fig, ax = plt.subplots(figsize=(12.2, 3.55))
    ax.axis("off")
    y0 = 2.55
    if network == "teacher":
        rows = [
            (0.95, 2.55, "in", "输入", "176 维", 1, (4, 5)),
            (0.95, 2.15, "conv", "卷积 ×3", "5→24→24→24", 3, (4, 4)),
            (0.95, 2.40, "cat", "拼接", "76 维", 1, (4, 5)),
            (1.55, 3.35, "core", "主干", "512 × 8", 1, (6, 7)),
            (0.95, 1.95, "head", "均值头", "512 → 1", 1, (4, 4)),
            (0.50, 0.80, "out", "ΔF", "力速率", 1, (2, 2)),
        ]
        gap = 1.15
        title = "教师网络（平铺）：本体感受 + 特权信息（privileged information），输出力速率"
    else:
        rows = [
            (0.95, 2.55, "in", "输入", "176 维", 1, (4, 5)),
            (0.95, 1.95, "conv", "卷积 ×3", "5→8→8→8", 3, (4, 4)),
            (0.95, 2.20, "cat", "拼接", "32 维", 1, (4, 5)),
            (1.25, 2.55, "core", "主干", "32 × 1", 1, (5, 5)),
            (0.85, 1.70, "head", "均值头", "32 → 1", 1, (3, 3)),
            (0.45, 0.70, "out", "ΔF", "力速率", 1, (2, 2)),
        ]
        gap = 1.45
        title = "学生网络（平铺）：只读本体感受与历史，不读 θ"

    x = 0.0
    spans = []
    for w, h, kind, name, shape, sl, cells in rows:
        right = flat(ax, x, w, h, kind, y0, slices=sl, cells=cells)
        spans.append((x, right, h))
        caption(ax, x, w, name, shape, y0)
        x = right + gap
    x_end = spans[-1][1]

    for (x0, x1, h), (x2, _, _) in zip(spans, spans[1:]):
        arrow(ax, x1 + 0.12, x2 - 0.16, y0 + h * 0.5)
    arrow(ax, -0.75, spans[0][0] - 0.12, y0 + 1.15)
    arrow(ax, x_end + 0.12, x_end + 0.75, y0 + 0.40)

    bottom = y0 - 1.35
    if network == "teacher":
        px, pw, ph = 0.35, 0.95, 1.35
        flat(ax, px, pw, ph, "priv", y0 - 3.25, slices=2, cells=(3, 4))
        caption(ax, px, pw, "特权信息", "θ 8 + 状态 4", y0 - 3.25)
        ax.add_patch(FancyArrowPatch((px + pw + 0.42, y0 - 2.85),
                                     (4.50, y0 + 0.10), arrowstyle="-|>",
                                     mutation_scale=9, lw=1.0, color=FILL["priv"][1],
                                     connectionstyle="arc3,rad=0.28", zorder=9))
        bottom = y0 - 4.45

    top = y0 + max(h for _, _, h in spans) + 0.60
    ax.set_xlim(-0.95, x_end + 0.95)
    ax.set_ylim(bottom, top)
    ax.set_title(title, fontsize=11.6, color=INK, fontweight="bold", pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.03)
    path = OUT / f"{network}_network_flat.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


if __name__ == "__main__":
    draw("teacher")
    draw("student")
