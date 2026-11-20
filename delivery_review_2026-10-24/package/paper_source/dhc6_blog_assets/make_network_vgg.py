"""VGG16-paper style diagrams: thin stacked plates in perspective.

    python dhc6_blog_assets/make_network_vgg.py

Reference style: the VGG16 architecture figure — each layer is a thin
parallelogram ("plate"); a convolution stage is several plates stacked with a
small offset; dense stages are tall plates.  Arrows are thin routed lines.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Polygon
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
EDGE = "#8a6a3f"
DEPTH, SKEW = 1.5, 0.55      # plate depth and vertical skew per depth unit

CREAM = ("#f7ecd2", "#e8d5ab")
ORANGE = ("#f2a86a", "#dd7f3c")
RED = ("#e2705f", "#c34f3d")
BLUE = ("#3c5b8c", "#2a4468")
PURPLE = ("#b9a3d8", "#9a83c0")


def plate(ax, x, y, h, palette, alpha=1.0, z=3):
    front, edge = palette
    verts = [(x, y), (x + DEPTH, y + SKEW * DEPTH),
             (x + DEPTH, y + SKEW * DEPTH + h), (x, y + h)]
    ax.add_patch(Polygon(verts, closed=True, facecolor=front, edgecolor=edge,
                         lw=0.9, alpha=alpha, zorder=z))


def stack(ax, x, y, h, n, palette, step=(0.30, 0.16), shrink=0.94):
    """n plates of a conv stage, each slightly smaller and offset back."""
    for i in range(n - 1, -1, -1):
        hh = h * (shrink ** i)
        plate(ax, x + step[0] * i, y + step[1] * i + (h - hh) / 2, hh, palette,
              alpha=0.96, z=3 + i)
    return x + step[0] * (n - 1) + DEPTH


def label(ax, x, y, name, shape, width=None):
    ax.text(x, y, name, ha="left", va="top", fontsize=9.6, color=INK,
            fontweight="bold")
    ax.text(x, y - 1.25, shape, ha="left", va="top", fontsize=8.0, color=MUTED)


def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                 lw=0.9, color="#5c6678", zorder=8,
                                 connectionstyle=f"arc3,rad={rad}"))


def shrink(path: Path, colors: int = 96) -> None:
    Image.open(path).convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT).save(
        path, optimize=True)


def draw(network: str) -> None:
    fig, ax = plt.subplots(figsize=(13.6, 6.6))
    ax.axis("off")
    y = 7.4
    if network == "teacher":
        groups = [
            ("in",   1.0, 4.4, 1, CREAM, "输入", "176 维"),
            ("conv", 1.0, 3.9, 3, ORANGE, "卷积 ×3", "5→24→24→24"),
            ("cat",  1.0, 4.2, 1, RED, "拼接", "76 维"),
            ("core", 1.0, 5.4, 6, BLUE, "主干", "512 × 8"),
            ("head", 1.0, 3.4, 1, CREAM, "均值头", "512 → 1"),
            ("out",  0.6, 1.5, 1, CREAM, "ΔF", "力速率"),
        ]
        x = 2.2
        gap = 2.5
        title = "教师网络：本体感受 + 特权输入，输出力速率"
        note = "每片薄板是一层，卷积阶段用多片叠放；深蓝为 8 个残差块"
    else:
        groups = [
            ("in",   1.0, 4.4, 1, CREAM, "输入", "176 维"),
            ("conv", 1.0, 3.6, 3, ORANGE, "卷积 ×3", "5→8→8→8"),
            ("cat",  1.0, 4.0, 1, RED, "拼接", "32 维"),
            ("core", 1.0, 4.4, 3, BLUE, "主干", "32 × 1"),
            ("head", 0.9, 3.0, 1, CREAM, "均值头", "32 → 1"),
            ("out",  0.5, 1.3, 1, CREAM, "ΔF", "力速率"),
        ]
        x = 2.6
        gap = 2.9
        title = "学生网络：只读本体感受与历史，不读 θ"
        note = "同样的层，薄板更窄、叠得更少：总计 3,856 参数"

    far = []
    for kind, w, h, n, pal, name, shape in groups:
        if n > 1:
            right = stack(ax, x, y, h, n, pal)
        else:
            plate(ax, x, y, h, pal)
            right = x + DEPTH
        label(ax, x, y - 0.9, name, shape)
        far.append((x, right, h))
        x = right + gap

    for (x0, x1, h), (x2, _, _) in zip(far, far[1:]):
        arrow(ax, (x1 + 0.25, y + h * 0.45), (x2 - 0.35, y + h * 0.45))

    # input arrow and output arrow
    arrow(ax, (1.0, y + 2.0), (far[0][0] - 0.3, y + 2.0))
    arrow(ax, (far[-1][1] + 0.2, y + 0.8), (far[-1][1] + 1.5, y + 0.8))

    if network == "teacher":
        px = 3.6
        stack(ax, px, y - 6.0, 2.6, 2, PURPLE, step=(0.28, 0.15))
        ax.text(px, y - 6.7, "特权 12", ha="left", va="top", fontsize=9.4,
                color=INK, fontweight="bold")
        ax.text(px, y - 7.9, "θ 8 + 状态 4", ha="left", va="top", fontsize=8.0,
                color=MUTED)
        arrow(ax, (px + DEPTH + 0.6, y - 5.4), (10.2, y + 0.9), rad=-0.12)

    ax.set_xlim(0, 34)
    ax.set_ylim(-0.5, 17)
    ax.set_title(title, fontsize=13.0, color=INK, fontweight="bold", pad=8)
    fig.tight_layout()
    path = OUT / f"{network}_network_vgg.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


if __name__ == "__main__":
    draw("teacher")
    draw("student")
