"""PlotNeuralNet-style 3D block diagrams (teacher / student).

    python dhc6_blog_assets/make_network_cuboid.py

Each layer is a translucent cuboid: front face = layer height, depth = channels,
offset along an oblique depth axis.  This is the style used by PlotNeuralNet,
NN-SVG's AlexNet mode and ConvNetDraw.
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
EDGE = "#42506b"

DEPTH = (0.62, 0.42)      # oblique depth axis per unit of d

COL = {
    "in":   ("#cdd9ef", "#e3ebf8", "#b3c4e0"),
    "conv": ("#9dc0ea", "#c3d9f3", "#7fa6d8"),
    "cat":  ("#bfb2e2", "#d8cff0", "#a294cd"),
    "core": ("#6f9fd8", "#9dc0ea", "#5484c2"),
    "head": ("#f2bd84", "#f8d6b3", "#dda265"),
    "out":  ("#e89b6a", "#f3c3a4", "#cf7f4c"),
    "priv": ("#a98fd6", "#c9b8e8", "#8d6fbe"),
}


def cuboid(ax, x, y, w, h, d, kind, alpha=0.92):
    front, top, side = COL[kind]
    dx, dy = DEPTH[0] * d, DEPTH[1] * d
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                         closed=True, facecolor=front, edgecolor=EDGE, lw=1.05,
                         alpha=alpha, zorder=3))
    ax.add_patch(Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy),
                          (x + dx, y + h + dy)],
                         closed=True, facecolor=top, edgecolor=EDGE, lw=1.05,
                         alpha=alpha, zorder=2))
    ax.add_patch(Polygon([(x + w, y), (x + w, y + h), (x + w + dx, y + h + dy),
                          (x + w + dx, y + dy)],
                         closed=True, facecolor=side, edgecolor=EDGE, lw=1.05,
                         alpha=alpha, zorder=2))
    return x + w + dx


def caption(ax, x, w, y, name, shape):
    ax.text(x + w / 2, y - 1.4, name, ha="center", va="top", fontsize=10.2,
            color=INK, fontweight="bold", zorder=5)
    ax.text(x + w / 2, y - 3.4, shape, ha="center", va="top", fontsize=8.4,
            color=MUTED, zorder=5)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=EDGE, zorder=4))


def shrink(path: Path, colors: int = 128) -> None:
    img = Image.open(path).convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT)
    img.save(path, optimize=True)


def teacher() -> None:
    fig, ax = plt.subplots(figsize=(14.2, 7.4))
    ax.axis("off")
    base_y = 16.0
    x = 4.0
    h, d = 7.2, 1.6
    blocks = [
        (2.2, 6.4, 0.7, "in", "输入", "176 维"),
        (2.6, 7.2, 1.5, "conv", "卷积 ×3", "5→24→24→24"),
        (2.4, 7.4, 1.0, "cat", "拼接", "76 维"),
        (3.4, 9.2, 1.9, "core", "主干", "512 × 8"),
        (2.2, 6.6, 1.1, "head", "均值头", "512 → 1"),
        (1.2, 4.0, 0.5, "out", "ΔF", "力速率"),
    ]
    centers = []
    for w, hh, dd, kind, name, shape in blocks:
        right = cuboid(ax, x, base_y, w, hh, dd, kind)
        centers.append((x, w, hh))
        caption(ax, x, w, base_y, name, shape)
        x = right + 2.2
    for (cx, cw, ch), nxt in zip(centers, centers[1:]):
        arrow(ax, cx + cw + 0.35, nxt[0] - 0.45, base_y + ch / 2)

    # privileged branch, feeding the concat block
    px, pw, ph = 13.4, 2.0, 4.2
    cuboid(ax, px, base_y - 11.2, pw, ph, 1.2, "priv")
    ax.text(px + pw / 2, base_y - 11.2 - 1.3, "特权 12", ha="center", va="top",
            fontsize=10.0, color=INK, fontweight="bold")
    ax.text(px + pw / 2, base_y - 11.2 - 3.3, "θ 8 + 状态 4", ha="center", va="top",
            fontsize=8.4, color=MUTED)
    ax.add_patch(FancyArrowPatch((px + pw / 2, base_y - 11.2 + ph + 0.4),
                                 (14.9, base_y + 1.4), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.3, color=COL["priv"][2],
                                 connectionstyle="arc3,rad=0.18", zorder=4))

    ax.set_xlim(0, 34)
    ax.set_ylim(-3.4, 30)
    ax.set_title("教师网络：本体感受 + 特权输入，输出力速率", fontsize=13.5,
                 color=INK, fontweight="bold", pad=10)
    ax.text(17, -1.6, "立方块正面 = 层高，深度 = 通道/宽度；紫色块只进教师，不部署",
            ha="center", fontsize=9.2, color=MUTED)
    fig.tight_layout()
    path = OUT / "teacher_network_blocks.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


def student() -> None:
    fig, ax = plt.subplots(figsize=(13.4, 7.0))
    ax.axis("off")
    base_y = 17.0
    x = 5.0
    blocks = [
        (2.2, 6.4, 0.7, "in", "输入", "176 维"),
        (2.4, 6.8, 0.9, "conv", "卷积 ×3", "5→8→8→8"),
        (2.2, 7.0, 0.7, "cat", "拼接", "32 维"),
        (2.6, 7.6, 1.0, "core", "主干", "32 × 1"),
        (1.4, 4.6, 0.5, "out", "ΔF", "力速率"),
    ]
    centers = []
    for w, hh, dd, kind, name, shape in blocks:
        right = cuboid(ax, x, base_y, w, hh, dd, kind)
        centers.append((x, w, hh))
        caption(ax, x, w, base_y, name, shape)
        x = right + 2.8
    for (cx, cw, ch), nxt in zip(centers, centers[1:]):
        arrow(ax, cx + cw + 0.35, nxt[0] - 0.45, base_y + ch / 2)
    ax.text(13.6, 4.6, "教师 4.26M → 学生 3,856，压缩 1,100 倍；同分布误差 1.28×",
            ha="center", fontsize=10.4, color=INK)
    ax.set_xlim(0, 30)
    ax.set_ylim(-3.4, 30)
    ax.set_title("学生网络：只读本体感受与历史，不读 θ", fontsize=13.5,
                 color=INK, fontweight="bold", pad=10)
    fig.tight_layout()
    path = OUT / "student_network_blocks.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


if __name__ == "__main__":
    teacher()
    student()
