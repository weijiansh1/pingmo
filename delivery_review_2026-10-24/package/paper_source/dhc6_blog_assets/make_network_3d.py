"""Isometric (pseudo-3D) network diagrams for the Teacher and the Student.

    python dhc6_blog_assets/make_network_3d.py

Layers are drawn as slabs with a front face, a top face and a side face, connected
by thin lines; no real perspective, just an isometric offset, which keeps the
labels readable.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Polygon

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
EDGE = "#5b6b85"
DX, DY = 1.5, 1.0          # isometric offset

PALETTE = {
    "in":   ("#dbe7fa", "#eef4fd", "#c3d4ee"),
    "net":  ("#d8e6ff", "#eaf1ff", "#bed2f2"),
    "priv": ("#e4dcf7", "#efe9fb", "#cfc3ec"),
    "head": ("#fbe3c6", "#fdf0e0", "#efcfa8"),
}


def slab(ax, cx, cy, w, h, kind, label, sub=""):
    front, top, side = PALETTE[kind]
    ax.add_patch(Polygon([(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2),
                          (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)],
                         closed=True, facecolor=front, edgecolor=EDGE, lw=1.0, zorder=2))
    ax.add_patch(Polygon([(cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2),
                          (cx + w / 2 + DX, cy + h / 2 + DY),
                          (cx - w / 2 + DX, cy + h / 2 + DY)],
                         closed=True, facecolor=top, edgecolor=EDGE, lw=1.0, zorder=1))
    ax.add_patch(Polygon([(cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2),
                          (cx + w / 2 + DX, cy + h / 2 + DY),
                          (cx + w / 2 + DX, cy - h / 2 + DY)],
                         closed=True, facecolor=side, edgecolor=EDGE, lw=1.0, zorder=1))
    ax.text(cx, cy + 1.2, label, ha="center", va="center", fontsize=9.6,
            color=INK, fontweight="bold", zorder=3)
    if sub:
        ax.text(cx, cy - 2.0, sub, ha="center", va="center", fontsize=8.0,
                color=MUTED, zorder=3)


def connect(ax, a, b, n=5, alpha=0.30):
    """Lines from the right face of slab a to the left face of slab b."""
    x1, y1, h1, w1 = a
    x2, y2, h2, w2 = b
    for i in range(n):
        ya = y1 - h1 * 0.4 + h1 * 0.8 * i / max(n - 1, 1)
        yb = y2 - h2 * 0.4 + h2 * 0.8 * i / max(n - 1, 1)
        ax.plot([x1 + w1 / 2, x2 - w2 / 2], [ya, yb], color="#8fa0bb", lw=0.7,
                alpha=alpha, zorder=0)


def frame(title, note):
    fig, ax = plt.subplots(figsize=(13.4, 6.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")
    ax.text(50, 59, title, ha="center", fontsize=13, color=INK, fontweight="bold")
    ax.text(50, 2.0, note, ha="center", fontsize=9.2, color=MUTED)
    return fig, ax


def teacher() -> None:
    fig, ax = frame("教师网络：本体感受 + 特权输入，输出力速率",
                    "卷积做延迟自适应（感受野 27 步 = 540 ms）；特权块只抬高上限，不部署")
    y = 32
    layers = [
        (9, 8.0, 12, "in", "输入", "176 维"),
        (25, 7.0, 10, "net", "卷积 ×3", "5→24→24→24"),
        (41, 8.0, 12, "net", "拼接", "76 维"),
        (60, 11.0, 18, "net", "主干", "512 × 8"),
        (78, 8.0, 12, "head", "均值头", "512 → 1"),
        (92, 5.0, 7, "head", "ΔF", "力速率"),
    ]
    boxes = []
    for cx, w, h, kind, lab, sub in layers:
        slab(ax, cx, y, w, h, kind, lab, sub)
        boxes.append((cx, y, h, w))
    for a, b in zip(boxes, boxes[1:]):
        connect(ax, a, b)
    # privileged branch
    slab(ax, 25, 14, 9, 6, "priv", "特权 12", "θ 8 + 状态 4")
    ax.add_patch(FancyArrowPatch((25, 17.2), (37.6, 25.0), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#7b6ca8",
                                 connectionstyle="arc3,rad=-0.15"))
    # output arrow
    ax.add_patch(FancyArrowPatch((94.6, y), (99.0, y), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, color=EDGE))
    fig.tight_layout()
    path = OUT / "teacher_network_3d.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path)


def student() -> None:
    fig, ax = frame("学生网络：只读本体感受与历史，不读 θ",
                    "同一份 176 维输入，卷积前端窄一半、主干 32 宽一层——总计 3,856 参数")
    y = 34
    layers = [
        (12, 8.0, 12, "in", "输入", "176 维"),
        (30, 7.0, 10, "net", "卷积 ×3", "5→8→8→8"),
        (48, 8.0, 12, "net", "拼接", "32 维"),
        (67, 8.0, 12, "net", "主干", "32 × 1"),
        (84, 7.0, 10, "head", "输出", "32 → 1"),
    ]
    boxes = []
    for cx, w, h, kind, lab, sub in layers:
        slab(ax, cx, y, w, h, kind, lab, sub)
        boxes.append((cx, y, h, w))
    for a, b in zip(boxes, boxes[1:]):
        connect(ax, a, b)
    ax.text(50, 20, "教师 4.26M → 学生 3,856，压缩 1,100 倍；同分布误差是教师的 1.28 倍",
            ha="center", fontsize=10.2, color=INK)
    ax.add_patch(FancyArrowPatch((87.6, y), (93.0, y), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.2, color=EDGE))
    fig.tight_layout()
    path = OUT / "student_network_3d.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    teacher()
    student()
