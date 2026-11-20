"""Classic 3D-looking neural network diagrams (nodes + connection mesh).

    python dhc6_blog_assets/make_network_mesh.py

Each layer is an oblique-projected grid of neurons; consecutive layers are joined
by a sampled connection mesh.  Schematic, not to scale.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
DX, DY = 0.30, 0.20          # oblique projection of one depth step
SP = 1.15                      # neuron spacing

COL = {
    "in":   ("#3b74c4", "#b9cdea"),
    "net":  ("#2f6fd0", "#c8daf4"),
    "priv": ("#7b5cb8", "#d8cdf0"),
    "head": ("#d07a1e", "#f3d9b4"),
}


def layer(ax, x, rows, cols, kind, rng, size=26, alpha=0.95):
    """Draw a rows×cols neuron grid on an oblique plane; return projected points."""
    dark, light = COL[kind]
    pts = []
    for r in range(rows):
        for c in range(cols):
            px = x + DX * r
            py = (c - (cols - 1) / 2) * SP + DY * r
            pts.append((px, py))
    pts = np.array(pts)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, color=dark, alpha=alpha,
               edgecolors="white", linewidths=0.5, zorder=3)
    # faint plane outline
    corners = np.array([[x, -(cols - 1) / 2 * SP],
                        [x, (cols - 1) / 2 * SP],
                        [x + DX * (rows - 1), (cols - 1) / 2 * SP + DY * (rows - 1)],
                        [x + DX * (rows - 1), -(cols - 1) / 2 * SP + DY * (rows - 1)]])
    ax.plot(*np.vstack([corners, corners[:1]]).T, color=light, lw=1.0, zorder=1)
    return pts


def mesh(ax, a, b, rng, per=2, alpha=0.10):
    """Connect each neuron of a to a few neurons of b."""
    for p in a:
        idx = rng.choice(len(b), size=min(per, len(b)), replace=False)
        for j in idx:
            ax.plot([p[0], b[j][0]], [p[1], b[j][1]], color="#6d7f9c",
                    lw=0.45, alpha=alpha, zorder=0)


LABEL_Y = -6.4


def label(ax, pts, text, sub=""):
    x = pts[:, 0].mean()
    y = LABEL_Y
    ax.text(x, y, text, ha="center", va="top", fontsize=10.0, color=INK,
            fontweight="bold")
    if sub:
        ax.text(x, y - 1.5, sub, ha="center", va="top", fontsize=8.2, color=MUTED)


def shrink(path: Path, colors: int = 96) -> None:
    """Palette-quantise the line art; keeps it crisp, cuts the file by ~5x."""
    img = Image.open(path).convert("RGB")
    img = img.quantize(colors=colors, method=Image.MEDIANCUT)
    img.save(path, optimize=True)


def teacher() -> None:
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(14.5, 8.4))
    ax.axis("off")
    layers = [
        (0.0, 8, 6, "in", "输入", "176 维"),
        (3.2, 6, 5, "net", "卷积 ×3", "5→24→24→24"),
        (6.6, 8, 6, "net", "拼接", "76 维"),
        (10.4, 10, 8, "net", "主干 512 × 8", "8 个残差块"),
        (14.2, 7, 5, "head", "均值头", "512 → 1"),
    ]
    pts = []
    for x, r, c, kind, lab, sub in layers:
        pts.append(layer(ax, x, r, c, kind, rng, size=30 if kind != "net" else 24))
    for a, b in zip(pts, pts[1:]):
        mesh(ax, a, b, rng, per=2, alpha=0.09)
    for (x, r, c, kind, lab, sub), p in zip(layers, pts):
        label(ax, p, lab, sub)
    # privileged branch
    priv = layer(ax, 3.4, 4, 3, "priv", rng, size=30)
    for p in priv:
        for j in rng.choice(len(pts[2]), size=2, replace=False):
            ax.plot([p[0], pts[2][j][0]], [p[1], pts[2][j][1]], color="#7b5cb8",
                    lw=0.6, alpha=0.22, zorder=0)
    ax.text(4.1, -9.6, "特权 12（θ 8 + 状态 4）", ha="center", va="top",
            fontsize=9.4, color="#6b4fa8", fontweight="bold")
    ax.text(17.4, 0, "ΔF", ha="center", va="center", fontsize=11, color=INK,
            fontweight="bold")
    ax.annotate("", xy=(17.0, 0), xytext=(16.0, 0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    ax.set_xlim(-1.2, 18.2)
    ax.set_ylim(-10.8, 6.6)
    ax.set_title("教师网络：本体感受 + 特权输入，输出力速率", fontsize=13.5,
                 color=INK, fontweight="bold", pad=16)
    fig.tight_layout()
    path = OUT / "teacher_network_mesh.png"
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


def student() -> None:
    rng = np.random.default_rng(11)
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.axis("off")
    layers = [
        (0.0, 8, 6, "in", "输入", "176 维"),
        (3.4, 5, 4, "net", "卷积 ×3", "5→8→8→8"),
        (6.8, 6, 4, "net", "拼接", "32 维"),
        (10.4, 7, 5, "net", "主干 32 × 1", "1 个残差块"),
        (14.0, 4, 3, "head", "输出", "32 → 1"),
    ]
    pts = []
    for x, r, c, kind, lab, sub in layers:
        pts.append(layer(ax, x, r, c, kind, rng, size=26 if kind == "net" else 34))
    for a, b in zip(pts, pts[1:]):
        mesh(ax, a, b, rng, per=2, alpha=0.10)
    for (x, r, c, kind, lab, sub), p in zip(layers, pts):
        label(ax, p, lab, sub)
    ax.text(7.8, -10.6, "教师 4.26M → 学生 3,856，压缩 1,100 倍；同分布误差 1.28×",
            ha="center", va="center", fontsize=10.4, color=INK)
    ax.annotate("", xy=(15.6, 0), xytext=(14.6, 0),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
    ax.text(16.6, 0, "ΔF", ha="left", va="center", fontsize=11, color=INK,
            fontweight="bold")
    ax.set_xlim(-1.2, 17.0)
    ax.set_ylim(-11.8, 6.4)
    ax.set_title("学生网络：只读本体感受与历史，不读 θ", fontsize=13.5,
                 color=INK, fontweight="bold", pad=16)
    fig.tight_layout()
    path = OUT / "student_network_mesh.png"
    fig.savefig(path, dpi=125, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


if __name__ == "__main__":
    teacher()
    student()
