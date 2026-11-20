"""Publication-style architectures (English labels) for the teacher / student.

    python dhc6_blog_assets/make_network_modern.py

The concat is drawn explicitly: the temporal convolution produces ``c``, which is
concatenated (‖) with the current signals, the recent actions and — for the
teacher only — the privileged vector π, giving ``h``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from PIL import Image

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#1b2a41"
MUTED = "#16233a"
RULE = "#4a5a72"

BLUE = ("#dce8f7", "#b9cfe9")
GOLD = ("#fbeccb", "#e6c98c")
LILAC = ("#e3daf6", "#c3b2e2")
PEACH = ("#fbe0cd", "#e8b78f")

MOD_W, MOD_H = 1.55, 0.86
W, H, D = 0.56, 0.34, 0.13


def module(ax, x, y, name, spec, pal, w=MOD_W, h=MOD_H):
    face, edge = pal
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.09",
                                facecolor=face, edgecolor=edge, lw=0.9, zorder=3))
    ax.text(x + w / 2, y + h * 0.63, name, ha="center", va="center", fontsize=8.8,
            color=INK, fontweight="bold", zorder=4)
    ax.text(x + w / 2, y + h * 0.27, spec, ha="center", va="center", fontsize=6.8,
            color=MUTED, zorder=4)
    return x + w


def tensor(ax, x, y, pal=GOLD):
    face, edge = pal
    ax.add_patch(Rectangle((x, y), W, H, facecolor=face, edgecolor=edge, lw=0.8, zorder=3))
    ax.add_patch(Polygon([(x, y + H), (x + W, y + H), (x + W + D, y + H + D * 0.85),
                          (x + D, y + H + D * 0.85)], closed=True, facecolor="white",
                         edgecolor=edge, lw=0.7, zorder=2))
    ax.add_patch(Polygon([(x + W, y), (x + W, y + H), (x + W + D, y + H + D * 0.85),
                          (x + W + D, y + D * 0.85)], closed=True, facecolor=face,
                         edgecolor=edge, lw=0.7, alpha=0.75, zorder=2))
    return x + W + D


def cat(ax, x, y, h=0.42):
    w = 0.34
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.05",
                                facecolor="white", edgecolor=RULE, lw=0.8, zorder=4))
    ax.text(x + w / 2, y, "‖", ha="center", va="center", fontsize=9.0, color=INK,
            fontweight="bold", zorder=5)
    return x + w


def arrow(ax, x0, x1, y, label="", color=RULE, dy=0.10, size=6.3):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7,
                                 lw=0.8, color=color, zorder=6))
    if label:
        ax.text((x0 + x1) / 2, y + dy, label, ha="center", va="bottom",
                fontsize=size, color=MUTED)


def legend(ax, x, y):
    w, h = 2.45, 1.25
    ax.add_patch(Rectangle((x, y), w, h, facecolor="white", edgecolor="#cfd8e6", lw=0.7))
    ax.text(x + 0.13, y + h - 0.22, "legend", fontsize=6.0, color=MUTED, style="italic")
    ax.add_patch(FancyBboxPatch((x + 0.13, y + 0.66), 0.28, 0.18,
                                boxstyle="round,pad=0,rounding_size=0.03",
                                facecolor=BLUE[0], edgecolor=BLUE[1], lw=0.8))
    ax.text(x + 0.50, y + 0.70, "trainable module", fontsize=6.2, color=INK)
    ax.add_patch(Rectangle((x + 0.13, y + 0.34), 0.28, 0.18, facecolor=GOLD[0],
                           edgecolor=GOLD[1], lw=0.8))
    ax.text(x + 0.50, y + 0.38, "feature tensor", fontsize=6.2, color=INK)
    ax.add_patch(FancyArrowPatch((x + 0.13, y + 0.15), (x + 0.41, y + 0.15),
                                 arrowstyle="-|>", mutation_scale=6, lw=0.7, color=RULE))
    ax.text(x + 0.50, y + 0.11, "forward pass", fontsize=6.2, color=INK)


def shrink(path: Path, colors: int = 96) -> None:
    Image.open(path).convert("RGB").quantize(colors=colors, method=Image.MEDIANCUT).save(
        path, optimize=True)


def draw(network: str) -> None:
    fig, ax = plt.subplots(figsize=(11.8, 3.15))
    ax.axis("off")
    base = 1.70

    if network == "teacher":
        title = "Teacher  ·  proprioception + privileged information π"
        conv = ("Conv1D ×3", "5→24→24→24")
        c_dim, h_dim = "48", "76"
        trunk = ("MLP trunk", "512 × 8")
        z_dim = "512"
        head = ("Mean head", "512 → 1")
        cat_note = "48 + 9 + 7 + 12"
    else:
        title = "Student  ·  proprioception + 32-step history, no θ"
        conv = ("Conv1D ×3", "5→8→8→8")
        c_dim, h_dim = "16", "32"
        trunk = ("MLP trunk", "32 × 1")
        z_dim = "32"
        head = ("Mean head", "32 → 1")
        cat_note = "16 + 9 + 7"

    x = 0.25
    r = tensor(ax, x, base + 0.28)
    ax.text(x + W / 2, base + 0.16, "x (176)", ha="center", va="top", fontsize=7.8,
            color=INK)
    arrow(ax, r + 0.07, r + 0.22, base + MOD_H / 2)

    x = r + 0.30
    conv_r = module(ax, x, base, conv[0], conv[1], BLUE)
    arrow(ax, conv_r + 0.06, conv_r + 0.25, base + MOD_H / 2)
    x = conv_r + 0.32
    c_r = tensor(ax, x, base + 0.28)
    ax.text(x + W / 2, base + 0.16, f"c ({c_dim})", ha="center", va="top",
            fontsize=7.8, color=INK)

    x = c_r + 0.30
    cat_x = x
    cat_r = cat(ax, cat_x, base + MOD_H / 2)
    arrow(ax, c_r + 0.05, cat_x - 0.04, base + MOD_H / 2)
    arrow(ax, cat_r + 0.05, cat_r + 0.25, base + MOD_H / 2)

    x = cat_r + 0.32
    h_r = tensor(ax, x, base + 0.28)
    ax.text(x + W / 2, base + 0.16, f"h ({h_dim})", ha="center", va="top",
            fontsize=7.8, color=INK)
    ax.text(x + W / 2, base - 0.10, cat_note, ha="center", va="top", fontsize=6.0,
            color=MUTED)

    arrow(ax, h_r + 0.06, h_r + 0.25, base + MOD_H / 2)
    x = h_r + 0.32
    trunk_r = module(ax, x, base, trunk[0], trunk[1], BLUE)
    arrow(ax, trunk_r + 0.06, trunk_r + 0.25, base + MOD_H / 2)
    x = trunk_r + 0.32
    z_r = tensor(ax, x, base + 0.28)
    ax.text(x + W / 2, base + 0.16, f"z ({z_dim})", ha="center", va="top",
            fontsize=7.8, color=INK)

    arrow(ax, z_r + 0.06, z_r + 0.25, base + MOD_H / 2)
    x = z_r + 0.32
    head_r = module(ax, x, base, head[0], head[1], PEACH)
    ax.add_patch(FancyArrowPatch((head_r + 0.10, base + MOD_H / 2),
                                 (head_r + 0.52, base + MOD_H / 2), arrowstyle="-|>",
                                 mutation_scale=7, lw=0.8, color=RULE, zorder=6))
    ax.text(head_r + 0.31, base + MOD_H / 2 + 0.10, "ΔF", ha="center", va="bottom",
            fontsize=6.8, color=MUTED)

    if network == "teacher":
        py = base - 0.78
        tensor(ax, cat_x - 0.17, py, LILAC)
        ax.text(cat_x + 0.55, py + 0.14, "π (θ, plant state)", ha="left", va="center",
                fontsize=7.6, color=INK)
        ax.add_patch(FancyArrowPatch((cat_x + 0.17, py + 0.34),
                                     (cat_x + 0.17, base - 0.02), arrowstyle="-|>",
                                     mutation_scale=7, lw=0.8, color=LILAC[1], zorder=5))

    legend(ax, head_r + 0.55, base - 0.05)
    x_hi = head_r + 0.55 + 2.45 + 0.22
    y_lo = (base - 1.12) if network == "teacher" else (base - 0.50)
    y_hi = base + MOD_H + 0.04
    ax.set_xlim(-0.08, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    span_x, span_y = x_hi + 0.08, y_hi - y_lo
    width_in = 6.4
    fig.set_size_inches(width_in, width_in * span_y / span_x + 0.30)
    ax.set_title(title, fontsize=10.6, color=INK, loc="left", pad=2, x=0.004)
    fig.subplots_adjust(left=0.004, right=0.996, top=1.0 - 0.26 / (width_in * span_y / span_x + 0.30),
                        bottom=0.01)
    path = OUT / f"{network}_network_modern.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    shrink(path)
    print("wrote", path)


if __name__ == "__main__":
    draw("teacher")
    draw("student")
