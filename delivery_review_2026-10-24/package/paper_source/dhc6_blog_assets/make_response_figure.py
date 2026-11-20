"""Closed-loop response traces: teacher vs student, and DHC-6 before/after.

    python dhc6_blog_assets/make_response_figure.py

Reads _report_review/remote/traces.json (produced on the GPU host by trace_all.py)
and writes response_traces.png.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

WS = Path(__file__).resolve().parents[1]
TRACES = json.loads((WS / "_report_review/remote/traces.json").read_text(encoding="utf-8"))
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
GRID = "#e4eaf3"
REF = "#5b6c85"
TEACH = "#2f6fd0"
STUD = "#e08a1e"
BEFORE = "#b9c4d6"
AFTER = "#2f8f5b"

DEG = 180.0 / math.pi


def style(ax, title):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#5b6c85")
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title(title, fontsize=10.2, color=INK, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("时间 (s)", fontsize=8.6, color=MUTED)
    ax.set_ylabel("滚转角速度 (deg/s)", fontsize=8.6, color=MUTED)


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 4.8))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]

    # (a)(b) P-channel: reference, teacher, student
    for ax, pid, title in ((ax1, "train_core-0421", "(a) 2 级对象 · 0421"),
                           (ax2, "train_core-0701", "(b) 3 级对象 · 0701")):
        d = TRACES["p_channel"][pid]
        t = d["t"]
        ref = d["ref"]
        pt = d["p_teacher"]
        ps = d["p_student"]
        ax.plot(t, ref, "--", color=REF, lw=1.3, label="指令参考")
        ax.plot(t, pt, color=TEACH, lw=1.6, label="教师")
        ax.plot(t, ps, color=STUD, lw=1.4, label="学生（3,856 参数）")
        ax.set_xlim(0, 8)
        style(ax, title)
    ax1.legend(fontsize=7.8, frameon=False, loc="lower right")
    ax1.annotate(f"教师 RMSE 0.18 · 学生 0.66 deg/s", xy=(0.03, 0.93), xycoords="axes fraction", fontsize=7.6, color=MUTED,
                 va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax2.annotate("教师 RMSE 0.25 · 学生 0.91 deg/s", xy=(0.03, 0.93), xycoords="axes fraction", fontsize=7.6, color=MUTED,
                 va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    # (c)(d) DHC-6: before vs after fine-tuning
    for ax, state, title in ((ax3, "V070_H3000", "(c) DHC-6 · V070_H3000（适配）"),
                             (ax4, "V050_H2500", "(d) DHC-6 · V050_H2500（留出）")):
        d = TRACES["dhc6"][state]
        t = d["finetuned"]["t"]
        ref = d["finetuned"]["ref"]
        before = d["before"]["p"]
        after = d["finetuned"]["p"]
        ax.plot(t, ref, "--", color=REF, lw=1.3, label="指令参考")
        ax.plot(t, before, color=BEFORE, lw=1.4, label="微调前")
        ax.plot(t, after, color=AFTER, lw=1.6, label="微调后")
        ax.set_xlim(0, 8)
        style(ax, title)
    ax3.legend(fontsize=7.8, frameon=False, loc="lower right")
    ax3.annotate("2.91 → 0.20 deg/s", xy=(0.03, 0.93), xycoords="axes fraction",
                 fontsize=7.6, color=MUTED, va="top",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax4.annotate("2.51 → 2.10 deg/s", xy=(0.03, 0.93), xycoords="axes fraction",
                 fontsize=7.6, color=MUTED, va="top",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))

    fig.tight_layout(pad=0.9)
    path = OUT / "response_traces.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    Image.open(path).convert("RGB").quantize(colors=128, method=Image.MEDIANCUT).save(
        path, optimize=True)
    print("wrote", path)


if __name__ == "__main__":
    main()
