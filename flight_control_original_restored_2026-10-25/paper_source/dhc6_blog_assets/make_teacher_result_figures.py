"""Figures for the teacher / student / transfer sections of the report page.

    python dhc6_blog_assets/make_teacher_result_figures.py

Reads _report_review/results_summary.json, writes
teacher_bank_metrics.png / student_dagger.png / dhc6_transfer.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

WS = Path(__file__).resolve().parents[1]
SUMMARY = json.loads((WS / "_report_review/results_summary.json").read_text(encoding="utf-8"))
OUT = WS / "dhc6_blog_assets"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#16233c"
MUTED = "#16233a"
EDGE = "#5b6c85"
GRID = "#e3e9f3"
OK = "#2f6fd0"
BAD = "#d8382f"
OK2 = "#e08a1e"


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(EDGE)
    ax.tick_params(labelsize=8, colors=MUTED)


def teacher_bank() -> None:
    gate = SUMMARY["gate"]
    plants = SUMMARY["teacher"]["plants"]
    ok = [p for p in plants if p["pass"]]
    bad = [p for p in plants if not p["pass"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.scatter([p["rmse"] for p in ok], [p["peak"] for p in ok], s=42, color=OK,
               alpha=0.85, label=f"通过 {len(ok)}", linewidths=0)
    ax.scatter([p["rmse"] for p in bad], [p["peak"] for p in bad], s=60, color=BAD,
               marker="X", label=f"未通过 {len(bad)}", linewidths=0)
    ax.axvline(gate["rmse"], color=MUTED, ls=(0, (5, 4)), lw=1.0)
    ax.axhline(gate["peak"], color=MUTED, ls=(0, (5, 4)), lw=1.0)
    ax.set_xlabel("跟踪 RMSE (deg/s)", fontsize=9.2, color=INK)
    ax.set_ylabel("峰值误差 (deg/s)", fontsize=9.2, color=INK)
    ax.set_title("46 架通过、3 架未通过的教师分布（虚线为门限）", fontsize=9.8,
                 color=INK, fontweight="bold")
    ax.set_xlim(-1, 16)
    ax.set_ylim(-2, 36)
    ax.legend(fontsize=7.8, frameon=False, loc="center right")
    style(ax)
    for p in bad:
        ax.annotate(p["plant_id"].split("-")[-1], (p["rmse"], p["peak"]),
                    textcoords="offset points", xytext=(-30, 8), fontsize=7.2, color=BAD)
    fig.tight_layout()
    path = OUT / "teacher_bank_metrics.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def student_curves() -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for tag, color, name in (("tiny", OK, "3,856 参数"), ("small", OK2, "75,040 参数")):
        rounds = SUMMARY["students"][tag]["rounds"]
        xs = [r["round"] for r in rounds]
        ys = [r["median_ratio"] for r in rounds]
        ax.plot(xs, ys, marker="o", color=color, lw=2.0, ms=5, label=name)
    ax.axhline(1.0, color=MUTED, ls=(0, (4, 4)), lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("DAgger 轮次", fontsize=9.2, color=INK)
    ax.set_ylabel("学生误差 / 教师误差（中位）", fontsize=9.2, color=INK)
    ax.set_title("蒸馏收敛：一轮 DAgger 就把差距从数十倍压到 2 倍以内", fontsize=9.8,
                 color=INK, fontweight="bold")
    ax.legend(fontsize=7.8, frameon=False)
    ax.set_xticks(range(len(SUMMARY["students"]["tiny"]["rounds"])))
    style(ax)
    fig.tight_layout()
    path = OUT / "student_dagger.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def transfer() -> None:
    t = SUMMARY["transfer"]
    names = t["adapt"] + t["held"]
    before = [t["before"][n] for n in names]
    after = [t["after"][n] for n in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    ax.bar(x - 0.2, before, width=0.38, color="#b9c4d6", label="微调前")
    ax.bar(x + 0.2, after, width=0.38, color=OK, label="微调后")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7.0)
    ax.set_ylabel("跟踪 RMSE (deg/s)", fontsize=9.2, color=INK)
    ax.set_title("DHC-6 迁移：8 个适配状态大幅改善，3 个 V050 留出状态几乎未动",
                 fontsize=9.8, color=INK, fontweight="bold")
    ax.axvline(len(t["adapt"]) - 0.5, color=MUTED, ls=(0, (4, 4)), lw=1.0)
    ax.text(len(t["adapt"]) - 0.4, max(before) * 0.92, "留出 →", fontsize=9, color=MUTED)
    ax.legend(fontsize=7.8, frameon=False)
    style(ax)
    fig.tight_layout()
    path = OUT / "dhc6_transfer.png"
    fig.savefig(path.with_suffix(".svg"), facecolor="white")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    teacher_bank()
    student_curves()
    transfer()
