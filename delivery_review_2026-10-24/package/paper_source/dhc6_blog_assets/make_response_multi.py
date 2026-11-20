"""Multi-command response figures (compact, wide panels) for the paper.

    python dhc6_blog_assets/make_response_multi.py

Produces
  response_commands.svg  : 2x4 grid, teacher vs student on the P-channel
  dhc6_states.svg        : 1x4 grid, DHC-6 before/after across four states
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

WS = Path(__file__).resolve().parents[1]
TR = json.loads((WS / "_report_review/remote/traces_multi.json").read_text(encoding="utf-8"))
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
GRID = "#e6ebf3"
REF = "#5b6c85"
TEACH = "#2f6fd0"
STUD = "#e08a1e"
BEFORE = "#b9c4d6"
AFTER = "#2f8f5b"


def classify(d: dict) -> tuple[str, float, int]:
    ref = d["ref"]
    peak = max(ref, key=abs)
    cross = sum(1 for a, b in zip(ref, ref[1:]) if (a > 1e-6) != (b > 1e-6))
    kind = "sine" if cross >= 6 else ("doublet" if cross >= 2 else "step")
    return kind, peak, cross


def label_of(kind: str, peak: float) -> str:
    if kind == "sine":
        return f"多正弦 {abs(peak):.0f}°"
    if kind == "doublet":
        return f"双脉冲 {abs(peak):.0f}°"
    return f"阶跃 {'+' if peak > 0 else '−'}{abs(peak):.0f}°"


def pick_commands(commands: dict) -> list[tuple[str, dict]]:
    """Six steps spanning both signs and all three amplitudes, one doublet, one sine."""
    tagged = [(k, v) + classify(v) for k, v in commands.items()]  # (k, v, kind, peak, cross)
    steps = [t for t in tagged if t[2] == "step"]
    picks: list = []
    seen: set[tuple[int, int]] = set()
    for k, v, kind, peak, _cross in sorted(steps, key=lambda t: (abs(t[3]), t[3] > 0)):
        key = (1 if peak > 0 else -1, round(abs(peak)))
        if key in seen:
            continue
        seen.add(key)
        picks.append((k, v, kind, peak, _cross))
    for kind_wanted in ("doublet", "sine"):
        group = [t for t in tagged if t[2] == kind_wanted]
        if group:
            picks.append(max(group, key=lambda t: t[4]))
    picks.sort(key=lambda t: (t[2] != "step", t[3])) 
    return [(t[0], t[1]) for t in picks[:8]]


def fig_commands() -> None:
    cmds = TR["commands"]
    picked = pick_commands(cmds)
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 3.7), sharex=True)
    for ax, (idx, d) in zip(axes.ravel(), picked):
        t = d["t"]
        ax.plot(t, d["ref"], "--", color=REF, lw=1.0, label="参考")
        ax.plot(t, d["teacher"], color=TEACH, lw=1.2, label="教师")
        ax.plot(t, d["student"], color=STUD, lw=1.1, label="学生")
        kind, peak, _ = classify(d)
        ax.set_title(label_of(kind, peak), fontsize=8.2, color=INK, fontweight="bold", pad=3)
        ax.set_xlim(0, 8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#5b6c85")
        ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
    for ax in axes[1]:
        ax.set_xlabel("时间 (s)", fontsize=7.4, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("p (deg/s)", fontsize=7.4, color=MUTED)
    axes[0, 0].legend(fontsize=6.4, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=0.6)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"response_commands.{ext}", facecolor="white",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote response_commands.svg")


def fig_dhc6() -> None:
    states = list(TR["dhc6"])
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.15), sharex=True)
    for ax, state in zip(axes, states):
        d = TR["dhc6"][state]
        ax.plot(d["t"], d["ref"], "--", color=REF, lw=1.0, label="参考")
        ax.plot(d["t"], d["before"], color=BEFORE, lw=1.1, label="微调前")
        ax.plot(d["t"], d["after"], color=AFTER, lw=1.2, label="微调后")
        tag = "适配" if not state.startswith("V050") else "留出"
        ax.set_title(f"{state}（{tag}）", fontsize=7.6, color=INK, fontweight="bold", pad=3)
        ax.set_xlim(0, 8)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#5b6c85")
        ax.tick_params(labelsize=6.8, colors=MUTED, length=2)
        ax.set_xlabel("时间 (s)", fontsize=7.2, color=MUTED)
    axes[0].set_ylabel("p (deg/s)", fontsize=7.2, color=MUTED)
    axes[0].legend(fontsize=6.2, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4, w_pad=0.5)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"dhc6_states.{ext}", facecolor="white",
                    dpi=150 if ext == "png" else None)
    plt.close(fig)
    print("wrote dhc6_states.svg")


if __name__ == "__main__":
    fig_commands()
    fig_dhc6()
