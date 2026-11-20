"""Four-step pipeline figure, aligned with sections 3.1-3.4.

    python dhc6_blog_assets/make_pipeline_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets" / "pipeline_framework.png"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["svg.fonttype"] = "path"

INK = "#1b2a41"
MUTED = "#16233a"
EDGE = "#5b6c85"
BLUE = "#eef4fd"
ACC = "#e8f0fb"
WARN = "#fdf1e3"
PRIV = "#f3eefb"
GREEN = "#eaf5ee"


def box(ax, x, y, w, title, lines, fill=BLUE, title_size=9.6, line_size=7.8):
    h = 2.0 + 2.3 * (len(lines) + 1)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.3,rounding_size=0.9",
                                linewidth=1.0, edgecolor=EDGE, facecolor=fill))
    top = y + h / 2 - 2.0
    ax.text(x, top, title, ha="center", va="center", fontsize=title_size, color=INK,
            fontweight="bold")
    for i, line in enumerate(lines):
        ax.text(x, top - 2.3 * (i + 1), line, ha="center", va="center",
                fontsize=line_size, color=MUTED)
    return {"x": x, "y": y, "w": w, "h": h, "l": x - w / 2, "r": x + w / 2}


def arrow(ax, p, q, color=EDGE):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.0,
                                 color=color, shrinkA=0, shrinkB=0))


fig, ax = plt.subplots(figsize=(10.4, 5.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

rows = [
    ("① 采样", 84, [
        ("参数空间", ["8 参数 θ", "GJB 判据"], BLUE),
        ("分层采样", ["1/2/3 级配额", "＋分布外"], BLUE),
        ("对象库", ["5,000 架", "4,500 + 500"], ACC),
        ("训练对象", ["49 架进入教师", "每级各 16–17"], GREEN),
    ]),
    ("② 教师", 62, [
        ("单架对象", ["θ_i 与品模", "1 kHz / 50 Hz"], BLUE),
        ("PPO 教师", ["176 + 特权 12", "力速率动作"], PRIV),
        ("验收门限", ["RMSE / 峰值 / TV", "46 / 49 通过"], WARN),
        ("教师库", ["49 架", "4.26 M 参数/架"], ACC),
    ]),
    ("③ 学生", 40, [
        ("教师轨迹", ["状态与动作", "同分布采集"], BLUE),
        ("DAgger 6 轮", ["教师驱动→学生", "教师持续标注"], BLUE),
        ("回归训练", ["无 θ、无状态", "32 步历史"], WARN),
        ("统一学生", ["3,856 参数", "同分布 1.28×"], ACC),
    ]),
    ("④ 迁移", 18, [
        ("DHC-6 数据", ["12 个配平状态", "72 条 50 Hz"], BLUE),
        ("ARX 辨识", ["na 4–5、nk 1", "VAF 98.0–98.7%"], BLUE),
        ("小样本微调", ["75,040 参数学生", "40 步、lr 3e-5"], WARN),
        ("迁移结果", ["适配 2.91→0.60", "留出 2.66→2.15"], ACC),
    ]),
]

xs = [14, 38, 62, 86]
for label, y, items in rows:
    ax.text(2.0, y + 6.8, label, fontsize=10.4, color=INK, fontweight="bold",
            ha="left", va="bottom")
    boxes = []
    for x, (title, lines, fill) in zip(xs, items):
        boxes.append(box(ax, x, y - 2.0, 21, title, lines, fill))
    for a, b in zip(boxes, boxes[1:]):
        arrow(ax, (a["r"] + 0.6, a["y"]), (b["l"] - 0.6, b["y"]))

for (_, y1, _), (_, y2, _) in zip(rows, rows[1:]):
    ax.add_patch(FancyArrowPatch((86, y1 - 10.0), (86, y2 + 10.0), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.1, color="#5c6678"))

ax.text(50, 2.6, "同一接口贯穿四步：策略只输出力速率；积分、限幅、作动器与品模/ARX 对象属于外部回路",
        ha="center", fontsize=8.0, color=MUTED)
ax.set_title("四步链路：采样 → 教师 → 学生 → 迁移", fontsize=12.4, color=INK,
             fontweight="bold", pad=10)

fig.tight_layout()
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
fig.savefig(OUT, dpi=150, facecolor="white")
print("wrote", OUT.with_suffix(".svg"))
