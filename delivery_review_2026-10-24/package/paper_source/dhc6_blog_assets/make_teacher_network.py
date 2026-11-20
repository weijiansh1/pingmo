"""Teacher network diagram (C variant: Actor also reads privileged theta/state).

    python dhc6_blog_assets/make_teacher_network.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

WS = Path(__file__).resolve().parents[1]
OUT = WS / "dhc6_blog_assets" / "teacher_network.png"

for candidate in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

INK = "#16233c"
MUTED = "#16233a"
EDGE = "#5b6c85"
FILL = "#f2f6fc"
ACCENT = "#e8f0fb"
NET = "#eef4fd"
WARN = "#fdf1e3"
PRIV = "#f3eefb"

fig, ax = plt.subplots(figsize=(13.6, 9.4))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

LINE = 2.55
PAD = 3.2


def box(x, y, w, title, lines, fill=FILL, size=10.5, line_size=8.8):
    height = PAD + LINE * (len(lines) + 1)
    ax.add_patch(FancyBboxPatch((x - w / 2, y - height / 2), w, height,
                                boxstyle="round,pad=0.3,rounding_size=1.0",
                                linewidth=1.1, edgecolor=EDGE, facecolor=fill))
    top = y + height / 2 - PAD / 2 - LINE * 0.55
    ax.text(x, top, title, ha="center", va="center", fontsize=size,
            color=INK, fontweight="bold")
    for index, line in enumerate(lines):
        ax.text(x, top - LINE * (index + 1), line, ha="center", va="center",
                fontsize=line_size, color=MUTED)
    return {"left": x - w / 2, "right": x + w / 2,
            "top": y + height / 2, "bottom": y - height / 2,
            "cx": x, "cy": y, "h": height}


def arrow(start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.15, color=EDGE, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------- actor
ax.text(33, 98.5, "Actor：本体感受 + 特权（教师不部署，可以看 θ 和对象状态）",
        ha="center", fontsize=12.5, color=INK, fontweight="bold")

h_hist = box(9, 90, 15, "历史 32×5", ["160 维"])
h_cur = box(26, 90, 18, "当前 9 维", ["指令 参考 实测", "误差 ∫e 加速度"])
h_old = box(42, 90, 13, "旧动作 7 步", ["力速率"])
h_priv = box(55, 90, 14, "特权 12 维", ["θ 8", "对象状态 4"], fill=PRIV)

c_conv = box(9, 76, 16, "卷积前端 3 层",
             ["5→24 k3 d1", "24→24 k3 d4", "24→24 k3 d8", "感受野 27 步"], fill=NET)
c_cat = box(33, 61, 40, "拼接 76 维",
            ["卷积 48 + 当前 9 + 旧动作 7 + 特权 12"], fill=ACCENT)
c_trunk = box(33, 49, 40, "Linear(76→512) + 8 × 残差块",
              ["pre-LN · SiLU · 残差 0.1 · 4.25M"], fill=NET)
c_norm = box(33, 39, 18, "LayerNorm(512)", [])
c_head = box(33, 30, 24, "Linear(512→1) → tanh", ["均值 μ ∈ (−1, 1)"])
c_samp = box(33, 21, 22, "高斯采样 a ~ N(μ, σ)", [])
c_out = box(33, 11, 26, "ΔF 力速率", ["±1 → ±1.76 N/步，再限幅 ±22 N"], fill=WARN)
c_std = box(55, 21, 14, "log_std 标量", ["初值 −2.0"])

arrow((h_hist["cx"], h_hist["bottom"]), (c_conv["cx"], c_conv["top"]))
arrow((h_cur["cx"], h_cur["bottom"]), (c_cat["cx"] - 4, c_cat["top"]))
arrow((h_old["cx"], h_old["bottom"]), (c_cat["cx"] + 4, c_cat["top"]))
arrow((h_priv["cx"], h_priv["bottom"]), (c_cat["right"] - 2, c_cat["top"]))
arrow((c_conv["right"] - 1, c_conv["bottom"]), (c_cat["left"] + 3, c_cat["top"]))
arrow((c_cat["cx"], c_cat["bottom"]), (c_trunk["cx"], c_trunk["top"]))
arrow((c_trunk["cx"], c_trunk["bottom"]), (c_norm["cx"], c_norm["top"]))
arrow((c_norm["cx"], c_norm["bottom"]), (c_head["cx"], c_head["top"]))
arrow((c_head["cx"], c_head["bottom"]), (c_samp["cx"], c_samp["top"]))
arrow((c_samp["cx"], c_samp["bottom"]), (c_out["cx"], c_out["top"]))
arrow((c_std["left"], c_std["cy"]), (c_samp["right"], c_samp["cy"]))

# ------------------------------------------------------------------ critic
ax.text(82, 98.5, "Critic：训练时看全部特权，部署不用", ha="center", fontsize=12.5,
        color=INK, fontweight="bold")
v_obs = box(72, 90, 17, "观测 176", ["与 Actor 同一份"])
v_priv = box(92, 90, 15, "特权 207", ["θ 8 + 状态 4", "FIFO 193 + 进度 1 + 施力 1"], fill=PRIV)
v_cat = box(82, 75, 33, "拼接 383", ["观测 176 + 特权 207"], fill=ACCENT)
v_trunk = box(82, 63, 33, "Linear(→512) + 4 × 残差块", ["pre-LN · SiLU · 2.30M"], fill=NET)
v_head = box(82, 52, 18, "Linear(512→1)", [])
v_out = box(82, 43, 16, "V(s)", ["标量价值"])

arrow((v_obs["cx"] + 5, v_obs["bottom"]), (v_cat["cx"] - 8, v_cat["top"]))
arrow((v_priv["cx"] - 5, v_priv["bottom"]), (v_cat["cx"] + 8, v_cat["top"]))
arrow((v_cat["cx"], v_cat["bottom"]), (v_trunk["cx"], v_trunk["top"]))
arrow((v_trunk["cx"], v_trunk["bottom"]), (v_head["cx"], v_head["top"]))
arrow((v_head["cx"], v_head["bottom"]), (v_out["cx"], v_out["top"]))

ax.text(82, 33, "特权 = θ 8 + 对象状态 4 + 延迟 FIFO（4–503）", ha="center", fontsize=8.6, color=MUTED)
ax.text(82, 29, "+ episode 进度 1 + 施加力 1", ha="center", fontsize=8.6, color=MUTED)
ax.text(82, 24, "FIFO 宽度随 τ_p 变化，Critic 按对象单独构建", ha="center", fontsize=8.6,
        color=MUTED)

ax.plot([67, 67], [4, 96], color="#dfe5ee", lw=1.0, ls=(0, (4, 4)))
ax.text(33, 3.2, "参数量：Actor 4.26M（卷积 3.9k + 主干 4.25M）· Critic 2.30M",
        ha="center", fontsize=9.2, color=MUTED)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor="white")
print("wrote", OUT)

