"""Explain the existing simulation loop without introducing measured data."""
from pathlib import Path
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]
TEXT = """源域闭环仿真与时域响应的生成 指令与实验目标 按各组协议给定 品质相关参数
采样后固定对象 学生策略 测试时权重固定 观测与历史输入 力接口与执行机构
积分 限幅 速率约束 执行机构滞后 参数化滚转对象 数值推进对象状态
观测与历史更新 反馈到下一控制周期 记录响应序列 得到时域响应
控制周期 对象数值步长 动作与状态随时间变化 ；、"""


def fonts():
    """Give existing subsets distinct fallback names; preserve source fonts."""
    needed = {ord(c) for c in TEXT if ord(c) > 127 and not c.isspace()}
    candidates = list((ROOT / "webpage/vendor/fonts").glob("*.ttf"))
    output = ROOT / "paper_source/closed_loop_fonts"
    output.mkdir(exist_ok=True)
    names, records = [], []
    while needed:
        path = max(candidates, key=lambda p: len(needed.intersection(TTFont(p).getBestCmap())))
        font = TTFont(path)
        covered = needed.intersection(font.getBestCmap())
        if not covered:
            raise ValueError("Missing figure glyphs: " + "".join(chr(i) for i in sorted(needed)))
        name = f"LoopCJK{len(names)}"
        for record in font["name"].names:
            if record.nameID in (1, 3, 4, 6, 16):
                record.string = name.encode(record.getEncoding())
        target = output / (name + ".ttf")
        font.save(target)
        font_manager.fontManager.addfont(target)
        names.append(name)
        records.append({"source": str(path.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "fallback_family": name})
        needed -= covered
        candidates.remove(path)
    return names + ["DejaVu Sans"], records


def main():
    families, font_sources = fonts()
    ink, blue, muted, edge = "#18324f", "#17649b", "#45596d", "#91adc4"
    with plt.rc_context({"font.family": families, "font.size": 10.5,
                         "svg.fonttype": "path", "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(8.7, 5.0))
        fig.subplots_adjust(left=.015, right=.985, bottom=.02, top=.98)
        ax.set(xlim=(0, 100), ylim=(0, 80)); ax.axis("off")

        def text(x, y, content, size=10.5, color=ink, weight="normal", **kwargs):
            if '$' in content and any('\u4e00' <= char <= '\u9fff' for char in content):
                children=[]
                for part in re.split(r'(\$[^$]+\$)',content):
                    if not part:
                        continue
                    is_math=part.startswith('$')
                    children.append(TextArea(part,textprops={'fontsize':size,'color':color,
                        'fontweight':weight,'fontfamily':['DejaVu Sans'] if is_math else families,
                        'parse_math':is_math}))
                packed=HPacker(children=children,align='center',pad=0,sep=1)
                annotation=AnnotationBbox(packed,(x,y),frameon=False,box_alignment=(.5,.5),pad=0)
                ax.add_artist(annotation)
                return annotation
            return ax.text(x, y, content, fontsize=size, color=color, fontweight=weight,
                           ha="center", va="center",fontfamily=['DejaVu Sans'] if '$' in content else families, **kwargs)

        def box(x, y, width, height, title, body, title_size=11):
            ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
                boxstyle="round,pad=.35,rounding_size=1.1", facecolor="#f4f8fc",
                edgecolor=edge, linewidth=1))
            text(x, y+height*.20, title, title_size, weight="bold")
            text(x, y-height*.18, body, 9.5, color=muted, linespacing=1.6)

        def arrow(start, end, color=blue, width=1.3):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                        linewidth=width, color=color))

        text(50, 76.5, "源域闭环仿真与时域响应的生成", 15, weight="bold")
        box(15, 63, 27, 13, "指令与实验目标", "按各组协议给定")
        box(78, 63, 28, 13, r"品质相关参数 $\theta$", "采样后固定对象")
        box(15, 39, 27, 21, r"学生策略 $\pi_{\mathrm{S}}$", "观测与历史输入\n测试时权重固定")
        box(46.5, 39, 27, 21, "力接口与执行机构", "积分、限幅、速率约束\n执行机构滞后", 10.8)
        box(78, 39, 28, 21, r"参数化滚转对象 $G_\theta$", "数值推进对象状态", 10.6)
        arrow((15, 56), (15, 50))
        arrow((78, 56), (78, 50))
        arrow((29, 39), (32.4, 39)); text(30.7, 44, r"$a_t$", 11)
        arrow((60.4, 39), (63.2, 39)); text(61.8, 44, r"$F_t$", 11)
        arrow((92.6, 39), (98, 39)); text(96, 44, r"$p_t$", 11)
        ax.plot([98, 98, 15], [39, 20, 20], color=blue, linewidth=1.3)
        arrow((15, 20), (15, 28))
        ax.add_patch(FancyBboxPatch((28, 15.7), 40, 8.6,
            boxstyle="round,pad=.25,rounding_size=.7", facecolor="white", edgecolor="none"))
        text(48, 21.8, "观测与历史更新", 10.5)
        text(48, 17.5, "反馈到下一控制周期", 9.5, color=muted)
        ax.add_patch(Circle((98, 20), .55, facecolor=blue, edgecolor="none"))
        ax.plot([98, 98], [20, 7.5], color=blue, linewidth=1.3)
        arrow((98, 7.5), (94.4, 7.5))
        box(78, 7.5, 31, 12, r"记录响应序列 $p_0,p_1,p_2,\ldots$",
            r"得到时域响应 $p(t)$", 9.7)
        text(28, 8.5, "控制周期 20 ms；对象数值步长 1 ms", 9.2, color=muted)
        text(28, 3.5, "动作与状态随时间变化", 9.2, color=muted)
        output = ROOT / "webpage/paper"
        for extension in ["svg", "pdf", "png"]:
            fig.savefig(output / ("closed_loop_simulation."+extension), dpi=240, facecolor="white")
        plt.close(fig)
    (ROOT / "revision/closed_loop_figure_manifest.json").write_text(json.dumps({
        "type": "mechanism schematic; no experimental response curve added",
        "source": "Original PPO batched environment and original tiny-student implementation",
        "control_period_s": .02, "plant_step_s": .001, "font_sources": font_sources,
        "explanation": "Sampled parameters define the simulated object. Fixed student weights map observations to actions; force/actuator dynamics drive the object; recorded responses form the time-domain plot."
    }, ensure_ascii=False, indent=2)+"\n")
    print("Rendered closed-loop mechanism: SVG, PDF and PNG")


if __name__ == "__main__":
    main()
