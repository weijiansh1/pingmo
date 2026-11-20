"""Add the missing simulation-loop explanation to the paper, then number figures."""
from pathlib import Path
import hashlib
from bs4 import BeautifulSoup
from integrate_sae import renumber_figures

ROOT = Path(__file__).resolve().parents[1]


def add_closed_loop_explanation(soup):
    if soup.select_one('#closed-loop-simulation'):
        return
    image = ROOT / 'webpage/paper/closed_loop_simulation.svg'
    version = hashlib.sha256(image.read_bytes()).hexdigest()[:12]
    block = BeautifulSoup('''
<p id="closed-loop-explanation">图 0 给出源域学生测试时的闭环仿真及曲线来源。首先由一组品质相关参数确定对象传递函数，并离散为可逐步推进的状态方程。学生每 20 ms 根据当前观测与历史输出动作，动作经力接口和执行机构形成作用力；仿真器以 1 ms 步长推进对象状态，得到新的滚转角速度，再反馈至下一控制周期。按时间记录各时刻响应并连线，即得到后文的时域曲线。单条测试中，对象参数和学生权重保持固定，动作、控制力与对象状态随反馈变化。</p>
<figure class="w100" id="closed-loop-simulation" data-source="original PPO simulation and tiny-student input contracts">
<a class="figure-zoom" target="_blank" title="点击查看闭环仿真图"><img alt="图 0　学生闭环仿真与时域曲线的生成：采样参数确定飞机模型；学生输出动作，经力接口与执行机构作用于模型；新的角速度反馈至学生，并按时间记录成响应曲线。"/></a>
<figcaption>图 0　学生闭环仿真与时域曲线的生成。品质相关参数确定被控对象；学生读取观测和历史，输出力速率动作；对象在控制力作用下逐步产生角速度响应。将响应序列按时间记录和绘制，得到后文的时域曲线。图中展示仿真计算链路，指令与目标信号沿用各组实验定义。</figcaption>
</figure>''', 'html.parser')
    source = f'paper/closed_loop_simulation.svg?v={version}'
    block.find('img')['src'] = source
    block.find('a')['href'] = source
    anchor = soup.select_one('#s21').find_next_sibling('p')
    for element in reversed(list(block.find_all(['p', 'figure'], recursive=False))):
        anchor.insert_after(element.extract())
    renumber_figures(soup)
