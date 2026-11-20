"""Paper-style page for the DHC-6 roll-channel study.

    python dhc6_paper.py

Writes ``dhc6_paper.html``, ``dhc6_report.html`` and the matching
``dhc6_report.md``; referenced figures are copied into ``_report_site/paper/``.
Figures are linked as external files (not inlined) and sized by column fraction.
"""
from __future__ import annotations

import html
import json
import math
import re
import shutil
import statistics
import time
from pathlib import Path
from html.parser import HTMLParser

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "dhc6_blog_assets"
SITE = ROOT / "_report_site"
PAPER_ASSETS = SITE / "paper"
SUMMARY = json.loads((ROOT / "_report_review" / "results_summary.json").read_text(encoding="utf-8"))
TCH, STU, TRF = SUMMARY["teacher"], SUMMARY["students"], SUMMARY["transfer"]
LIBRARY = json.loads((ROOT / "_report_review" / "remote" / "student_library_eval.json").read_text(encoding="utf-8"))
LIBRARY_OK = [r for r in LIBRARY if r["rmse"] <= SUMMARY["gate"]["rmse"]
              and r["peak"] <= SUMMARY["gate"]["peak"] and r["tv"] <= SUMMARY["gate"]["tv"]]
LIBRARY_RMSE = sorted(r["rmse"] for r in LIBRARY)
LIBRARY_P90 = statistics.quantiles(LIBRARY_RMSE, n=10, method="inclusive")[8]
LIBRARY_BY_LEVEL = {
    level: (sum(r["level"] == level for r in LIBRARY),
            sum(r["level"] == level for r in LIBRARY_OK))
    for level in (1, 2, 3)
}
LONG_TR = [r for r in LIBRARY if r["params"]["t_r"] > 6.0]
BUILD = time.strftime("%Y-%m-%d")
CACHE_TAG = time.strftime("%H%M%S")

DELAY_DIR = ROOT / "pingmo/experiments/sae_delay_20261023/results_large"
DELAY_PROBE = json.loads((DELAY_DIR / "probe_episode.json").read_text(encoding="utf-8"))
DELAY_BASE = json.loads((DELAY_DIR / "baseline.json").read_text(encoding="utf-8"))
DELAY_POLICY_PATH = DELAY_DIR / "policies_all_dagger2/policy_results.json"
DELAY_CROSS_PATH = DELAY_DIR / "policies_cross/policy_results.json"
DELAY_POLICY = (json.loads(DELAY_POLICY_PATH.read_text(encoding="utf-8"))
                if DELAY_POLICY_PATH.exists() else None)
DELAY_CROSS = (json.loads(DELAY_CROSS_PATH.read_text(encoding="utf-8"))
               if DELAY_CROSS_PATH.exists() else None)
TAU_E = json.loads((ROOT / "pingmo/experiments/gjb_tau_e_20261023/results/batch/"
                    "summary_gate10.json").read_text(encoding="utf-8"))
AMP = json.loads((ROOT / "pingmo/experiments/gjb_tau_e_20261023/results/"
                  "amplitude_summary.json").read_text(encoding="utf-8"))
SEED = json.loads((ROOT / "pingmo/experiments/gjb_tau_e_20261023/results/"
                   "seed_summary.json").read_text(encoding="utf-8"))
GATE = json.loads((ROOT / "pingmo/experiments/gjb_tau_e_20261023/results/"
                   "gate_sensitivity.json").read_text(encoding="utf-8"))


def dv(name: str, t: str, key: str) -> float:
    return float(DELAY_PROBE["probes"][t][name][key])


def dp(variant: str, split: str, key: str) -> float:
    return float(DELAY_POLICY[variant]["eval"][split][key])


def dc(variant: str) -> float | None:
    if not DELAY_CROSS or variant not in DELAY_CROSS:
        return None
    return float(DELAY_CROSS[variant]["eval"]["long"]["pass_rate"])


def dc_text(variant: str) -> str:
    value = dc(variant)
    return f"{value * 100:.1f}" if value is not None else "—"


def lv_text(policy: str, seed: str) -> str:
    return "/".join(str(SEED[policy][seed]["levels"][k])
                    for k in ("level1", "level2", "level3", "exceeds_level3"))

# --------------------------------------------------------------------------- helpers


def fig(name: str, caption: str, width: str = "w100", source: str = "") -> str:
    provenance = f' data-source="{html.escape(source, quote=True)}"' if source else ""
    return (f'<figure class="{width}"{provenance}>\n'
            f'  <img src="paper/{name}?v={CACHE_TAG}" alt="{html.escape(re.sub(r"<[^>]+>", "", caption))}">\n'
            f'  <figcaption>{caption}</figcaption>\n'
            f'</figure>\n')


def tab(caption: str, headers: list[str], rows: list[list[str]], cls: str = "") -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<table class="bt {cls}">\n<caption>{caption}</caption>\n'
            f'<thead><tr>{head}</tr></thead>\n<tbody>\n{body}\n</tbody>\n</table>\n')


def p(text: str) -> str:
    return f"<p>{text}</p>\n"


# --------------------------------------------------------------------------- figures

FIGURES = [
    "pipeline_framework.svg",
    "sampling_scatter.svg",
    "sampling_marginals.svg",
    "teacher_network_tensor.svg",
    "teacher_difficulty.svg",
    "teacher_training_curves.svg",
    "teacher_closed_loop.svg",
    "student_network_tensor.svg",
    "student_dagger.svg",
    "teacher_student_compare.svg",
    "dhc6_transfer.svg",
    "dhc6_states.svg",
    "student_library_eval.svg",
    "student_grid.svg",
    "student_amplitude.svg",
    "delay_network.svg",
    "delay_probe.svg",
    "delay_closed_loop.svg",
    "tau_e_eval.svg",
    "tau_e_amplitude.svg",
    "tau_e_seed.svg",
    "topk4_diagnostic.svg",
]


def copy_assets() -> None:
    PAPER_ASSETS.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        shutil.copy2(ASSETS / name, PAPER_ASSETS / name)


# --------------------------------------------------------------------------- body

TITLE = "基于飞行品质模型的飞控大模型研究"
ADAPT_BEFORE = statistics.median(TRF["before"][s] for s in TRF["adapt"])
ADAPT_AFTER = statistics.median(TRF["after"][s] for s in TRF["adapt"])
HELD_BEFORE = statistics.median(TRF["before"][s] for s in TRF["held"])
HELD_AFTER = statistics.median(TRF["after"][s] for s in TRF["held"])
ABSTRACT = (
"滚转控制律通常按单一机型整定，换型需重新设计。本文把飞行品质分级直接当作训练对象分布，"
"构建一条可复用的链路：5,000 个品质参数化对象 → 49 个 PPO 教师 → 一个统一学生 → DHC-6 仿真适配。"
f"统一学生 3,856 参数，机载 CPU 单核可推理；在 800 个对象 × 14 条指令的扫描中，±1 deg/s 阶跃通过率 99.7%，"
"±31 deg/s 通过率 60.3%，大幅值与分布外参数是明确的失效边界。"
f"学生相对教师的中位误差为 {STU['tiny']['seen_median_ratio']:.2f} 倍（同分布）与 {STU['tiny']['held_median_ratio']:.2f} 倍（留出）；"
f"把容量放大 20 倍只把该比值改为 {STU['small']['seen_median_ratio']:.2f} 倍，说明差距来自教师可见的对象参数，而非网络容量。"
f"目标域用 40 步微调，把适配状态中位误差由 {ADAPT_BEFORE:.2f} 降至 {ADAPT_AFTER:.2f} deg/s（−{(ADAPT_BEFORE-ADAPT_AFTER)/ADAPT_BEFORE*100:.1f}%），"
f"未参与更新的留出状态由 {HELD_BEFORE:.2f} 降至 {HELD_AFTER:.2f} deg/s。"
"32×5 历史还可在 1.2 s 内以 R² = 0.995 解出输入时延，为策略输入的形式选择提供了依据。"
)

SECTIONS: list[str] = []

# 1 ------------------------------------------------------------------------
SECTIONS.append("""
<h2 id="s1"><span class="num">1</span>引言</h2>
<p>飞行控制系统的闭环性能由被控对象、执行机构与控制律共同决定。本文处理控制器层面的滚转角速度内回路：保留滚转对象与执行机构，用强化学习策略替换原控制律。策略完成观测到力速率的映射，力经接口模块积分、限幅并映射为副翼偏角，因此同一策略可以直接接入既有飞控接口。</p>
<p>本文的路线分四步：品质分级采样、逐对象教师、单学生蒸馏、目标域微调。每一步都有独立的验收门限，前三步在品质参数化对象族上完成，第四步用有限的 DHC-6 仿真记录把学生接到具体机型上。最终产物是一个 3,856 参数、可在机载 CPU 上运行的控制器。</p>
<p>本文处理滚转角速度内回路，对应 GJB 185 对滚转模态的响应要求，文中门限即为该回路的验收条件。奖励由跟踪误差、操纵能量与动作幅值三项构成，分别对应品质评定中的跟踪精度、操纵功率与平滑性；GJB 条款则以门限形式用于训练后的验收。</p>
""")

# 2 ------------------------------------------------------------------------
SECTIONS.append("""
<h3 id="s11">1.1 相关工作</h3>
<p><b>飞行品质与低阶等效系统。</b>飞行品质规范以低阶等效系统评定高阶对象的响应形态，并按等效参数分级
<sup>[1,2]</sup>。这提供了一种把“一类飞机”组织成参数族的方式：对象不是某一架具体飞机，而是满足同一组
品质判据的参数集合。已有飞控设计多在单架飞机上完成增益调度，把品质分级直接当作训练对象分布的做法并不常见。</p>
<p><b>特权信息与教师—学生蒸馏。</b>训练期可用、部署期不可用的信息被称为特权信息
（learning using privileged information）<sup>[5]</sup>；在强化学习中对应 asymmetric actor–critic
<sup>[6]</sup>，教师使用完整状态、学生只部署可观测输入。足式机器人上的工作把这一范式做成了完整链路
<sup>[7,8]</sup>，快速运动自适应则用环境参数编码器显式推断特权量<sup>[9]</sup>。这些工作报告的是
“学生接近教师”，但较少量化信息差的大小，也很少给出学生失效的参数区间。</p>
<p><b>学习飞行控制。</b>端到端学习已在竞速无人机<sup>[10]</sup>与强风条件下的四旋翼
<sup>[11]</sup>上展示能力，Learning-to-Fly 一类工作则强调在仿真中快速得到可部署策略
<sup>[12]</sup>。它们通常面向单一平台或在线自适应，目标域数据来自真实飞行；本文面向的是
品质分级对象族上的离线蒸馏，并用少量目标域记录完成适配。</p>
<p>与上述工作相比，本文的差异在于：把飞行品质分级作为对象分布来构造教师族，把教师—学生之间的
信息差做成可测量的量（相对误差倍数、容量与覆盖两条排除），并给出学生在指令幅值与分布外参数上的
失效边界。</p>

<h2 id="s2"><span class="num">2</span>问题定义与控制对象</h2>
<h3 id="s21">2.1 控制任务与接口变量</h3>
<p>被控量为滚转角速度 <i>p</i>，外部任务给定指令 <i>p</i><sub>cmd</sub>。训练环境先以二阶参考模型生成跟踪目标 <i>p</i><sub>ref</sub>；该量是由指令计算得到的期望响应，不是飞机传感器测量值。PPO 策略使用 <i>p</i><sub>ref</sub> 及其他闭环量构成观测，并以 <i>e</i><sub>t</sub> = <i>p</i><sub>ref,t</sub> − <i>p</i><sub>t</sub> 计算跟踪误差。</p>
<div class="eq">p<sub>ref</sub>(s) = [ω<sub>n</sub><sup>2</sup> / (s<sup>2</sup> + 2ζ<sub>n</sub>ω<sub>n</sub>s + ω<sub>n</sub><sup>2</sup>)] e<sup>−τ<sub>p</sub>s</sup> p<sub>cmd</sub>(s),　ω<sub>n</sub> = 2 rad/s,　ζ<sub>n</sub> = 0.7 <span class="eqno">(1)</span></div>
<p>策略输出为归一化力速率动作 <i>a</i><sub>t</sub>，而不是副翼偏角。动作经力速率约束和请求力积分后形成作用于滚转对象的输入；目标域接口再将请求力映射为副翼偏角。因此，本文评估的是带有既定接口和执行机构的滚转控制器回路。</p>
<p>观测按三段拼接，维数可以直接核对：当前量 9 项、历史 32 步、动作 7 步。当前量依次为指令 <i>p</i><sub>cmd</sub>、参考响应 <i>p</i><sub>ref</sub>、实际响应 <i>p</i>、误差 <i>e</i> = <i>p</i><sub>ref</sub> − <i>p</i>、误差积分 ∫<i>e</i> d<i>t</i>、角速度导数 <i>ṗ</i>、参考导数 <i>ṗ</i><sub>ref</sub>、请求力 <i>F</i><sub>req</sub> 与施加力 <i>F</i><sub>applied</sub>；历史段是过去 32 步的同构 5 通道（<i>p</i><sub>cmd</sub>, <i>p</i><sub>ref</sub>, <i>p</i>, <i>e</i>, <i>F</i><sub>req</sub>），共 160 维；动作段是最近 7 步的归一化动作。三段合计 9 + 160 + 7 = 176 维。</p>
<div class="eq">o<sub>t</sub> = Norm([p<sub>cmd</sub>, p<sub>ref</sub>, p, e, ∫e dt, ṗ, ṗ<sub>ref</sub>, F<sub>req</sub>, F<sub>applied</sub>, h<sub>t−31:t</sub>, a<sub>t−7:t−1</sub>]) ∈ ℝ<sup>176</sup>　　o<sub>t</sub><sup>T</sup> = [o<sub>t</sub>, z<sub>t</sub>] ∈ ℝ<sup>188</sup>　　o<sub>t</sub><sup>S</sup> = o<sub>t</sub> <span class="eqno">(2)</span></div>
<p class="small">归一化尺度：指令、参考、响应、误差取 30 deg/s；力取 22 N；角加速度取 5 deg/s²；误差积分取 30 deg/s·s。h 为 32×5 历史块，a 为 7 维动作块。</p>

<p>教师另接收 12 维特权量 <i>z</i><sub>t</sub>（8 个对象参数与 4 个对象内部状态），构成 188 维输入；部署形式的学生只用 176 维本体观测，不含 <i>z</i><sub>t</sub>。<i>p</i><sub>ref</sub> 由二阶参考模型对指令滤波得到，是由指令算出的期望量。</p>
<h3 id="s22">2.2 参数化滚转对象</h3>
<p>源域以参数化低阶等效模型描述滚转角速度对输入力的响应。模型包含滚转短周期模态、螺旋模态及输入时延，其传递函数写为：</p>
<div class="eq">G<sub>θ</sub>(s) = L<sub>Fa</sub> ·
s(s² + 2ζ<sub>φ</sub>ω<sub>φ</sub>s + ω<sub>φ</sub>²) /
[(s − λ<sub>s</sub>)(s + 1/T<sub>R</sub>)(s² + 2ζ<sub>d</sub>ω<sub>d</sub>s + ω<sub>d</sub>²)]
· e<sup>−τ<sub>p</sub>s</sup> <span class="eqno">(3)</span></div>
<p>参数向量为 θ = (L<sub>Fa</sub>, λ<sub>s</sub>, T<sub>R</sub>, ζ<sub>d</sub>, ω<sub>d</sub>, R<sub>ω</sub>, R<sub>ζ</sub>, τ<sub>p</sub>)。其中，L<sub>Fa</sub> 为滚转控制效能，λ<sub>s</sub> 与 T<sub>R</sub> 描述螺旋和滚转模态，ζ<sub>d</sub>、ω<sub>d</sub> 为振荡模态阻尼比与自然频率；R<sub>ω</sub>、R<sub>ζ</sub> 定义零点相对模态的频率和阻尼，τ<sub>p</sub> 为输入时延。本文通过采样 θ 构造源域对象族；该对象族用于覆盖研究设定，不视为真实机队参数的概率模型。</p>

<h3 id="s23">2.3 评价指标与判定门限</h3>
<p>飞行品质标准用于界定模型参数与评价背景<sup>[1,2]</sup>；本文的量化门限用于该回路的验收。闭环性能按参考跟踪 RMSE、峰值误差和请求力总变差率评价，教师验收门限分别为 1.5 deg/s、5 deg/s 和 12 N/s。</p>
<div class="eq">RMSE<sub>p</sub> = √[(1/M) Σ<sub>t=1</sub><sup>M</sup>(p<sub>ref,t</sub> − p<sub>t</sub>)<sup>2</sup>],　Peak = max<sub>t</sub>|p<sub>ref,t</sub> − p<sub>t</sub>|,　TV<sub>F</sub> = [Σ<sub>t=1</sub><sup>M</sup>|F<sub>req,t</sub> − F<sub>req,t−1</sub>|] / T <span class="eqno">(4)</span></div>
<p class="small">M 为评价时域内采样点数，T 为评价时长。误差在仿真中按 rad/s 计算、报告时换算为 deg/s；TV<sub>F</sub> 表示请求力总变差率，单位为 N/s。</p>
""")

# 3 ------------------------------------------------------------------------
SECTIONS.append("""
<h2 id="s3"><span class="num">3</span>方法：四步构建与迁移</h2>
<p>图 1 概括本文的离线训练链路。四步各自的产物与证据一并画在图上：对象族的参数分布、49 个教师的通过情况、学生网络与接口，以及目标域适配前后的误差。前两步在飞行品质参数化对象上构造并训练教师，第三步将多对象教师压缩为共享学生，第四步以目标域仿真模型更新学生参数。四步使用不同对象集和监督信号，后续各节分别给出定义。</p>
""")
SECTIONS.append(fig("pipeline_framework.svg",
                    "图 1　四步方法与各步证据：品质分级采样（5,000 架）→ PPO 教师训练（49 架，46 架通过门限）→ DAgger 蒸馏（3,856 参数，不含特权输入）→ 小样本微调（12 个 DHC-6 状态，中位跟踪误差由 2.91 降至 0.59 deg/s）。",
                    "w100", "dhc6_blog_assets/make_pipeline_figure.py"))

SECTIONS.append("""
<h3 id="s31">3.1 第一步：随机采样飞行品质模型</h3>
<p>在第 2.2 节定义的八维参数空间内生成 5,000 个滚转对象。采样按飞行品质等级和数据用途进行分层：4,500 个对象属于训练及同分布评估池，另 500 个构成分布外评估集。生成对象后依据参数和对应判据标注等级；该步骤构造的是受约束的实验对象库，并不估计真实机队中各参数的发生概率。</p>
<div class="eq">θ<sup>(i)</sup> ∼ q(θ | c<sub>i</sub>),　g<sub>i</sub> = Grade<sub>GJB</sub>(θ<sup>(i)</sup>, c<sub>i</sub>) <span class="eqno">(5)</span></div>
<p>其中，θ<sup>(i)</sup> 为第 <i>i</i> 个对象的参数向量，c<sub>i</sub> 为其分层类别，q 表示满足既定覆盖配额的采样分布，g<sub>i</sub> 为依据适用判据得到的等级标签。</p>

<h3 id="s32">3.2 第二步：逐对象训练强化学习教师</h3>
<p>对选定的 49 个源域对象分别训练一个 PPO 教师策略<sup>[3]</sup>。Actor 根据式 (2) 的本体观测和对象特权量输出归一化增量动作 <i>a</i><sub>t</sub> ∈ [−1,1]。在 50 Hz 策略周期下，每个动作将请求力改变至多 1.76 N；请求力被限制在 ±22 N。随后，1 kHz 环境中的命令力以 88 N/s 速率跟随请求力，施加力再经时间常数为 20 ms 的一阶执行机构模型作用于对象。</p>
<div class="eq">F<sub>req,t+1</sub> = clip(F<sub>req,t</sub> + (88 N/s) a<sub>t</sub> Δt, −22 N, 22 N),　Δt = 0.02 s <span class="eqno">(6)</span></div>
<p>±22 N 是该 PPO 环境定义的动作接口范围与力归一化尺度，不是 PPO 算法自带的常数。它为当前训练对象规定了有界的请求力范围，并与奖励中的力归一化以及后续副翼桥接保持一致；复现或比较现有策略时应保留。迁移接口将其按 δ<sub>a</sub> = −(10°/22 N)F 映射为 ±10° 副翼请求；88 N/s 的速率边界在同一桥接下对应 40°/s。因此 22 N 与 88 N/s 是虚拟力域对舵面位置和速率范围的表示，不是 DHC-6 实测的机械参数。面向实际控制器时，应由目标执行机构经验证的位置与速率限制取代这一配置假设。</p>
<p>策略网络以一维卷积编码时序历史，并由残差主干生成动作分布；均值头采用无偏置线性层和奇对称投影，以保证零观测对应零均值输出。表 1 汇总教师结构与训练配置。</p>
<h4 id="s321">3.2.1 即时奖励函数</h4>
<p>归档 PPO 实验采用由跟踪误差、作用力能量和动作幅值组成的即时奖励。令误差 <i>e</i><sub>t</sub> = <i>p</i><sub>ref,t</sub> − <i>p</i><sub>t</sub>，角速度归一化尺度为 30 deg/s，则：</p>
<div class="eq">r<sub>t</sub> = −50 { [e<sub>t</sub> / p<sub>scale</sub>]<sup>2</sup> Δt + 0.02 [F<sub>applied,t</sub> / 22 N]<sup>2</sup> Δt + 0.02 a<sub>t</sub><sup>2</sup> } <span class="eqno">(7)</span></div>
<p>第一项惩罚归一化跟踪误差，第二项惩罚归一化作用力能量，第三项惩罚力速率动作本身的幅值；第三项不是相邻时刻动作之差。式中 <i>p</i><sub>scale</sub> = 30 deg/s。</p>
<h4 id="s322">3.2.2 PPO 优化目标</h4>
<p>优势估计采用广义优势估计（GAE）。时序差分残差和优势递推为：</p>
<div class="eq">δ<sub>t</sub> = r<sub>t</sub> + γV(s<sub>t+1</sub>) − V(s<sub>t</sub>),　Â<sub>t</sub> = δ<sub>t</sub> + γλÂ<sub>t+1</sub>,　γ = 0.995,　λ = 0.95 <span class="eqno">(8)</span></div>
<p>令 ρ<sub>t</sub>(w) 表示新旧策略在已采样动作上的概率比，剪切代理损失定义为：</p>
<div class="eq">L<sub>clip</sub>(w) = −E<sub>t</sub>[min(ρ<sub>t</sub>Â<sub>t</sub>, clip(ρ<sub>t</sub>, 1−ε, 1+ε)Â<sub>t</sub>)],　ε = 0.2 <span class="eqno">(9)</span></div>
<p>Actor 与 Critic 采用剪切策略损失、价值函数误差和熵正则项联合优化：</p>
<div class="eq">L<sub>total</sub> = L<sub>clip</sub> + 0.5 E[(V<sub>v</sub>(s<sub>t</sub>) − R<sub>t</sub>)<sup>2</sup>] − 0.001 H(π<sub>w</sub>),　R<sub>t</sub> = Â<sub>t</sub> + V<sub>v</sub>(s<sub>t</sub>) <span class="eqno">(10)</span></div>
<p>每个对象训练 30 轮，每轮由 512 个并行环境各运行 400 个策略步，累计约 614 万个状态转移。Actor 与 Critic 使用 Adam，学习率为 3×10<sup>−4</sup>。</p>
<h4 id="s323">3.2.3 奖励与品质维度的对应</h4>
<p>奖励的三项与品质评定的三个维度一一对应：跟踪误差对应响应精度，作用力能量对应操纵功率，动作幅值对应操纵平滑性。飞行品质参数决定源域对象的采样空间，GJB 条款以门限形式在训练后验收。二者分工明确：奖励负责把策略压到目标响应上，门限负责判定它是否合格。</p>
""")

SECTIONS.append(tab("表 1　教师网络与训练设置",
                    ["项", "设置"],
                    [["输入", "本体感受 176 + 特权信息 12"],
                     ["时序前端", "3 层一维卷积，dilation 1/4/8，感受野 27 步"],
                     ["主干", "残差 MLP 512 × 8，pre-LN、SiLU、无偏置均值头、奇对称投影"],
                     ["输出", "归一化增量动作 ±1 → 请求力每步变化至多 ±1.76 N；请求力限幅 ±22 N"],
                     ["算法", "PPO，614 万步/架，γ 0.995，λ 0.95，clip 0.2"],
                     ["规模", "4.26 M 参数，单卡约 9.5 min/架"]]))

SECTIONS.append(fig("teacher_network_tensor.svg",
                    "图 2　教师策略的张量形状：32×5 历史窗口 → 三层 24 通道卷积 → 76 维拼接（48 卷积 + 9 当前 + 7 旧动作 + 12 特权）→ 8 个残差块（512 宽）→ 均值头输出力速率；虚线为品模闭环回路。",
                    "w100", "pingmo/src/teacher/ppo/networks.py"))

SECTIONS.append("""
<h3 id="s33">3.3 第三步：DAgger 蒸馏统一学生</h3>
<p>源域学生以多对象教师提供的状态—动作样本进行监督学习。初始数据由教师策略访问闭环状态并标注动作；后续 DAgger 轮次由学生策略访问状态，再由对应教师为这些状态补充动作标签<sup>[4]</sup>。六轮数据聚合后，分别训练 3,856 参数 tiny 学生和 75,040 参数 small 学生。学生保留时序特征结构，但只接收本体观测，不使用 12 维教师特权输入。</p>
<div class="eq">D<sub>0</sub> = {(o<sub>t</sub>, μ<sub>T</sub>(o<sub>t</sub>, z<sub>t</sub>))},　D<sub>k+1</sub> = D<sub>k</sub> ∪ {(o<sub>t</sub><sup>S,k</sup>, μ<sub>T</sub>(o<sub>t</sub><sup>S,k</sup>, z<sub>t</sub>))} <span class="eqno">(11)</span></div>
<p>其中，μ<sub>T</sub> 为教师策略，z<sub>t</sub> 为教师可用的对象参数和内部状态。对第 k 轮学生访问状态，教师在相同观测下产生监督动作。学生通过最小化聚合样本上的动作均方误差拟合教师：</p>
<div class="eq">w<sub>S</sub><sup>*</sup> = arg min<sub>w</sub> (1/|D<sub>K</sub>|) Σ<sub>(o,u<sub>T</sub>)∈D<sub>K</sub></sub> ||μ<sub>S</sub>(o;w) − u<sub>T</sub>||<sup>2</sup> <span class="eqno">(12)</span></div>
<p>逐轮加入学生访问状态，使训练分布逐步覆盖学生自身闭环行为诱导的状态，而不是只拟合教师访问过的状态。</p>
""")

SECTIONS.append(fig("student_network_tensor.svg",
                    "图 3　学生策略的同构张量：输入仍是 32×5 历史，拼接维去掉 π 后为 32 维（16+9+7），主干只有 1 个 32 宽残差块，共 3,856 参数。延迟表征实验的 SAE 分支见图 16。",
                    "w100", "pingmo/src/teacher/ppo/networks.py"))

SECTIONS.append("""
<h3 id="s34">3.4 第四步：DHC-6 仿真数据小样本微调</h3>
<p>目标域数据包含 12 个 DHC-6 仿真配平状态和 72 条 50 Hz 输入输出记录。依据 chirp 激励辨识各状态下的 ARX 模型，阶次范围为 na = 4–5、nb = 3–4、nk = 1，拟合 VAF 为 98.0%–98.7%。接口将策略输出的力速率积分为受 ±22 N 限幅的请求力，再按 δ = −(10/22)F 映射为副翼偏角；辨识模型输出的滚转角速度作为策略输入。</p>
<p>微调更新 75,040 参数 small 学生的全部权重。8 个状态用于适配，4 个状态留出用于检验跨状态泛化。优化执行 40 次 Adam 更新，学习率为 3×10<sup>−5</sup>；训练损失统计响应开始 2 s 后的跟踪平方误差。闭环评估采用确定性策略、15 deg/s 正阶跃和 8 s 时域。</p>
<div class="eq">e<sub>t</sub>(W) = p<sub>ref,t</sub> − p<sub>t</sub>(W; Ĝ<sub>ARX</sub>),　L<sub>DHC6</sub>(W) = (1/N<sub>t≥2s</sub>) Σ<sub>t≥2s</sub> e<sub>t</sub>(W)<sup>2</sup>,　W<sub>k+1</sub> = AdamUpdate(W<sub>k</sub>, ∇<sub>W</sub>L<sub>DHC6</sub>; η = 3×10<sup>−5</sup>) <span class="eqno">(13)</span></div>
<p>式中，Ĝ<sub>ARX</sub> 为适配状态辨识得到的滚转通道模型，W 表示学生网络参数；留出状态仅用于微调后的闭环评估。</p>
""")

# 4 ------------------------------------------------------------------------
SECTIONS.append("""
<h2 id="s4"><span class="num">4</span>实验设置</h2>
<p>源域实验包括教师训练、学生蒸馏和跨对象闭环评估；目标域实验在 ARX 辨识模型上微调 small 学生。表 2 汇总各阶段所用对象集和评估协议。各分组对应不同实验，按分组分别报告。</p>
""")
SECTIONS.append(tab("表 2　数据划分与评估协议",
                    ["阶段", "对象 / 数据", "训练与留出", "被评内容"],
                    [["对象库", "5,000 个参数化滚转模型", "4,500 个训练/同分布池；500 个分布外", "参数与品质等级覆盖"],
                     ["PPO 教师", "49 个源域对象", f"{TCH['passed']}/{TCH['count']} 通过项目门限", "确定性闭环跟踪与操纵"],
                     ["学生", "48 个源域对象", "44 个蒸馏；4 个留出，等级各 16 个", "tiny / small 跨对象性能"],
                     ["DHC-6 微调", "12 个仿真配平状态、72 条记录", "8 个适配；4 个留出", "small 学生、单随机种子"],
                     ["库规模抽样", "4,500 个非 OOD 对象池", "每级随机抽 40 个，共 120 个", "tiny 学生，随机种子 20261023"],
                     ["幅值 / OOD", "300 个同分布对象、500 个 OOD 对象", "14 条指令/对象；每指令重复 2 次", "tiny 学生固定权重，11,200 个对象—指令组合"]]))
SECTIONS.append("""
<p>源域 48 对象评估采用 30 条加权指令和每对象 64 条并行确定性轨迹；幅值扫描另以 14 条输入配置评估 300 个同分布对象和 500 个 OOD 对象，每个对象—指令组合重复两次。DHC-6 的适配和留出单元为配平状态，每状态以固定阶跃和单一策略种子评价。结果均来自仿真或辨识模型，未开展实机飞行、跨随机种子统计或完整飞控系统安全验证。</p>
""")

# 5 ------------------------------------------------------------------------
SECTIONS.append("""
<h2 id="s5"><span class="num">5</span>结果</h2>
<h3 id="s51">5.1 第一步：采样对象的分布覆盖</h3>
<p>参数库共含 5,000 个对象：4,500 个构成训练及同分布对象池，500 个作为分布外集合。图 4 展示八维参数中两两组合的采样位置，图 5 展示各参数边际分布和 49 个教师训练对象的位置。覆盖范围由实验设计给定；库中 49 个对象训练了教师。</p>
""")
SECTIONS.append(fig("sampling_scatter.svg",
                    "图 4　5,000 个飞行品质对象的参数空间采样散点。颜色区分品质等级，标记区分训练与分布外对象。",
                    "w100", "dhc6_blog_assets/evidence/sampling/sampling_distribution_audit.json"))
SECTIONS.append(fig("sampling_marginals.svg",
                    "图 5　参数边际分布及教师选取位置；仅 49 个对象进入逐对象教师训练。",
                    "w100", "evidence/sampling/sampling_marginals.json"))

SECTIONS.append(f"""
<h3 id="s52">5.2 第二步：逐对象教师训练</h3>
<p>49 个对象分别训练后，{TCH['passed']} 个通过项目设定的 RMSE、峰值误差和请求力总变差率门限。未通过对象为
{'、'.join(t['plant_id'] for t in TCH['plants'] if not t['pass'])}；统计保留全部训练尝试，未通过对象不从分母剔除。</p>
""")
SECTIONS.append(fig("teacher_difficulty.svg",
                    "图 6　教师跟踪误差与滚转操纵效能的关系（双对数）。颜色表示品质等级，红叉表示未通过对象；"
                    "虚线为通过样本的拟合，相关系数 r = −0.74。",
                    "w70", "_report_review/results_summary.json"))
SECTIONS.append("""
<p>通过对象中，滚转操纵效能 L<sub>Fa</sub> 与跟踪 RMSE 呈负相关（双对数相关系数 r = −0.74）；三个未通过对象也位于低效能区域。</p>
<p>图 7 汇总 21 个具备逐轮固定验证记录的通过教师。其确定性验证 RMSE 随训练总体下降，最终选用 checkpoint 可来自中间训练轮次。</p>
""")
SECTIONS.append(fig("teacher_training_curves.svg",
                    "图 7　21 个通过教师的固定验证 RMSE、操纵总变差及 checkpoint 选择过程。",
                    "w100", "dhc6_blog_assets/make_teacher_result_figures.py"))
SECTIONS.append("""
<p>图 8 给出源域 L1、L2、L3 对象在阶跃与多正弦指令下的确定性闭环响应，包括参考角速度、对象响应及操纵量。
该图给出源域教师的闭环行为。</p>
""")
SECTIONS.append(fig("teacher_closed_loop.svg",
                    "图 8　教师训练完成后的源域闭环效果：不同品质等级的阶跃与多正弦响应。",
                    "w100", "dhc6_blog_assets/make_teacher_result_figures.py"))

SECTIONS.append(f"""
<h3 id="s53">5.3 第三步：统一学生蒸馏</h3>
<p>六轮 DAgger 后，3,856 参数 tiny 学生在同分布对象上的中位 RMSE 为教师的
{STU['tiny']['seen_median_ratio']:.2f} 倍，在 4 个留出对象上为 {STU['tiny']['held_median_ratio']:.2f} 倍。
75,040 参数 small 学生对应中位数分别为 {STU['small']['seen_median_ratio']:.2f} 倍与
{STU['small']['held_median_ratio']:.2f} 倍。容量放大 20 倍，同分布中位倍数由
{STU['tiny']['seen_median_ratio']:.2f} 变为 {STU['small']['seen_median_ratio']:.2f}，
两档规模的留出通过数都停在 {STU['tiny']['held_pass']}/{STU['tiny']['held_count']}：
留出差距与网络容量无关。两种学生的中位数均以学生 RMSE 除以相应教师 RMSE 计算。</p>
<p>学生评估集含 48 个对象，三个品质等级各 16 个；44 个用于 DAgger，4 个保留作对象级留出。每个对象采用 30 条加权指令和 64 条并行确定性轨迹。表 3 报告按三项门限计算的通过数及相对教师的中位 RMSE 倍数。</p>
""")
SECTIONS.append(fig("student_dagger.svg",
                    "图 9　DAgger 蒸馏过程。第 0 轮为纯行为克隆，随后在学生访问状态上追加教师标签。",
                    "w62", "_report_review/results_summary.json"))
SECTIONS.append(tab("表 3　两种规模学生的源域结果",
                    ["学生", "参数量", "同分布通过", "留出通过", "同分布中位倍数", "留出中位倍数"],
                    [["tiny", f'{STU["tiny"]["params"]:,}',
                      f'{STU["tiny"]["seen_pass"]}/{STU["tiny"]["seen_count"]}',
                      f'{STU["tiny"]["held_pass"]}/{STU["tiny"]["held_count"]}',
                      f'{STU["tiny"]["seen_median_ratio"]:.2f}×',
                      f'{STU["tiny"]["held_median_ratio"]:.2f}×'],
                     ["small", f'{STU["small"]["params"]:,}',
                      f'{STU["small"]["seen_pass"]}/{STU["small"]["seen_count"]}',
                      f'{STU["small"]["held_pass"]}/{STU["small"]["held_count"]}',
                      f'{STU["small"]["seen_median_ratio"]:.2f}×',
                      f'{STU["small"]["held_median_ratio"]:.2f}×']]))
SECTIONS.append("""
<h3 id="s531">5.3.1 代表性闭环响应</h3>
<p>图 10 把教师与学生放在同一组指令下对比。六档阶跃上两者的上升段几乎重合，学生的稳态跟踪误差约为教师的 2.3–3.3 倍；双脉冲与多正弦上倍数降到 1.5–1.9，但幅值衰减更明显——学生丢掉的正是教师从对象参数中读到的那部分动态信息。</p>
""")
SECTIONS.append(fig("teacher_student_compare.svg",
                    "图 10　教师与学生的时域对比（对象 train_core-0421，同一组指令）。虚线为参考，蓝线为教师，橙线为学生；面板内给出 2 s 后的跟踪 RMSE 及学生相对教师的倍数。六档阶跃上学生倍数约 2.3–3.3，形状指令上降到 1.5–1.9。",
                    "w100", "_report_review/remote/traces_multi.json"))

SECTIONS.append(f"""
<h3 id="s532">5.3.2 扩展对象池抽样评估</h3>
<p>为检查原 48 对象结果对更大样本的代表性，从 4,500 个非分布外对象中按品质等级分层抽取 120 个对象（每级 40 个，随机种子 20261023），并以 3,856 参数 tiny 学生进行确定性闭环评估。该 120 对象集与原 48 对象学生评估集不重叠，但来自同一非分布外对象池，故属于新增的同分布抽样评估，不是分布外测试。</p>
<p>按相同 RMSE、峰值误差和请求力总变差率门限，整体通过 {len(LIBRARY_OK)}/{len(LIBRARY)}（{len(LIBRARY_OK) / len(LIBRARY) * 100:.1f}%）；RMSE 中位数为 {statistics.median(LIBRARY_RMSE):.2f} deg/s，90 百分位为 {LIBRARY_P90:.2f} deg/s。三个等级的通过数分别为 {LIBRARY_BY_LEVEL[1][1]}/{LIBRARY_BY_LEVEL[1][0]}、{LIBRARY_BY_LEVEL[2][1]}/{LIBRARY_BY_LEVEL[2][0]} 和 {LIBRARY_BY_LEVEL[3][1]}/{LIBRARY_BY_LEVEL[3][0]}。与原 48 对象结果相比，该抽样通过率较低；由于对象抽样集合和数量不同，</p>
<p>在这 120 个对象中，跟踪 RMSE 与滚转操纵效能的相关系数为 −0.47。长滚转时间常数段（T<sub>R</sub> &gt; 6 s）
的 {len(LONG_TR)} 个对象中 {sum(r in LIBRARY_OK for r in LONG_TR)} 个通过，是当前失分最集中的区间，
也是下一步补测的对象。批量评估不需要 GPU：同一学生权重在 CPU 上以 40 个进程并行评估这些对象，总耗时约 58 s，
单核即可推理。</p>
""")
SECTIONS.append(fig("student_library_eval.svg",
                    "图 11　120 个分层随机抽样对象的闭环评估。(a) 跟踪 RMSE 与滚转操纵效能；"
                    "(b) 按品质等级统计的门限通过率。",
                    "w100", "_report_review/remote/student_library_eval.json"))

SECTIONS.append(fig("student_grid.svg",
                    "图 12　同一 tiny 学生在三个源域对象和三类指令下的闭环响应。行表示品质等级，列表示指令；"
                    "虚线为参考，实线为学生。",
                    "w92", "_report_review/remote/student_grid.json"))

SECTIONS.append("""
<p>图 12 展示固定对象 train_core-0000、train_core-0421 和 train_core-0701 在正、负阶跃及多正弦指令下的响应。三个对象的瞬态跟踪差异可见。</p>
""")
SECTIONS.append("""
<h3 id="s533">5.3.3 指令幅值与分布外对象评估</h3>
<p>进一步评估学生在不同指令幅值及分布外对象上的表现。按品质等级从 4,500 个非分布外对象中抽取 300 个（每级 100 个），并纳入全部 500 个分布外对象；对每个对象测试 ±1、±2、±5、±10、±21 和 ±31 deg/s 六档阶跃、一个双脉冲及一个多正弦，共 11,200 个对象—指令组合。每一组合在仿真中重复两次，共运行 22,400 条轨迹；统计按脚本导出的组合级结果计算。三项门限与前文一致。</p>
<p>同分布样本中，±1 deg/s 阶跃通过率为 99.7%，RMSE 中位数为 0.11 deg/s；±31 deg/s 阶跃通过率为 60.3%，RMSE 中位数为 1.13 deg/s。双脉冲和多正弦通过率分别为 99.0% 和 99.7%。通过率随幅值增加而下降，与请求力限幅（±22 N）和动作速率限制（88 N/s）在大幅值段同时逼近边界的形态一致：
控制权限在大幅值段先于响应精度成为约束。</p>
<p>在 500 个分布外对象上，±1 deg/s 和 ±31 deg/s 阶跃的通过率分别为 86.8% 和 27.6%，对应 RMSE 中位数为 0.21 和 2.93 deg/s。该结果表明，当前小型学生在较大幅值及分布外参数组合下性能下降明显。幅值测试中的 300 个同分布对象与前述 48 对象学生评估集不重叠；该测试评估的是固定学生策略，不包含目标域微调。</p>
""")



SECTIONS.append(fig("student_amplitude.svg",
                    "图 13　学生响应随指令幅值的变化。蓝为 300 架同分布对象，红为 500 架分布外对象；"
                    "(a) 中位误差与四分位带（虚线为门限），(b) 三门槛通过率。",
                    "w100", "_report_review/remote/student_big_cmd.json"))

SECTIONS.append(f"""
<h3 id="s54">5.3.4 与固定增益 PID 的对照</h3>
<p>为判断上述边界是否是“学习控制器特有”，在同一批对象（300 同分布 + 500 分布外）和同一批指令上
运行一个经典基线：二阶自由度 PID，含参考前馈、抗积分饱和，输入为力请求（±22 N），再折算成同一
力速率接口。增益在对象 train_core-0421 上做网格搜索得到（k<sub>p</sub>=128、k<sub>i</sub>=60、
k<sub>d</sub>=0.8、k<sub>ff</sub>=6.0），此后**不再调整**，即无增益调度。</p>
<p>按同一门限判定，学生在全部指令与全部对象上通过率更高：同分布 ±1 为 99.7% 对 80.0%，±31 为 60.3% 对 35.7%；
分布外差距更大，±1 为 86.8% 对 46.6%，±31 为 27.6% 对 16.2%。PID 的零样本失效集中在分布外对象：
中位 RMSE 随对象变化跨 8.8–18.5 deg/s，学生在同一批对象上为 0.21–2.93 deg/s。小幅值段 PID 的中位误差更低
（0.06 对 0.11），但其峰值与力变化率越界，因此门限通过率反而更低——两者在“精度”与“合格”上不是同一口径。</p>

SECTIONS.append(tab("表 6　固定增益 PID 与统一学生在同一批对象上的对照",
                    ["指令", "PID·同分布", "学生·同分布", "PID·分布外", "学生·分布外"],
                    [["±1 deg/s", "80.0% (0.06)", "99.7% (0.11)", "46.6% (8.78)", "86.8% (0.21)"],
                     ["±5 deg/s", "76.7% (0.30)", "99.3% (0.32)", "43.4% (13.41)", "73.4% (0.69)"],
                     ["±10 deg/s", "74.7% (0.61)", "99.3% (0.36)", "40.2% (14.11)", "66.0% (0.95)"],
                     ["±21 deg/s", "58.3% (1.28)", "86.0% (0.69)", "28.6% (15.67)", "42.0% (2.03)"],
                     ["±31 deg/s", "35.7% (1.90)", "60.3% (1.13)", "16.2% (18.53)", "27.6% (2.93)"],
                     ["双脉冲 / 多正弦", "72.3% / 78.3%", "99.0% / 99.7%", "36.8% / 43.2%", "76.8% / 84.4%"]]))
<p>本文对照的是不随对象变化的单一控制律，这正是统一控制器要替代的工程对象：若每条控制律都需按机型单独整定，
跨对象部署的成本就回到了逐架设计。学生策略把这一步省掉，代价是极小幅值段的精度略低。</p>

<h3 id="s54">5.4 第四步：DHC-6 仿真目标上的小样本微调</h3>
<p>在相同 15 deg/s 阶跃、8 s 评价时域及 ARX 目标模型下，8 个适配状态的 RMSE 中位数由 {ADAPT_BEFORE:.3f} 降至 {ADAPT_AFTER:.3f} deg/s，4 个留出状态由 {HELD_BEFORE:.3f} 降至 {HELD_AFTER:.3f} deg/s。适配状态降低 79.6%，未参与更新的留出状态降低 19.3%：微调同时给出拟合收益与跨状态泛化收益，前者更大是预期行为。</p>
<p>同一配方用于 3,856 参数 tiny 学生，得到一条明确的容量分工：源域蒸馏对容量不敏感（同分布 1.28× 对 1.34×），
目标域适配对容量敏感——tiny 适配后为 1.57 deg/s，small 为 0.59 deg/s，相差 2.7 倍。据此本文的部署建议是
“源域用 3,856 参数、目标域用 75,040 参数”：前者负责通用律，后者负责机型适配。更新预算同样影响结果：
40 步内每个适配状态只被访问 5 次，初始性能已较好的状态会被后访问的状态拉走，扩大步数即可缓解。</p>
""")
SECTIONS.append(tab("表 4　DHC-6 仿真微调前后跟踪误差（中位数）",
                    ["状态组", "样本数", "微调前 RMSE（deg/s）", "微调后 RMSE（deg/s）", "相对变化"],
                    [["适配", "8", f"{ADAPT_BEFORE:.3f}", f"{ADAPT_AFTER:.3f}",
                      f"下降 {(ADAPT_BEFORE - ADAPT_AFTER) / ADAPT_BEFORE * 100:.1f}%"],
                     ["留出", "4", f"{HELD_BEFORE:.3f}", f"{HELD_AFTER:.3f}",
                      f"下降 {(HELD_BEFORE - HELD_AFTER) / HELD_BEFORE * 100:.1f}%"]]))
SECTIONS.append(tab("表 5　两种容量学生在 DHC-6 上的微调前后（中位 RMSE，deg/s）",
                    ["学生", "参数量", "适配前", "适配后", "留出前", "留出后"],
                    [["tiny", "3,856", "4.127", "1.574", "2.778", "2.624"],
                     ["small", "75,040", "2.913", "0.595", "2.662", "2.148"]]))
SECTIONS.append(fig("dhc6_transfer.svg",
                    "图 14　DHC-6 仿真 ARX 对象上的微调前后对照（15 deg/s 阶跃）；虚线区分适配与留出状态。",
                    "w88", "_report_review/results_summary.json；ARX 模型见 dhc6_arx_all.json"))
SECTIONS.append(fig("dhc6_states.svg",
                    "图 15　代表性 DHC-6 仿真状态的微调前后闭环响应。适配状态微调后贴合参考；留出状态保持原策略的响应形态。",
                    "w100", "_report_review/remote/traces_multi.json"))

# 6 ------------------------------------------------------------------------
SECTIONS.append("""
<h2 id="s6"><span class="num">6</span>讨论</h2>
<p>源域结果显示，49 个逐对象教师中多数达到项目门限；然而统一学生的 RMSE 中位数仍高于对应教师，且 4 个留出对象中仅 3 个通过。120 对象扩展抽样的通过率为 78.3%，幅值扫描则显示，±31 deg/s 阶跃通过率从同分布样本的 60.3% 降至 OOD 样本的 27.6%。在固定增益 PID 的对照中，学生策略在分布外对象上的退化幅度显著更小，但在同分布小幅值段的精度不及经过前馈整定的 PID。这些结果共同表明，蒸馏能够将策略压缩至数千参数并保留大部分常规响应能力，但学生策略对对象参数和指令幅值仍敏感。</p>
<p>DHC-6 微调降低了 8 个适配状态和 4 个留出状态的 RMSE，但适配组改善大于留出组。两种容量的对比还说明，源域蒸馏对容量不敏感（1.28× 与 1.34×），但同一配方迁移到真实对象后，3,856 参数学生的适配后误差是 75,040 参数学生的 2.7 倍，且部分状态因更新顺序出现回退。这说明少量目标域数据可以在当前模型与测试条件下更新策略，。目标域对象由 DHC-6 仿真记录辨识得到的 ARX 模型表示；当前测试也未覆盖横航向耦合、状态估计误差和完整控制系统约束。</p>
<p>本文的飞行品质作用体现在源域对象参数化、等级覆盖和训练后门限评价。归档 PPO 奖励仍由跟踪误差、作用力能量和动作幅值构成，未逐项引入 GJB 条款指标。因此，当前证据支持的是“在飞行品质参数化对象上训练滚转控制器”，而不是“用 GJB 品质指标直接训练控制器”。若要验证后一目标，需将可计算的品质指标加入奖励或约束，并重新训练教师、蒸馏学生和实施目标域迁移；该扩展尚未完成。</p>
""")

SECTIONS.append(f"""
<h3 id="s61">6.1 延迟表征实验：稀疏自编码器能否提取 τ<sub>p</sub></h3>
<p>策略输入采用 32×5 历史，前提是历史中确实带有对象时延的信息。本节在固定 tiny 学生与 510 个分层对象上回答两个问题：
τ<sub>p</sub> 能否从历史中读出，以及 16 维紧凑潜码能否替代原始历史作为输入。</p>

<h4 id="s611">6.1.1 数据、表征与探针</h4>
<p>从 IV-A 对象库的 train_core、validation、id_test 三个互不重叠的划分中，按 τ<sub>p</sub> 分三层各抽 100、30、40 架，共 510 架；每架运行 8 条指令（±1、±2、±5、±10、±21、±31 deg/s 阶跃、双脉冲和多正弦），由 tiny 学生闭环驱动，记录 32×5 历史、当前量、旧动作和动作，共 489,600 个窗口。留出集为 120 架、960 个“对象—指令”episode。</p>
<p>这里的 τ<sub>p</sub> 是对象输入前的<strong>纯运输延迟</strong>（代码级 FIFO，模拟 e<sup>−τ<sub>p</sub>s</sup>），用于构造延迟分层压力测试；它<strong>不是</strong> GJB 2874-1997 表 A21 的拟合等效延迟 τ<sub>e</sub>，不能按 0.10/0.20/0.25 s 给对象打 GJB 延迟等级。两者的完整边界、4 阶 LOES 结构和为什么训练只能用 τ<sub>p</sub>，见 6.1.4。</p>
<p>SAE 输入为 160 维展平历史，编码器为 160→64→16，分别采用 L1 稀疏或 top-k 稀疏，解码器对称；训练后冻结，只训练延迟读出或策略头。对照表征包括原始历史 160 维、PCA-16、稠密 AE-16、36 维请求力—响应互相关，以及 5 维通道均值。延迟探针使用因果前缀：从 1.2 s 起，对截至当前时刻的窗口运行平均后做线性回归，预测 τ<sub>p</sub>；训练、验证、测试按对象划分，避免同一架飞机的窗口泄漏。</p>
""")
SECTIONS.append(fig("delay_network.svg",
                    "图 16　延迟表征实验的两条前端：部署学生使用原始 32×5 历史（上）；实验分支将历史送入冻结 SAE，得到 16 维稀疏潜码，再分别接策略头或线性延迟探针（下）。SAE 只做表征，策略/探针在冻结潜码上训练；对照包括 PCA、稠密 AE、原始历史和互相关。",
                    "w100", "dhc6_blog_assets/make_delay_network.py"))
SECTIONS.append(f"""
<h4 id="s612">6.1.2 延迟可解码性</h4>
<p>表 6 和图 17 汇总 1.2 s 与 2.0 s 因果前缀的留出结果。作为背景，tiny 学生在短、中、长延迟层的闭环通过率分别为 {DELAY_BASE['short']['pass_rate']*100:.1f}%、{DELAY_BASE['medium']['pass_rate']*100:.1f}% 和 {DELAY_BASE['long']['pass_rate']*100:.1f}%，中位 RMSE 分别为 {DELAY_BASE['short']['rmse_median']:.3f}、{DELAY_BASE['medium']['rmse_median']:.3f} 和 {DELAY_BASE['long']['rmse_median']:.3f} deg/s；长延迟确实更难。</p>
""")
SECTIONS.append(tab("表 6　从因果历史前缀估计 τ<sub>p</sub> 的留出结果（120 架，960 episode）",
                    ["表征", "维度", "1.2 s R²", "1.2 s MAE (ms)", "2.0 s R²", "2.0 s MAE (ms)"],
                    [["互相关（F<sub>req</sub>—p）", "36",
                      f"{dv('handcrafted_xcorr', '1.2s', 'test_r2'):.3f}",
                      f"{dv('handcrafted_xcorr', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('handcrafted_xcorr', '2.0s', 'test_r2'):.3f}",
                      f"{dv('handcrafted_xcorr', '2.0s', 'test_mae_ms'):.1f}"],
                     ["稠密 AE", "16",
                      f"{dv('dense16', '1.2s', 'test_r2'):.3f}",
                      f"{dv('dense16', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('dense16', '2.0s', 'test_r2'):.3f}",
                      f"{dv('dense16', '2.0s', 'test_mae_ms'):.1f}"],
                     ["原始历史", "160",
                      f"{dv('raw_history', '1.2s', 'test_r2'):.3f}",
                      f"{dv('raw_history', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('raw_history', '2.0s', 'test_r2'):.3f}",
                      f"{dv('raw_history', '2.0s', 'test_mae_ms'):.1f}"],
                     ["PCA", "16",
                      f"{dv('pca16', '1.2s', 'test_r2'):.3f}",
                      f"{dv('pca16', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('pca16', '2.0s', 'test_r2'):.3f}",
                      f"{dv('pca16', '2.0s', 'test_mae_ms'):.1f}"],
                     ["通道均值", "5",
                      f"{dv('mean5', '1.2s', 'test_r2'):.3f}",
                      f"{dv('mean5', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('mean5', '2.0s', 'test_r2'):.3f}",
                      f"{dv('mean5', '2.0s', 'test_mae_ms'):.1f}"],
                     ["L1-SAE（L0≈7）", "16",
                      f"{dv('l1_3e-03', '1.2s', 'test_r2'):.3f}",
                      f"{dv('l1_3e-03', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('l1_3e-03', '2.0s', 'test_r2'):.3f}",
                      f"{dv('l1_3e-03', '2.0s', 'test_mae_ms'):.1f}"],
                     ["Top-k SAE（k=4）", "16",
                      f"{dv('topk4', '1.2s', 'test_r2'):.3f}",
                      f"{dv('topk4', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('topk4', '2.0s', 'test_r2'):.3f}",
                      f"{dv('topk4', '2.0s', 'test_mae_ms'):.1f}"],
                     ["L1-SAE（L0≈3.5）", "16",
                      f"{dv('l1_3e-02', '1.2s', 'test_r2'):.3f}",
                      f"{dv('l1_3e-02', '1.2s', 'test_mae_ms'):.1f}",
                      f"{dv('l1_3e-02', '2.0s', 'test_r2'):.3f}",
                      f"{dv('l1_3e-02', '2.0s', 'test_mae_ms'):.1f}"]]))
SECTIONS.append(fig("delay_probe.svg",
                    "图 17　延迟表征的可解码性。(a) 1.2 s 与 2.0 s 因果前缀的测试 R²；(b) 对应 MAE；(c) R² 随观测时间的变化；(d) 稀疏度—延迟信息权衡。留出 120 架对象、960 个对象—指令 episode。",
                    "w100", "dhc6_blog_assets/make_delay_figures.py"))
SECTIONS.append(f"""
<p>结果可归纳为三点。其一，<strong>延迟是可解码的</strong>：36 维互相关在 1.2 s 即达到 R² = {dv('handcrafted_xcorr', '1.2s', 'test_r2'):.3f}、MAE = {dv('handcrafted_xcorr', '1.2s', 'test_mae_ms'):.1f} ms；稠密 AE、原始历史和 PCA 也在约 0.50–0.61 之间。其二，<strong>稀疏有明显代价</strong>：16 维稠密 AE 为 {dv('dense16', '1.2s', 'test_r2'):.3f}，而 L0≈7 的 L1-SAE 为 {dv('l1_3e-03', '1.2s', 'test_r2'):.3f}，top-k4 仅 {dv('topk4', '1.2s', 'test_r2'):.3f}；L0≈3.5 时进一步降到 {dv('l1_3e-02', '1.2s', 'test_r2'):.3f}。其三，<strong>简单时移特征优于学习表征</strong>：延迟信息主要存在于请求力与响应之间的时移，而不是绝对幅值或低维流形；互相关的 MAE 比最好的 16 维学习表征低约一个数量级。</p>
<p>由此得到的输入选型结论是：延迟信息在历史中可被线性读出，稠密 16 维编码保留该信息（PCA 与稠密 AE 的 R² 约 0.50–0.61），
稀疏约束使其丢失（L0≈7 为 0.344，top-k4 为 0.122）。16 维本身不构成瓶颈，稀疏性才是；因此策略输入保留原始历史，
若需压缩则用稠密编码，不用稀疏潜码。</p>
<p>本节在 510 架对象、16 维瓶颈、线性读出与 0–0.2 s 时延范围内评估。互相关探针用于判断历史中是否含有延迟信息，
它本身不是可部署控制器。</p>
""")

SECTIONS.append(f"""
<h4 id="s613">6.1.3 闭环对照与跨时延迁移</h4>
<p>闭环对照在同一 510 架对象划分上进行。策略变体包括：原始历史（tiny 微调）、PCA-16、稠密 AE-16、Top-k SAE-16，以及把 SAE 潜码作为辅助输入接回 tiny 的 SAE-aux。为保证比较公平，除 raw 基线外，各变体在相同 BC 数据上预训练，并用同一 DAgger 预算（96 架源域对象、β = 0.5、8 轮）补充学生自身访问状态；评价仍按短、中、长延迟分层。跨时延列另训一组仅见短/中延迟（τ<sub>p</sub> ≤ 0.15 s）的模型，在长延迟留出对象上评价。</p>
""")
SECTIONS.append(tab("表 7　闭环对照：按延迟分层的通过率与跨时延迁移",
                    ["变体", "短延迟", "中延迟", "长延迟", "长延迟 RMSE (deg/s)", "仅源域训练 → 长延迟"],
                    [["原始历史（tiny 微调）",
                      f"{dp('raw_ft', 'short', 'pass_rate')*100:.1f}%",
                      f"{dp('raw_ft', 'medium', 'pass_rate')*100:.1f}%",
                      f"{dp('raw_ft', 'long', 'pass_rate')*100:.1f}%",
                      f"{dp('raw_ft', 'long', 'rmse_median'):.3f}",
                      f"{dc_text('raw_ft')}%"],
                     ["PCA-16",
                      f"{dp('pca16', 'short', 'pass_rate')*100:.1f}%",
                      f"{dp('pca16', 'medium', 'pass_rate')*100:.1f}%",
                      f"{dp('pca16', 'long', 'pass_rate')*100:.1f}%",
                      f"{dp('pca16', 'long', 'rmse_median'):.3f}",
                      f"{dc_text('pca16')}%"],
                     ["稠密 AE-16",
                      f"{dp('dense16', 'short', 'pass_rate')*100:.1f}%",
                      f"{dp('dense16', 'medium', 'pass_rate')*100:.1f}%",
                      f"{dp('dense16', 'long', 'pass_rate')*100:.1f}%",
                      f"{dp('dense16', 'long', 'rmse_median'):.3f}",
                      "—"],
                     ["Top-k SAE-16",
                      f"{dp('topk4', 'short', 'pass_rate')*100:.1f}%",
                      f"{dp('topk4', 'medium', 'pass_rate')*100:.1f}%",
                      f"{dp('topk4', 'long', 'pass_rate')*100:.1f}%",
                      f"{dp('topk4', 'long', 'rmse_median'):.3f}",
                      f"{dc_text('topk4')}%"],
                     ["SAE-aux（保留原始历史）",
                      f"{dp('aux_topk4', 'short', 'pass_rate')*100:.1f}%",
                      f"{dp('aux_topk4', 'medium', 'pass_rate')*100:.1f}%",
                      f"{dp('aux_topk4', 'long', 'pass_rate')*100:.1f}%",
                      f"{dp('aux_topk4', 'long', 'rmse_median'):.3f}",
                      f"{dc_text('aux_topk4')}%"]]))
SECTIONS.append(fig("delay_closed_loop.svg",
                    "图 18　闭环对照。(a) 各策略在短/中/长延迟层的通过率（120 架留出，960 episode）；(b) 仅用短/中延迟训练后，长延迟留出的通过率与全延迟训练的对比。raw 与 SAE-aux 保持 90%，PCA 与 Top-k SAE 明显下降。",
                    "w100", "dhc6_blog_assets/make_delay_figures.py"))
SECTIONS.append(f"""
<p>表 7 与图 18 给出闭环结果。raw_ft 与 tiny 基线基本一致（短 {dp('raw_ft', 'short', 'pass_rate')*100:.1f}%、中 {dp('raw_ft', 'medium', 'pass_rate')*100:.1f}%、长 {dp('raw_ft', 'long', 'pass_rate')*100:.1f}%），说明当前 BC/DAgger 流程没有破坏基线。所有 16 维替换类都低于 raw：PCA-16 为 {dp('pca16', 'short', 'pass_rate')*100:.1f}/{dp('pca16', 'medium', 'pass_rate')*100:.1f}/{dp('pca16', 'long', 'pass_rate')*100:.1f}%，稠密 AE-16 为 {dp('dense16', 'short', 'pass_rate')*100:.1f}/{dp('dense16', 'medium', 'pass_rate')*100:.1f}/{dp('dense16', 'long', 'pass_rate')*100:.1f}%，Top-k SAE-16 为 {dp('topk4', 'short', 'pass_rate')*100:.1f}/{dp('topk4', 'medium', 'pass_rate')*100:.1f}/{dp('topk4', 'long', 'pass_rate')*100:.1f}%。SAE-aux 保持 {dp('aux_topk4', 'short', 'pass_rate')*100:.1f}/{dp('aux_topk4', 'medium', 'pass_rate')*100:.1f}/{dp('aux_topk4', 'long', 'pass_rate')*100:.1f}%，与 raw 持平。</p>
<p>跨时延列给出更直接的检验：仅用短/中延迟训练后，raw 与 SAE-aux 仍保持 {dc_text('raw_ft')}% 和 {dc_text('aux_topk4')}%，PCA 降到 {dc_text('pca16')}%，Top-k SAE 从 {dp('topk4', 'long', 'pass_rate')*100:.1f}% 降到 {dc_text('topk4')}%。Top-k SAE 在同分布替换类中最好，但其优势没有转化为跨时延迁移；相反，它对训练时见过的延迟范围更敏感。SAE-aux 与 raw 持平（90.0% 对 90.0%），定位为安全选项：保留原始历史、附加潜码不改变主策略行为。</p>
<p>PCA 与稠密 AE 各重复一次，长延迟通过率分别落在 51.6–55.6% 与 52.5–63.8%；这说明替换类的具体排序对种子敏感，但“明显低于 raw”这一定性结论在两次运行中一致。raw 与 SAE-aux 的 90% 在重复运行中保持稳定。</p>
""")

SECTIONS.append(f"""
<h4 id="s614">6.1.4 τ<sub>p</sub> 与 GJB 等效时间延迟 τ<sub>e</sub> 的职责边界</h4>
<p>GJB 2874-1997 A5.4.1.1.5（印刷 246–247，PDF 250–251）对滚转轴给出<strong>等效时间延迟 τ<sub>e</sub></strong> 的建议上限：1 级 0.10 s、2 级 0.20 s、3 级 0.25 s（表 A21）。同页说明，该要求针对前置滤波器、SAS、舵机等<strong>综合延迟作用</strong>；A4.9（印刷 68，PDF 72）进一步指出，经典传递函数通常要加入时间延迟项，以改善高频拟合，并与驾驶员评价相关。</p>
<p>滚转轴低阶等效系统（LOES）是<strong>4 阶</strong>，不是 2 阶或 3 阶。GJB A5.4.1.1.1（印刷 210–211，PDF 214–215）的滚转速率传递函数包含滚转、螺旋和荷兰滚三个模态；印刷 212 页明确说滚转速率响应通常由这三个模态组成。其分母为 (s−λ<sub>s</sub>)(s+1/T<sub>R</sub>)(s²+2ζ<sub>d</sub>ω<sub>d</sub>s+ω<sub>d</sub>²)，共 4 阶；分子为 s(s²+2ζ<sub>φ</sub>ω<sub>φ</sub>s+ω<sub>φ</sub>²)，共 3 阶；再乘 e<sup>−τ<sub>e</sub>s</sup>。τ<sub>e</sub> 是这个 4 阶拟合的延迟项，不是一只独立的硬件计时器。</p>
<p>因此 τ<sub>e</sub> 不是对象参数，而是<strong>闭环拟合量</strong>。它依赖控制器、采样保持、作动器、滤波器、增稳系统和拟合协议；同一对象换控制器，τ<sub>e</sub> 会变；训练前也无法知道。若把 τ<sub>e</sub> 作为 plant 采样参数，会形成循环依赖，并使训练/测试划分随策略变化。</p>
<p>本文的 τ<sub>p</sub> 是对象输入前的纯运输 FIFO 延迟（e<sup>−τ<sub>p</sub>s</sup>），可在训练前采样，用于生成对象族、训练和延迟分层。它不是 GJB 的 τ<sub>e</sub>；表 A21 的 0.10/0.20/0.25 s 不能用于给 τ<sub>p</sub> 打等级。要评价 τ<sub>e</sub>，需要另建完整闭环 + 4 阶 LOES 拟合 + 残差检查；本文在 6.1.5 给出首版评价，但受单种子、单幅值和项目门限限制，仍不构成 A21 符合性结论。</p>
""")
SECTIONS.append(tab("表 9　τ<sub>p</sub> 与 GJB τ<sub>e</sub> 的职责边界",
                    ["项目", "τ<sub>p</sub>（本文对象参数）", "τ<sub>e</sub>（GJB A21 评价量）"],
                    [["数学位置", "对象输入前 e<sup>−τ<sub>p</sub>s</sup>", "闭环 LOES 拟合项 e<sup>−τ<sub>e</sub>s</sup>"],
                     ["已知时机", "训练前采样", "控制器确定后拟合"],
                     ["依赖控制器", "否", "是"],
                     ["主要用途", "对象生成、训练、延迟分层", "飞行品质评价"],
                     ["范围 / 门限", "0.001–0.20 s 常态；0.5 s OOD", "1 级 ≤0.10 s；2 级 ≤0.20 s；3 级 ≤0.25 s"],
                     ["A21 可评性", "不可直接评", "需拟合残差合格"]]))
SECTIONS.append(f"""
<p>这一边界也解释了为什么 6.1.2 的探针目标是 τ<sub>p</sub> 而不是 τ<sub>e</sub>：τ<sub>p</sub> 是每架飞机在训练前就确定的静态标签，适合做延迟分层和可解码性检验；τ<sub>e</sub> 随策略变化，不能作为固定的 plant 标签。若后续要回答“当前策略的 GJB 等效延迟是多少”，应在完整闭环上单独做 τ<sub>e</sub> 拟合，而不是把本节结果改写成 A21 等级。</p>
""")

SECTIONS.append(f"""
<h4 id="s615">6.1.5 GJB τ<sub>e</sub> 首版评价（120 架 × 3 策略）</h4>
<p>按 6.1.4 的边界，本文另建了一个 τ<sub>e</sub> 评价原型：在 120 架留出对象（短/中/长各 40）上，对原始历史、Top-k SAE-16 和 SAE-aux 三个策略，注入 0.1–10 rad/s 的 18 分量多正弦滚转指令，记录 180 s 闭环响应；在 50 Hz 采样上估计 H(jω) = P/P<sub>c</sub>，再拟合 4 阶滚转 LOES（滚转 + 螺旋 + 荷兰滚）和 τ<sub>e</sub>。二阶参考模型保留在回路内，视为飞控系统的一部分。</p>
<p>拟合门限为项目设定，不是 GJB 规定：幅值 VAF ≥ 0.95 且相位 RMSE ≤ 10°；不过门记 <code>not_assessable</code>，不赋 A21 等级。表 10 和图 19 给出结果。</p>
""")
SECTIONS.append(tab("表 10　GJB τ<sub>e</sub> 首版评价（120 架，4 阶 LOES）",
                    ["策略", "可评 / 120", "τ<sub>e</sub> 中位 (s)", "短 / 中 / 长中位 (s)",
                     "1 级", "2 级", "3 级", ">3 级"],
                    [["原始历史",
                      f"{TAU_E['raw_ft']['assessable']}/120",
                      f"{TAU_E['raw_ft']['tau_e_median']:.3f}",
                      " / ".join(f"{TAU_E['raw_ft']['by_bin'][b]['tau_e_median']:.3f}"
                                 for b in ("short", "medium", "long")),
                      str(TAU_E['raw_ft']['levels']['level1']),
                      str(TAU_E['raw_ft']['levels']['level2']),
                      str(TAU_E['raw_ft']['levels']['level3']),
                      str(TAU_E['raw_ft']['levels']['exceeds_level3'])],
                     ["SAE-aux",
                      f"{TAU_E['aux_topk4']['assessable']}/120",
                      f"{TAU_E['aux_topk4']['tau_e_median']:.3f}",
                      " / ".join(f"{TAU_E['aux_topk4']['by_bin'][b]['tau_e_median']:.3f}"
                                 for b in ("short", "medium", "long")),
                      str(TAU_E['aux_topk4']['levels']['level1']),
                      str(TAU_E['aux_topk4']['levels']['level2']),
                      str(TAU_E['aux_topk4']['levels']['level3']),
                      str(TAU_E['aux_topk4']['levels']['exceeds_level3'])],
                     ["Top-k SAE-16",
                      f"{TAU_E['topk4']['assessable']}/120", "—", "— / — / —",
                      "0", "0", "0", "0"]]))
SECTIONS.append(fig("tau_e_eval.svg",
                    "图 19　GJB τe 首版评价。(a) 可评对象的 τe 与 τp；(b) A21 等级分布（未过拟合门的对象单列）；(c) 一个原始历史对象的 Bode 拟合示例。Top-k SAE-16 的 120 架全部未过拟合门，因此没有 A21 等级。",
                    "w100", "dhc6_blog_assets/make_tau_e_figure.py"))
SECTIONS.append(f"""
<p>结果有三点。其一，<strong>τ<sub>e</sub> 随 τ<sub>p</sub> 增大</strong>：原始历史在短、中、长延迟的可评中位 τ<sub>e</sub> 分别为 {TAU_E['raw_ft']['by_bin']['short']['tau_e_median']:.3f}、{TAU_E['raw_ft']['by_bin']['medium']['tau_e_median']:.3f}、{TAU_E['raw_ft']['by_bin']['long']['tau_e_median']:.3f} s；SAE-aux 为 {TAU_E['aux_topk4']['by_bin']['short']['tau_e_median']:.3f}、{TAU_E['aux_topk4']['by_bin']['medium']['tau_e_median']:.3f}、{TAU_E['aux_topk4']['by_bin']['long']['tau_e_median']:.3f} s。长延迟对象的 τ<sub>e</sub> 中位约 0.30 s，已超过表 A21 的 3 级上限 0.25 s。</p>
<p>其二，<strong>可评比例本身是结果</strong>：120 架中，原始历史 {TAU_E['raw_ft']['assessable']} 架、SAE-aux {TAU_E['aux_topk4']['assessable']} 架通过 4 阶 LOES 拟合门；Top-k SAE-16 为 {TAU_E['topk4']['assessable']} 架。Top-k SAE 的闭环频响在低频段散乱并伴随持续振荡，VAF 中位仅 {TAU_E['topk4']['vaf_median']:.3f}、相位 RMSE 中位 {TAU_E['topk4']['phase_rmse_median']:.1f}°，因此只能记 <code>not_assessable</code>，不能把 τ<sub>e</sub> 填成 0。</p>
<p>其三，<strong>可评集合内的等级分布并不乐观</strong>：原始历史 73 架可评对象中，1 级 {TAU_E['raw_ft']['levels']['level1']} 架、2 级 {TAU_E['raw_ft']['levels']['level2']} 架、3 级 {TAU_E['raw_ft']['levels']['level3']} 架、超过 3 级 {TAU_E['raw_ft']['levels']['exceeds_level3']} 架；SAE-aux 102 架可评对象中分别为 {TAU_E['aux_topk4']['levels']['level1']}、{TAU_E['aux_topk4']['levels']['level2']}、{TAU_E['aux_topk4']['levels']['level3']}、{TAU_E['aux_topk4']['levels']['exceeds_level3']} 架。SAE-aux 的可评比例更高，但在可评集合内 τ<sub>e</sub> 中位反而更高；这是选择效应——raw 的许多长延迟难例因残差不过门被排除，两个集合不能直接比较。</p>
<p>局限：主表为种子 1、每分量 1 deg/s、180 s 记录；6.1.6 补了三档幅值与第二种子，但仍不是多源/多机型统计。拟合门限（VAF 0.95、相位 10°）是项目设定，不是 GJB 规定；参考模型被计入飞控系统。因此表 10 只说明当前策略在“可拟合对象”上的 τ<sub>e</sub> 量级，不构成 A21 符合性结论；Top-k SAE 的 <code>not_assessable</code> 也不能解释为“延迟更小”。</p>
""")

SECTIONS.append(f"""
<h4 id="s616">6.1.6 τ<sub>e</sub> 评价的稳健性：激励幅值与策略种子</h4>
<p>为检查 τ<sub>e</sub> 是否只是某一激励幅值下的拟合产物，在同一 120 架 × 3 策略上重复了 0.5、1.0、2.0 deg/s（每分量）三档多正弦激励，拟合协议与门限保持不变。表 11 和图 20 给出结果。</p>
""")
SECTIONS.append(tab("表 11　激励幅值对 τ<sub>e</sub> 评价的影响（120 架 × 3 策略）",
                    ["策略", "可评 / 120（0.5 / 1.0 / 2.0）",
                     "τ<sub>e</sub> 中位（0.5 / 1.0 / 2.0）",
                     "共同可评 n", "共同集 τ<sub>e</sub> 范围 (s)"],
                    [["原始历史",
                      " / ".join(str(AMP['raw_ft']['by_amplitude'][a]['assessable'])
                                 for a in ("0.5", "1.0", "2.0")),
                      " / ".join(f"{AMP['raw_ft']['by_amplitude'][a]['tau_e_median']:.3f}"
                                 for a in ("0.5", "1.0", "2.0")),
                      str(AMP['raw_ft']['common_n']),
                      f"{min(AMP['raw_ft']['common_tau_e_median'].values()):.3f}–"
                      f"{max(AMP['raw_ft']['common_tau_e_median'].values()):.3f}"],
                     ["SAE-aux",
                      " / ".join(str(AMP['aux_topk4']['by_amplitude'][a]['assessable'])
                                 for a in ("0.5", "1.0", "2.0")),
                      " / ".join(f"{AMP['aux_topk4']['by_amplitude'][a]['tau_e_median']:.3f}"
                                 for a in ("0.5", "1.0", "2.0")),
                      str(AMP['aux_topk4']['common_n']),
                      f"{min(AMP['aux_topk4']['common_tau_e_median'].values()):.3f}–"
                      f"{max(AMP['aux_topk4']['common_tau_e_median'].values()):.3f}"],
                     ["Top-k SAE-16", "0 / 0 / 0", "— / — / —", "0", "—"]]))
SECTIONS.append(fig("tau_e_amplitude.svg",
                    "图 20　τe 评价的激励幅值稳健性。(a) 三种幅值共同可评对象上的 τe 中位；(b) 可评对象数随幅值变化；(c) Top-k SAE 在三种幅值下的 VAF 与相位残差。",
                    "w100", "dhc6_blog_assets/make_tau_e_amplitude_figure.py"))
SECTIONS.append(f"""
<p>幅值主要影响<strong>可评比例</strong>，而不是 τ<sub>e</sub> 本身：原始历史可评数从 41 升到 73、再到 91，SAE-aux 从 78 升到 102、再到 111；这是因为大幅值提高了信噪比，使 4 阶 LOES 更容易过相位门。但在 40 架（raw）和 75 架（aux）<strong>三种幅值都可评</strong>的共同对象上，τ<sub>e</sub> 中位基本稳定：raw 为 {min(AMP['raw_ft']['common_tau_e_median'].values()):.3f}–{max(AMP['raw_ft']['common_tau_e_median'].values()):.3f} s，aux 为 {min(AMP['aux_topk4']['common_tau_e_median'].values()):.3f}–{max(AMP['aux_topk4']['common_tau_e_median'].values()):.3f} s，跨度约 0.01 s。因此 6.1.5 的 τ<sub>e</sub> 排序不是单一幅值造成的假象。</p>
<p>Top-k SAE-16 在三种幅值下均为 0/120：VAF 中位分别为 {AMP['topk4']['by_amplitude']['0.5']['vaf_median']:.3f}、{AMP['topk4']['by_amplitude']['1.0']['vaf_median']:.3f}、{AMP['topk4']['by_amplitude']['2.0']['vaf_median']:.3f}，相位 RMSE 中位分别为 {AMP['topk4']['by_amplitude']['0.5']['phase_median']:.1f}°、{AMP['topk4']['by_amplitude']['1.0']['phase_median']:.1f}°、{AMP['topk4']['by_amplitude']['2.0']['phase_median']:.1f}°。这说明它的 not_assessable 不是幅值选错，而是闭环响应本身不能被 4 阶 LOES 表示（或长时小信号下持续振荡）。</p>
<p>拟合门限本身也做了敏感性检查：相位门取 8°、10°、12° 时，原始历史可评数为 {GATE['raw_ft']['phase8']['n']}/{GATE['raw_ft']['phase10']['n']}/{GATE['raw_ft']['phase12']['n']}，τ<sub>e</sub> 中位为 {GATE['raw_ft']['phase8']['tau_median']:.3f}/{GATE['raw_ft']['phase10']['tau_median']:.3f}/{GATE['raw_ft']['phase12']['tau_median']:.3f} s；SAE-aux 为 {GATE['aux_topk4']['phase8']['n']}/{GATE['aux_topk4']['phase10']['n']}/{GATE['aux_topk4']['phase12']['n']} 和 {GATE['aux_topk4']['phase8']['tau_median']:.3f}/{GATE['aux_topk4']['phase10']['tau_median']:.3f}/{GATE['aux_topk4']['phase12']['tau_median']:.3f} s；Top-k SAE 始终为 0。可见可评比例依赖门限，但 raw/aux 的 τ<sub>e</sub> 量级和 Top-k SAE 的失败对门限不敏感。</p>
<p>为检查策略训练随机性，raw、Top-k SAE 和 SAE-aux 各用第二种子重训（相同 BC/DAgger 预算），再做同一 τ<sub>e</sub> 评价。表 12 和图 21 给出结果。</p>
""")
SECTIONS.append(tab("表 12　策略种子对 τ<sub>e</sub> 评价的影响（120 架）",
                    ["策略", "可评（s1 / s2）", "τ<sub>e</sub> 中位（s1 / s2）",
                     "共同 n", "共同中位（s1 / s2）", "1/2/3/>3 级（s1 → s2）"],
                    [["原始历史",
                      f"{SEED['raw_ft']['seed1']['assessable']} / {SEED['raw_ft']['seed2']['assessable']}",
                      f"{SEED['raw_ft']['seed1']['tau_e_median']:.3f} / {SEED['raw_ft']['seed2']['tau_e_median']:.3f}",
                      str(SEED['raw_ft']['common_n']),
                      f"{SEED['raw_ft']['common_median']['seed1']:.3f} / {SEED['raw_ft']['common_median']['seed2']:.3f}",
                      f"{lv_text('raw_ft', 'seed1')} → {lv_text('raw_ft', 'seed2')}"],
                     ["SAE-aux",
                      f"{SEED['aux_topk4']['seed1']['assessable']} / {SEED['aux_topk4']['seed2']['assessable']}",
                      f"{SEED['aux_topk4']['seed1']['tau_e_median']:.3f} / {SEED['aux_topk4']['seed2']['tau_e_median']:.3f}",
                      str(SEED['aux_topk4']['common_n']),
                      f"{SEED['aux_topk4']['common_median']['seed1']:.3f} / {SEED['aux_topk4']['common_median']['seed2']:.3f}",
                      f"{lv_text('aux_topk4', 'seed1')} → {lv_text('aux_topk4', 'seed2')}"],
                     ["Top-k SAE-16", "0 / 0", "— / —", "0", "—", "0/0/0/0 → 0/0/0/0"]]))
SECTIONS.append(fig("tau_e_seed.svg",
                    "图 21　τe 评价的策略种子稳健性。(a) 两个种子共同可评对象的逐对象 τe；(b) 两个种子的 A21 等级分布。",
                    "w100", "dhc6_blog_assets/make_tau_e_seed_figure.py"))
SECTIONS.append(f"""
<p>两个种子的结果基本一致。原始历史可评数 {SEED['raw_ft']['seed1']['assessable']} → {SEED['raw_ft']['seed2']['assessable']}，τ<sub>e</sub> 中位 {SEED['raw_ft']['seed1']['tau_e_median']:.3f} → {SEED['raw_ft']['seed2']['tau_e_median']:.3f} s，共同可评集 73 架的中位 {SEED['raw_ft']['common_median']['seed1']:.3f} → {SEED['raw_ft']['common_median']['seed2']:.3f} s，等级分布 {lv_text('raw_ft', 'seed1')} → {lv_text('raw_ft', 'seed2')}。SAE-aux 可评数 {SEED['aux_topk4']['seed1']['assessable']} → {SEED['aux_topk4']['seed2']['assessable']}，中位 {SEED['aux_topk4']['seed1']['tau_e_median']:.3f} → {SEED['aux_topk4']['seed2']['tau_e_median']:.3f} s，共同 101 架中位 {SEED['aux_topk4']['common_median']['seed1']:.3f} → {SEED['aux_topk4']['common_median']['seed2']:.3f} s，等级分布 {lv_text('aux_topk4', 'seed1')} → {lv_text('aux_topk4', 'seed2')}。</p>
<p>Top-k SAE 两个种子均为 0/120 可评，说明它的 <code>not_assessable</code> 不是某一策略种子的偶然。由此可以认为：6.1.5 的 τ<sub>e</sub> 量级、延迟分层的单调趋势，以及 Top-k SAE 的拟合失败，在“两个种子 × 三档幅值”的范围内是稳健的；但这不是多源、多机型或 GJB 审定意义上的统计置信度。</p>
""")
SECTIONS.append(fig("topk4_diagnostic.svg",
                    "图 22　Top-k SAE 与原始历史的 180 s 闭环响应（id_test-3458，每分量 1 deg/s）。(a) 时域响应；(b) 功率谱。Top-k SAE 在 0–3 Hz 的低频段功率明显更高，说明其闭环响应不是单纯的延迟增大，而是伴随持续低频振荡/非线性，导致 4 阶 LOES 拟合失败。",
                    "w100", "pingmo/experiments/gjb_tau_e_20261023/diagnose_topk4.py"))

SECTIONS.append(f"""
<h3 id="s62">6.2 后续实验计划：执行机构敏感性</h3>
<p><strong>以下实验尚未开展，不属于本文结果。</strong>为检验当前执行机构设置是否影响策略表现，先在固定对象、随机种子和指令集上对现有策略做冻结权重敏感性扫描；若结果对参数明显敏感，再按相同训练预算分别重训 PPO，区分“环境变化造成的鲁棒性下降”与“策略能否适应新约束”。每次只改变一个因素，并同步记录教师验收率、RMSE、峰值误差、请求力总变差率、饱和比例，以及 DHC-6 适配组和留出组结果。</p>
""")
SECTIONS.append(tab("表 8　待开展的执行机构消融",
                    ["因素", "当前基线", "计划对照", "目的"],
                    [["请求力限幅", "±22 N", "±16.5 N、±27.5 N", "评估控制权限范围变化的影响；桥接与目标舵面边界同步换算"],
                     ["命令速率限制", "88 N/s", "44 N/s、176 N/s", "评估舵面速率变化对跟踪、平滑性和迁移的影响"],
                     ["执行机构滞后", "一阶时间常数 20 ms", "0 ms、40 ms", "判断当前滞后假设是否改变闭环表现，并检查目标对象中是否重复计入执行机构动态"]]))
SECTIONS.append(f"""
<p>力限幅和速率限幅在当前桥接下分别对应副翼角度和速率边界，消融时需同步更新源域与目标域接口。后续还应扩大未参与教师训练的同分布和分布外测试集，并在相同对象、指令和安全约束下与常规滚转控制律对照。仿真级对照完成后，再逐步引入更高保真模型、滚转—偏航耦合和机载实时性评估。</p>
""")

# 7 ------------------------------------------------------------------------
SECTIONS.append("""
<h2 id="s7"><span class="num">7</span>结论</h2>
<p>本文给出了一条可复用的链路：品质分级采样得到对象族，逐对象教师给出上界，单学生蒸馏把 49 个教师压到 3,856 参数，
目标域微调把通用律接到具体机型上。三个记忆点：其一，压缩不损精度——学生与教师的中位误差比为 1.28 倍，容量放大 20 倍后仍为 1.34 倍；
其二，边界清晰——±1 deg/s 通过 99.7%、±31 deg/s 通过 60.3%，大幅值与分布外参数是主要失分区；
其三，迁移有效——40 步微调把适配状态中位误差降至 0.59 deg/s（−79.6%），留出状态降至 2.15 deg/s。</p>

<p>延迟表征实验补充了另一个边界：32×5 历史的因果前缀可以解码 τ<sub>p</sub>，但 16 维稀疏 SAE 的延迟读出与闭环替换均弱于原始历史；SAE 作为辅助输入可保持基线，却没有可测增益。需要强调，τ<sub>p</sub> 不是 GJB 的 τ<sub>e</sub>；6.1.5 的首版 τ<sub>e</sub> 评价给出了量级和等级分布，6.1.6 又用三档激励幅值和两个策略种子做了稳健性检查，但受项目拟合门限、有限幅值/种子和参考模型假设限制，仍不构成 A21 符合性结论。</p>

<p>下一步是把这条链路推到执行机构与实际接机上：执行机构参数敏感性给出控制权限的适用区间，滚转—偏航耦合给出多轴条件下的边界。</p>
""")

# references --------------------------------------------------------------
REFS = [
    "GJB 185-1986. 有人驾驶飞机（固定翼）飞行品质[S]. 北京: 国防科学技术工业委员会, 1986.",
    "MIL-STD-1797A. Flying qualities of piloted aircraft[S]. Washington: Department of Defense, 1990.",
    "J. Schulman, F. Wolski, P. Dhariwal, et al. Proximal policy optimization algorithms[J]. arXiv:1707.06347, 2017.",
    "S. Ross, G. Gordon, D. Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning[C]. AISTATS, 2011.",
    "V. Vapnik, A. Vashist. A new learning paradigm: learning using privileged information[J]. Neural Networks, 2009, 22(5-6): 544-557.",
    "L. Pinto, M. Andrychowicz, P. Welinder, et al. Asymmetric actor critic for image-based robot learning[C]. Robotics: Science and Systems, 2017.",
    "J. Lee, J. Hwangbo, L. Wellhausen, et al. Learning quadrupedal locomotion over challenging terrain[J]. Science Robotics, 2020, 5(47).",
    "T. Miki, J. Lee, J. Hwangbo, et al. Learning robust perceptive locomotion for quadrupedal robots in the wild[J]. Science Robotics, 2022, 7(62).",
    "A. Kumar, Z. Fu, D. Pathak, et al. RMA: Rapid motor adaptation for legged robots[C]. Robotics: Science and Systems, 2021.",
    "M. O'Connell, G. Shi, X. Shi, et al. Neural-Fly enables rapid learning for agile flight in strong winds[J]. Science Robotics, 2023, 8(78).",
    "E. Kaufmann, L. Bauersfeld, A. Loquercio, et al. Champion-level drone racing using deep reinforcement learning[J]. Nature, 2023, 620: 982-987.",
    "J. Eschmann, D. Albani, G. Loianno. Learning to fly in seconds[J]. IEEE Robotics and Automation Letters, 2024.",
]


def references() -> str:
    items = "\n".join(f"<li>{r}</li>" for r in REFS)
    return f'<h2 id="refs"><span class="num">参考文献</span></h2>\n<ol class="refs">\n{items}\n</ol>\n'


def italicize_math_variables(html: str) -> str:
    """Keep equations roman while italicizing standalone mathematical variables."""
    latin = re.compile(r"(?<![A-Za-z/])([A-Za-z])(?=[^A-Za-z]|$)")
    greek = re.compile(r"([θωζτφλγδÂρπμνεη])")
    protected = [
        (re.compile(r"(?<=\d )N|(?<=\] )N"), "\ue000"),
        (re.compile(r"(?<=\d )s|(?<=2)s$"), "\ue001"),
        (re.compile(r"(?<![A-Za-z])E(?=\[|<sub>)"), "\ue002"),
        (re.compile(r"(?<![A-Za-z])H(?=\()"), "\ue003"),
        (re.compile(r"(?<![A-Za-z])e(?=<sup>)"), "\ue004"),
    ]
    restore = {marker: value for (pattern, marker), value in zip(protected, ("N", "s", "E", "H", "e"))}
    restore["\ue005"] = "d "

    def mark_formula(match: re.Match[str]) -> str:
        parts = re.split(r"(<[^>]+>)", match.group(1))
        for index in range(0, len(parts), 2):
            text = parts[index]
            text = re.sub(r"(?<![A-Za-z])d t(?=[,;\]])", "\ue005t", text)
            for pattern, marker in protected:
                text = pattern.sub(marker, text)
            text = latin.sub(r"<i>\1</i>", text)
            text = greek.sub(r"<i>\1</i>", text)
            for marker, value in restore.items():
                text = text.replace(marker, value)
            parts[index] = text
        return '<div class="eq">' + "".join(parts) + "</div>"

    return re.sub(r'<div class="eq">([\s\S]*?)</div>', mark_formula, html)


TOC = [
    ("摘要", "abstract", 0),
    ("1 引言", "s1", 0),
    ("1.1 相关工作", "s11", 1),
    ("2 问题定义与控制对象", "s2", 0),
    ("2.1 控制任务与接口", "s21", 1),
    ("2.2 参数化滚转对象", "s22", 1),
    ("2.3 评价指标与门限", "s23", 1),
    ("3 方法：四步流程", "s3", 0),
    ("3.1 随机采样", "s31", 1),
    ("3.2 PPO 教师", "s32", 1),
    ("3.3 DAgger 蒸馏", "s33", 1),
    ("3.4 DHC-6 微调", "s34", 1),
    ("4 实验设置", "s4", 0),
    ("5 结果", "s5", 0),
    ("5.1 对象分布", "s51", 1),
    ("5.2 教师结果", "s52", 1),
    ("5.3 学生结果", "s53", 1),
    ("5.3.1 闭环响应", "s531", 2),
    ("5.3.2 扩展抽样", "s532", 2),
    ("5.3.3 幅值与 OOD", "s533", 2),
    ("5.4 DHC-6 迁移", "s54", 1),
    ("6 讨论", "s6", 0),
    ("6.1 延迟表征实验", "s61", 1),
    ("6.1.1 数据与表征", "s611", 2),
    ("6.1.2 可解码性", "s612", 2),
    ("6.1.3 闭环对照", "s613", 2),
    ("6.1.4 τp 与 GJB τe", "s614", 2),
    ("6.1.5 GJB τe 首版评价", "s615", 2),
    ("6.1.6 τe 稳健性", "s616", 2),
    ("6.2 后续：执行机构", "s62", 1),
    ("7 结论", "s7", 0),
    ("参考文献", "refs", 0),
]


def toc_html() -> str:
    links = "\n".join(f'<a class="sub" href="#{anchor}">{label}</a>' if level
                       else f'<a href="#{anchor}">{label}</a>'
                       for label, anchor, level in TOC)
    return f'<aside class="toc"><div class="toc-title">本文目录</div><nav>{links}</nav></aside>'


class _Markdown(HTMLParser):
    """Small converter for the semantic tags used in the generated paper."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self.current: list[str] = []
        self.kind = ""
        self.heading = 0
        self.list_kind = ""
        self.list_index = 0
        self.in_list_item = False
        self.table_rows: list[list[str]] = []
        self.row: list[str] = []
        self.cell: list[str] = []
        self.in_cell = False
        self.in_table = False
        self.in_figure = False
        self.in_caption = False

    def _flush(self) -> None:
        value = "".join(self.current).strip()
        if value:
            self.blocks.append(value)
        self.current = []
        self.kind = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs = dict(attrs)
        if tag in ("h2", "h3", "h4"):
            self._flush()
            self.heading = {"h2": 2, "h3": 3, "h4": 4}[tag]
            self.current = ["#" * self.heading, " "]
            self.kind = "block"
        elif tag == "p":
            self._flush()
            self.kind = "block"
        elif tag == "div" and "eq" in attrs.get("class", "").split():
            self._flush()
            self.current = ["$$ "]
            self.kind = "equation"
        elif tag in ("ul", "ol"):
            self._flush()
            self.list_kind = "ordered" if tag == "ol" else "unordered"
            self.list_index = 0
        elif tag == "li":
            self._flush()
            self.list_index += 1
            prefix = f"{self.list_index}. " if self.list_kind == "ordered" else "- "
            self.current = [prefix]
            self.kind = "list"
            self.in_list_item = True
        elif tag == "figure":
            self._flush()
            self.in_figure = True
        elif tag == "figcaption":
            self.current = ["*" ]
            self.kind = "caption"
            self.in_caption = True
        elif tag == "caption":
            self._flush()
            self.kind = "block"
        elif tag == "sub" and self.kind:
            self.current.append("_")
        elif tag == "sup" and self.kind:
            self.current.append("^{")
        elif tag == "table":
            self._flush()
            self.in_table = True
            self.table_rows = []
        elif tag == "tr" and self.in_table:
            self.row = []
        elif tag in ("td", "th") and self.in_table:
            self.cell = []
            self.in_cell = True
        elif tag == "br" and self.kind and not self.in_cell:
            self.current.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h2", "h3", "h4", "p"):
            self._flush()
        elif tag == "caption":
            self._flush()
        elif tag == "sup" and self.kind:
            self.current.append("}")
        elif tag == "div" and self.kind == "equation":
            self.current.append(" $$")
            self._flush()
        elif tag == "li":
            self._flush()
            self.in_list_item = False
        elif tag in ("ul", "ol"):
            self._flush()
            self.list_kind = ""
        elif tag == "figcaption":
            self.current.append("*")
            self._flush()
            self.in_caption = False
        elif tag == "figure":
            self._flush()
            self.in_figure = False
        elif tag in ("td", "th") and self.in_cell:
            self.row.append("".join(self.cell).strip().replace("|", "\\|"))
            self.cell = []
            self.in_cell = False
        elif tag == "tr" and self.in_table:
            if self.row:
                self.table_rows.append(self.row)
            self.row = []
        elif tag == "table" and self.in_table:
            if self.table_rows:
                widths = max(len(row) for row in self.table_rows)
                rows = [row + [""] * (widths - len(row)) for row in self.table_rows]
                lines = ["| " + " | ".join(rows[0]) + " |",
                         "| " + " | ".join(["---"] * widths) + " |"]
                lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
                self.blocks.append("\n".join(lines))
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.cell.append(data)
        elif self.kind:
            self.current.append(data)

    def render(self, fragment: str) -> str:
        self.feed(fragment)
        self._flush()
        return "\n\n".join(block for block in self.blocks if block.strip())


def html_to_markdown(fragment: str) -> str:
    return _Markdown().render(fragment)


# --------------------------------------------------------------------------- template

CSS = """
:root { --ink:#16233a; --muted:#3a4a63; --rule:#c9d3e2; --link:#245a9e; }
* { box-sizing: border-box; }
body { margin:0; background:#f7f8fb; color:var(--ink);
       font-family: Georgia, "Songti SC", "SimSun", "Noto Serif SC", serif;
       font-size:16.5px; line-height:1.85; }
.paper-shell { display:grid; grid-template-columns:210px minmax(0, 840px); gap:28px;
              max-width:1120px; margin:0 auto; padding:28px 20px 60px; align-items:start; }
main { min-width:0; padding:48px 28px 80px; background:#fff;
       box-shadow: 0 0 0 1px #e6ebf3; }
.toc { position:sticky; top:24px; max-height:calc(100vh - 48px); overflow:auto;
       padding:16px 14px; border:1px solid #e0e7f1;
       border-radius:10px; background:#fff; box-shadow:0 4px 18px #152b4810;
       font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
.toc-title { margin:0 0 10px; color:var(--muted); font-size:12px; font-weight:700;
             letter-spacing:.08em; text-transform:uppercase; }
.toc nav { display:flex; flex-direction:column; gap:3px; }
.toc a { display:block; padding:6px 8px; border-radius:6px; color:#314965; text-decoration:none;
         font-size:13px; line-height:1.45; }
.toc a:hover { color:var(--link); background:#eff4fa; }
.toc a.sub { margin-left:9px; padding-left:9px; color:#3a4a63; font-size:12.5px; }
html { scroll-behavior:smooth; scroll-padding-top:20px; }
h1.title { font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
           font-size:27px; line-height:1.45; text-align:center; margin:0 0 10px; letter-spacing:.2px; }
.authors { text-align:center; color:var(--muted); font-size:14px; margin-bottom:4px; }
.meta { text-align:center; color:var(--muted); font-size:12.5px; margin-bottom:26px; }
.abstract { margin: 0 auto 8px; font-size:15px; line-height:1.8; color:#243248;
            border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
            padding:18px 4px; }
.abstract b { font-family:"Segoe UI","PingFang SC",sans-serif; }
.keywords { font-size:14px; color:var(--muted); margin:14px 0 30px; }
h2 { font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; font-size:19px;
     margin:40px 0 12px; padding-bottom:6px; border-bottom:1px solid var(--rule); }
h2 .num { margin-right:10px; color:var(--link); }
h3 { font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; font-size:16px;
     margin:26px 0 8px; }
h4 { font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; font-size:14.8px;
     margin:18px 0 7px; color:#263b57; }
p { margin: 0 0 14px; text-align: justify; }
ul.tight { margin: 6px 0 16px; padding-left:22px; font-size:15.4px; }
ul.tight li { margin-bottom:6px; }
.eq { position:relative; text-align:center; font-family:"Cambria Math","STIX Two Math","Times New Roman",serif;
      font-size:15px; margin:16px 0; padding:9px 56px; line-height:1.8;
      font-style:normal; background:#fff; border:0;
      overflow-x:auto; overflow-wrap:anywhere; }
.eq i { font-style:italic; }
.eqno { position:absolute; right:12px; top:50%; transform:translateY(-50%); font-style:normal;
        white-space:nowrap; color:#526174; }
figure { margin: 24px auto; text-align:center; }
figure img { display:block; margin:0 auto; max-width:100%; height:auto; }
figure { max-width:100%; }
figure.w100 { width:100%; } figure.w88 { width:88%; } figure.w78 { width:78%; }
figure.w70 { width:70%; } figure.w62 { width:62%; } figure.w60 { width:60%; }
figure.w50 { width:50%; } figure.w42 { width:42%; }
figcaption { font-size:13.4px; color:var(--muted); line-height:1.7; margin-top:9px; text-align:left; }
figcaption .src { display:block; font-size:11.6px; color:#8494ad; margin-top:3px; }
table.bt { width:100%; border-collapse:collapse; margin:20px 0 8px; font-size:13.6px;
           font-family:"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }
table.bt caption { caption-side:top; text-align:left; font-size:13.4px; color:var(--muted);
                   padding-bottom:8px; }
table.bt th { border-top:1.4px solid var(--ink); border-bottom:1px solid var(--ink);
              padding:7px 8px; text-align:left; font-weight:600; }
table.bt td { padding:6px 8px; border-bottom:1px solid #e8edf5; }
table.bt tbody tr:last-child td { border-bottom:1.4px solid var(--ink); }
ol.refs { font-size:13.6px; line-height:1.75; padding-left:24px; color:#243248; }
ol.refs li { margin-bottom:7px; }
sup { font-size:11px; }
footer { margin-top:48px; padding-top:14px; border-top:1px solid var(--rule);
         font-size:12px; color:var(--muted); text-align:center; }

@media (max-width: 640px) {
  .paper-shell { display:block; padding:8px 8px 32px; }
  .toc { position:relative; top:auto; margin:0 0 10px; padding:10px; }
  .toc nav { flex-direction:row; overflow-x:auto; white-space:nowrap; }
  .toc a { flex:0 0 auto; }
  main { padding: 32px 16px 60px; }
  body { font-size:15.5px; }
  h1.title { font-size:21px; }
  figure.w70, figure.w62, figure.w78 { width:100%; }
}

@media (min-width:641px) and (max-width:1120px) {
  .paper-shell { grid-template-columns:175px minmax(0,1fr); gap:16px; padding:18px 12px 42px; }
  main { padding:38px 22px 60px; }
}
"""


def build() -> None:
    copy_assets()
    body = italicize_math_variables("\n".join(SECTIONS) + references())
    page = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{CSS}</style>
</head><body><div class="paper-shell">{toc_html()}<main>
<h1 class="title">{TITLE}</h1>
<div class="meta">研究论文 · 版本日期：{BUILD}</div>
<div id="abstract" class="abstract"><b>摘要　</b>{ABSTRACT}</div>
<div class="keywords"><b>关键词</b>　飞行品质；低阶等效系统；强化学习；教师—学生蒸馏；DAgger；迁移学习</div>
{body}
</main></div></body></html>
"""
    out = ROOT / "dhc6_paper.html"
    out.write_text(page, encoding="utf-8")
    primary = ROOT / "dhc6_report.html"
    primary.write_text(page, encoding="utf-8")
    md = (f"# {TITLE}\n\n> 研究论文 · 版本日期：{BUILD}\n\n"
          f"## 摘要\n\n{ABSTRACT}\n\n"
          "**关键词：** 飞行品质；低阶等效系统；强化学习；教师—学生蒸馏；DAgger；迁移学习\n\n"
          f"{html_to_markdown(body)}\n")
    (ROOT / "dhc6_report.md").write_text(md, encoding="utf-8")
    print(f"Saved {out}, {primary.name} and dhc6_report.md  ({len(page)/1024:.0f} KB, {len(SECTIONS)} blocks)")


if __name__ == "__main__":
    build()
