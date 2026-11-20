"""Revise the manuscript argument and prose without changing experiment records."""
from collections import Counter
from pathlib import Path
import re

from bs4 import BeautifulSoup


TITLE = '基于飞行品质的飞控大模型构建与滚转通道验证'


def nodes(soup, ident):
    heading = soup.find(id=ident)
    result = []
    for node in heading.next_siblings:
        if getattr(node, 'name', None) in ('h2', 'h3', 'h4'):
            break
        if getattr(node, 'name', None):
            result.append(node)
    return heading, result


def set_paragraphs(soup, ident, contents):
    _, block = nodes(soup, ident)
    old = [node for node in block if node.name == 'p']
    assert len(old) == len(contents), (ident, len(old), len(contents))
    for paragraph, content in zip(old, contents):
        paragraph.clear()
        paragraph.append(BeautifulSoup(content, 'html.parser'))


def fig_number(soup, filename):
    figure = next(f for f in soup.find_all('figure') if f.img['src'].split('?')[0].endswith(filename))
    return int(re.match(r'图\s*(\d+)', figure.figcaption.get_text())[1])


def section_ends_before(soup, ident, next_ident):
    return [n for n in soup.find(id=ident).next_siblings
            if getattr(n, 'name', None) and n is not soup.find(id=next_ident)]


def revise_manuscript(soup):
    images_before = Counter(i['src'].split('?')[0] for i in soup.find_all('img'))
    data_before = Counter(tuple(c.get_text() for c in t.select('tbody td')) for t in soup.find_all('table'))
    equations_before = [str(e) for e in soup.select('.eq')]
    figs = {name: fig_number(soup, name) for name in [
        'pipeline_framework.svg', 'teacher_difficulty.svg', 'teacher_training_curves.svg',
        'teacher_closed_loop.svg', 'teacher_student_compare.svg', 'cross_object_matrix.svg',
        'cross_object_responses.svg', 'student_amplitude.svg', 'delay_closed_loop.svg',
        'dhc6_transfer.svg', 'dhc6_states.svg', 'delay_network.svg', 'delay_probe.svg',
        'tau_e_eval.svg', 'tau_e_amplitude.svg', 'tau_e_seed.svg', 'topk4_diagnostic.svg']}

    soup.find('h1', class_='title').string = TITLE
    soup.title.string = TITLE
    abstract = soup.find(id='abstract')
    abstract.clear(); label = soup.new_tag('b'); label.string = '摘要　'; abstract.append(label)
    abstract.append('面向跨对象控制经验的学习与复用，本文提出基于飞行品质的飞控大模型构建路线，并以滚转通道开展验证。通过飞行品质相关参数构造 5,000 个对象，从中选取 50 个训练特权 PPO 教师，再经多轮 DAgger 将逐对象控制经验汇聚为统一策略。蒸馏得到的 tiny 学生含 3,856 个参数，在 44 个蒸馏源对象上相对教师的跟踪 RMSE 中位比为 1.28。面向 500 个分布外对象的 ±1 deg/s 阶跃测试，统一学生通过率为 86.8%，固定增益 PID 为 46.6%，体现了统一策略的跨对象复用能力。目标域采用 75,040 参数的 small 学生，在 DHC-6 辨识模型上经 40 次参数更新，适配状态的 RMSE 中位数由 2.91 降至 0.59 deg/s，留出状态由 2.66 降至 2.15 deg/s。上述结果形成了从品质对象族、专用教师到统一控制策略及目标域适配的滚转通道验证链。')
    keywords = soup.select_one('.keywords')
    keywords.clear(); b = soup.new_tag('b'); b.string = '关键词'; keywords.append(b)
    keywords.append('　飞行品质；飞控大模型；特权强化学习；多教师蒸馏；DAgger；迁移学习')

    set_paragraphs(soup, 's1', [
        '面向不同对象复用控制经验，是飞控大模型需要解决的核心问题。飞机的操纵效能、模态阻尼、响应时间和输入时延随对象及状态变化，控制策略需要在这些差异下维持指令跟踪与振荡抑制能力。飞行品质以响应特征描述对象动态，为组织跨对象控制学习提供了统一的参数基础。',
        '本文据此构建“品质对象族—特权强化学习教师—多轮蒸馏统一策略—目标域适配”的方法链。对象族为学习提供具有明确动态含义的任务分布；逐对象教师形成控制经验；蒸馏将这些经验汇聚到共享策略中，使部署控制器依靠常规观测和历史响应完成反馈控制。目标域适配进一步将共享策略连接到具体机型的辨识模型。',
        '滚转角速度通道用于验证这条构建路线。研究依次回答三个问题：品质参数化对象能否支撑有效教师训练，多对象教师的能力能否由统一策略保留，以及该策略如何随对象变化并通过小样本更新适配目标模型。评价采用同对象教师—学生对照、专用教师跨对象控制、统一学生对象覆盖及 DHC-6 仿真适配，围绕飞控大模型的知识形成、汇聚和复用展开。',
    ])
    set_paragraphs(soup, 's11', [
        '<b>飞行品质与低阶等效模型。</b>飞行品质规范以响应模态和等效参数描述控制对象的动态特征，并给出相应分级判据 [1,2]。这些参数使不同对象的响应差异能够在同一坐标中表达。本文利用这一特点构造学习对象族，以操纵效能、模态参数和时延组织训练与测试。',
        '<b>特权学习与教师—学生蒸馏。</b>特权学习利用训练阶段可获得的信息辅助策略学习 [5]；非对称 actor–critic、教师—学生控制及运动自适应研究从不同角度处理训练信息与部署观测之间的差异 [6–9]。本文采用逐对象特权教师学习控制策略，再将多位教师的动作知识蒸馏到使用常规观测与历史响应的统一学生中。',
        '<b>学习飞行控制与目标域适配。</b>竞速无人机、强风环境控制及 Learning-to-Fly 等研究展示了学习策略在飞行任务中的应用 [10–12]。本文关注飞控策略的跨对象构建：先在品质参数化对象族上积累控制经验，再以 DHC-6 仿真辨识模型检验共享策略的小样本适配。',
        '本文的方法特点在于，以飞行品质组织控制学习的对象空间，通过专用教师积累控制经验，再以多轮蒸馏实现经验汇聚。滚转验证围绕统一策略的能力保留、跨对象复用、历史表征及目标域更新建立证据，形成飞控大模型构建路线的完整论证。',
    ])

    p = [n for n in nodes(soup, 's22')[1] if n.name == 'p']
    p[0].string = '源域采用参数化低阶等效模型描述输入力到滚转角速度的动态关系。模型包含滚转衰减、螺旋和荷兰滚模态，并以运输时延描述输入作用的滞后，其传递函数为：'
    p[1].clear(); p[1].append(BeautifulSoup('对象参数向量 <i>θ</i> 包含八个采样参数，零点频率与阻尼由比值关系派生。各参数的定义、单位及动态作用见<a href="#model-parameter-table">参数说明表</a>。通过改变这些参数，可以在统一结构下生成具有不同操纵效能、振荡特征和响应时间的控制对象。', 'html.parser'))
    p = [n for n in nodes(soup, 's21')[1] if n.name == 'p']
    p[0].clear(); p[0].append(BeautifulSoup('被控量为滚转角速度 <i>p</i>，外部指令为 <i>p</i><sub>cmd</sub>。控制任务包括指令跟踪，以及对初始偏差和外部扰动引起的振荡进行抑制。跟踪误差定义为 <i>e</i><sub>t</sub> = <i>p</i><sub>ref,t</sub> − <i>p</i><sub>t</sub>，其中 <i>p</i><sub>ref</sub> 为相应实验记录的目标速率信号。', 'html.parser'))
    p[1].clear(); p[1].append(BeautifulSoup('策略输出归一化力速率动作 <i>a</i><sub>t</sub>。动作经速率约束、积分和限幅形成请求力，再由执行机构动态生成施加力。目标域接口通过力—副翼偏角映射连接 DHC-6 辨识模型，使源域学习与目标域适配采用一致的控制接口。', 'html.parser'))
    for node in list(p[2].find_all(string=True)):
        node.replace_with(str(node).replace('观测按三段拼接，维数可以直接核对：', '常规观测由当前量、历史响应和动作历史三部分组成：'))
    p[4].clear(); p[4].append(BeautifulSoup('教师使用常规观测和 12 维特权量 <i>z</i><sub>t</sub>，后者包含八个对象参数与四个模型内部状态，合计输入维数为 188。学生使用 176 维常规观测，由历史编码分支提取对象动态信息。', 'html.parser'))

    _, block = nodes(soup, 's23')
    [n for n in block if n.name == 'p'][0].clear()
    [n for n in block if n.name == 'p'][0].append(BeautifulSoup('飞行品质规范用于确定对象模型及分级依据 [1,2]。闭环性能统一采用目标跟踪 RMSE、峰值误差和请求力总变差率评价，项目门限依次为 1.5 deg/s、5 deg/s 和 12 N/s。', 'html.parser'))
    set_paragraphs(soup, 's3', [f'图 {figs["pipeline_framework.svg"]} 给出基于飞行品质构建飞控大模型的四个环节。首先通过品质参数采样形成对象族，随后逐对象训练特权教师；再由多轮蒸馏将控制经验汇聚为统一学生，最后以目标域仿真数据更新学生参数。方法部分依次定义学习对象、教师优化、知识汇聚和目标域更新，结果部分按相同顺序验证各环节。'])
    set_paragraphs(soup, 's31', [
        '在第 2.1 节的八维参数空间内生成 5,000 个滚转对象。采样按飞行品质等级和数据用途设置覆盖配额：4,500 个对象用于训练及同分布评估，500 个组成分布外评估集。参数及适用判据共同确定对象的等级标签，从而将飞行品质描述转化为可控制、可分层评价的学习对象分布。',
        '其中，<i>θ</i><sup>(i)</sup> 为第 <i>i</i> 个对象的参数向量，<i>c</i><sub>i</sub> 表示采样类别，<i>q</i> 为覆盖配额约束下的采样分布，<i>g</i><sub>i</sub> 为相应等级标签。',
    ])
    set_paragraphs(soup, 's32', [
        '对选定的 50 个源域对象分别训练 PPO 教师 [3]。逐对象训练将对象动态差异分解为独立控制任务，教师利用已知参数和模型内部状态学习相应反馈策略。Actor 依据式 (2) 的常规观测与特权量输出归一化动作 <i>a</i><sub>t</sub> ∈ [−1,1]。在 50 Hz 策略频率下，请求力每步最多变化 1.76 N，并限制于 ±22 N。',
        '力接口以 1 kHz 更新，命令力按 88 N/s 跟随请求力，再经时间常数 20 ms 的一阶执行机构模型作用于对象。±22 N 与 88 N/s 分别定义本实验的虚拟力幅值和速率范围；目标域采用 <i>δ</i><sub>a</sub> = −(10°/22 N)<i>F</i> 桥接为副翼请求，对应 ±10° 的位置范围和 40°/s 的速率范围。',
        '教师网络以一维卷积编码历史，并由残差主干产生动作分布。均值头使用无偏置线性层和奇对称投影，使零观测对应零均值输出。表 1 列出网络结构和训练设置。',
    ])
    set_paragraphs(soup, 's321', [
        '即时奖励由跟踪误差、施加力和力速率动作三项组成。令 <i>e</i><sub>t</sub> = <i>p</i><sub>ref,t</sub> − <i>p</i><sub>t</sub>，角速度归一化尺度 <i>p</i><sub>scale</sub> = 30 deg/s，则：',
        '误差项驱动策略跟踪目标速率，施加力项约束操纵强度，动作幅值项约束请求力的更新强度。三项共同形成跟踪精度与控制活动之间的优化权衡。',
    ])
    set_paragraphs(soup, 's322', [
        '采用广义优势估计（GAE）评价已采样动作的相对收益，时序差分残差与优势递推为：',
        '令 <i>ρ</i><sub>t</sub>(w) 为新旧策略对同一已采样动作的概率比，PPO 剪切代理损失为：',
        '训练同时优化策略损失、价值估计误差与熵正则项：',
        '每个对象训练 30 轮，每轮由 512 个并行环境各运行 400 个策略步，累计约 614 万个状态转移。Actor 与 Critic 使用 Adam，学习率为 3×10<sup>−4</sup>。',
    ])
    soup.find(id='s323').string = '3.2.3 飞行品质、奖励与评价的分工'
    set_paragraphs(soup, 's323', ['飞行品质参数定义控制对象的动态特征和等级覆盖，奖励函数驱动教师优化跟踪与操纵强度，性能门限则统一评价跟踪误差、峰值和操纵变化。三者分别承担对象组织、策略学习和结果评价的职责，构成从品质描述到控制能力验证的联系。'])
    set_paragraphs(soup, 's33', [
        '统一学生通过动作模仿汇聚多对象教师的控制经验。初始样本由教师访问闭环状态并标注动作；随后由学生在对象上运行，再由对应教师为学生访问到的状态提供动作标签 [4]。六轮 DAgger 数据聚合后，分别得到 3,856 参数的 tiny 学生和 75,040 参数的 small 学生。两者均以常规观测和历史作为部署输入。',
        '其中，<i>μ</i><sub>T</sub> 为教师策略，<i>z</i><sub>t</sub> 为对象参数与模型内部状态。第 <i>k</i> 轮学生产生访问状态后，对应教师利用该状态给出监督动作。学生通过最小化聚合样本上的动作均方误差学习共享策略：',
        '多轮聚合使训练样本随学生闭环行为更新，逐步覆盖学生自身会访问的状态。这将专用教师的动作知识与统一策略的实际运行状态联系起来，是从逐对象控制走向共享控制的关键环节。',
    ])
    set_paragraphs(soup, 's34', [
        '目标域采用 12 个 DHC-6 仿真配平状态的 72 条、50 Hz 输入输出记录。通过扫频激励辨识各状态的 ARX 模型，阶次范围为 na = 4–5、nb = 3–4、nk = 1，拟合 VAF 为 98.0%–98.7%。策略输出经相同力接口积分与限幅，再按 <i>δ</i> = −(10/22)<i>F</i> 映射为副翼偏角，辨识模型的滚转角速度反馈至策略。',
        '采用 small 学生进行目标域更新，优化其全部 75,040 个参数。8 个配平状态提供适配样本，另外 4 个状态用于更新后的留出评价。优化执行 40 次 Adam 更新，学习率为 3×10<sup>−5</sup>；损失统计响应开始 2 s 后的跟踪平方误差。评价使用确定性策略、15 deg/s 正阶跃及 8 s 时域。',
        '式中，<i>W</i> 为学生参数，Ĝ<sub>ARX</sub> 为适配状态的辨识模型。适配组用于参数更新，留出组用于观察更新所得控制知识的跨状态复用。',
    ])
    set_paragraphs(soup, 's4', ['实验围绕滚转通道的构建路线展开：主试验检验教师学习与学生蒸馏，配对实验检验专用教师迁移与统一学生的多对象控制，表征对照分析历史输入设计，DHC-6 仿真检验目标域适配。各组的对象集、控制器与评价协议见表 2。'])
    set_paragraphs(soup, 'experiment-data-protocol', ['主试验学生在 48 个对象上采用 30 条加权指令和每对象 64 条并行确定性轨迹评价。幅值扫描包含 300 个同分布对象、500 个分布外对象及 14 条输入配置，每个对象—指令组合重复两次。12 对象配对组包含三组训练重复；DHC-6 以配平状态为适配和留出单元，在固定阶跃及单一策略种子下评价。各组分别按其对象和测试单元统计。'])

    set_paragraphs(soup, 's51', [f'品质参数化对象库包含 5,000 个对象，其中 4,500 个用于训练及同分布评估，500 个用于分布外评估。采样散点与边际分布展示了操纵效能、模态和时延的覆盖，教师训练对象在同一参数空间中标出。该对象库将品质描述落实为可重复的控制任务，为后续教师训练和跨对象评价提供共同基础。'])
    set_paragraphs(soup, 's52', [
        '主试验选取 50 个源域对象开展逐对象教师训练，已有 46 个教师达到统一性能门限。已记录的未通过样例包括 train_core-0174、train_core-0236、train_core-0467。',
        f'图 {figs["teacher_difficulty.svg"]} 显示，教师跟踪误差随对象操纵效能变化：通过对象中的双对数相关系数为 −0.74，所列未通过样例也位于低效能区域。对象族因此既提供了多样化控制经验，也为分析训练任务的动态差异提供了依据。',
        f'图 {figs["teacher_training_curves.svg"]} 展示 21 个具有逐轮固定验证记录的达标教师。验证 RMSE 随训练总体下降，所选权重由验证表现确定，可对应不同训练轮次。',
        f'图 {figs["teacher_closed_loop.svg"]} 给出三个品质等级代表对象的阶跃与多正弦响应，同时展示角速度跟踪和操纵量变化，为蒸馏前的教师控制能力提供时域依据。',
    ])
    set_paragraphs(soup, 's53', [
        '六轮 DAgger 后，tiny 学生在 44 个蒸馏源对象上的 RMSE 中位比为 1.28，在 4 个留出对象上为 1.68；small 学生相应为 1.34 和 1.41。两者在留出对象上均通过 3/4。统一学生以共享权重保留了多对象教师的主要控制能力，两档网络则呈现不同的精度与参数规模组合。',
        '学生评估集共 48 个对象，三个品质等级各 16 个；44 个用于 DAgger，4 个作为对象级留出。每个对象采用 30 条加权指令与 64 条并行确定性轨迹。表 3 汇总通过数和相对误差，其中中位比按各对象的学生 RMSE 除以对应教师 RMSE 后取中位数。',
    ])
    table = next(t for t in soup.find_all('table') if '两种规模学生的源域结果' in t.caption.get_text())
    for th, text in zip(table.find_all('th'), ['学生', '参数量', '源对象通过', '留出对象通过', '源对象中位倍数', '留出对象中位倍数']):
        th.string = text
    set_paragraphs(soup, 's531', [f'图 {figs["teacher_student_compare.svg"]} 在同一源对象、同一指令下比较教师与学生。六档阶跃的上升过程接近，学生在 2 s 后的跟踪 RMSE 约为教师的 2.3–3.3 倍；双脉冲与多正弦上的比值为 1.5–1.9。该样例展示了蒸馏后响应形态的保留及其精度变化，与总体统计共同刻画知识汇聚的效果。'])

    revise_cross_object_section(soup, figs)
    set_paragraphs(soup, 's533', [
        '进一步在不同指令幅值和对象分布上评价统一学生。按品质等级从同分布对象池抽取 300 个对象，每级 100 个，并纳入全部 500 个分布外对象。各对象测试 ±1、±2、±5、±10、±21、±31 deg/s 阶跃，以及双脉冲和多正弦，共 11,200 个对象—指令组合；每组合运行两次，共 22,400 条轨迹。通过率按组合级记录及统一门限统计。',
        '同分布对象上，±1 deg/s 阶跃通过率为 99.7%，RMSE 中位数为 0.11 deg/s；±31 deg/s 相应为 60.3% 和 1.13 deg/s。双脉冲、多正弦通过率分别为 99.0%、99.7%。这些结果展示了统一策略在多种指令形态下的复用能力，并给出通过率随指令幅值变化的关系。',
        '在 500 个分布外对象上，±1 deg/s 和 ±31 deg/s 的通过率分别为 86.8%、27.6%，RMSE 中位数分别为 0.21、2.93 deg/s。结合固定的 ±22 N 力幅值和 88 N/s 速率接口，可将指令需求与对象动态共同作为分析控制范围的依据。300 个同分布扫描对象与主试验的 48 个评估对象不重叠，扫描期间学生权重固定。',
    ])
    set_paragraphs(soup, 's534', [
        '在相同的 300 个同分布对象、500 个分布外对象及指令配置上，以固定增益二自由度 PID 为对照。PID 含参考前馈和抗积分饱和，采用相同的 ±22 N 请求力范围及力速率接口。增益在主试验对象 7 上通过网格搜索确定，取 <i>k</i><sub>p</sub> = 128、<i>k</i><sub>i</sub> = 60、<i>k</i><sub>d</sub> = 0.8、<i>k</i><sub>ff</sub> = 6.0，随后在全部对象上保持固定。',
        '统一学生在下表各类指令的汇总通过率均高于固定增益 PID。同分布 ±1 deg/s 为 99.7% 对 80.0%，±31 deg/s 为 60.3% 对 35.7%；分布外相应为 86.8% 对 46.6%、27.6% 对 16.2%。分布外指令扫描中，PID 的 RMSE 中位数为 8.8–18.5 deg/s，学生为 0.21–2.93 deg/s。±1 deg/s 同分布条件下，PID 的中位误差为 0.06 deg/s，学生为 0.11 deg/s；联合峰值误差和操纵变化评价时，学生获得更高通过率。',
        '该对照突出统一策略的跨对象复用：同一组学生权重在多种动态对象上保持较高的联合指标通过率。固定增益 PID 提供单一控制律的参照，比较结果同时呈现小幅值跟踪精度与多对象覆盖两方面的特征。',
    ])
    set_paragraphs(soup, 's535', [
        '部署学生通过历史响应利用对象动态信息，表征的评价因而需要与闭环控制目标一致。本节比较基线的任务驱动时序编码与 PCA、稠密 AE、SAE 冻结前端，各替换前端均输出 16 维历史特征。对照采用 510 个分层对象和 489,600 个窗口，按对象划分训练、验证与测试，其中 120 个对象用于留出评价。',
        f'图 {figs["delay_closed_loop.svg"]} 显示，保留基线时序编码的学生与 SAE 辅助输入方案在长延迟层均达到 90.0% 通过率；PCA-16、稠密 AE-16、Top-k SAE-16 替换方案分别为 51.6%、63.7%、71.9%。这一对照体现了基线历史编码方案对闭环控制能力的保留。',
        '历史读出结果进一步解释了表征评价的维度：1.2 s 因果前缀上的请求力—响应互相关达到 R² = 0.995、MAE = 2.9 ms，稠密 AE-16 和 Top-k SAE（k=4）的 R² 分别为 0.608、0.122。参数读出与闭环控制的排序不同，表明历史中的动态线索需要通过策略有效转化为控制动作。完整探针对照见<a href="#s612">附录 A.2</a>。',
        '本路线据此采用与动作模仿共同优化的时序编码，将观测历史映射为 16 维特征并参与统一决策。替换与辅助输入分支的训练设置见第 3.3.1 节，完整对照和长时响应分析见<a href="#s613">附录 A.3–A.6</a>。',
    ])
    set_paragraphs(soup, 's54', [
        '在 DHC-6 ARX 目标模型上，small 学生经 40 次参数更新后，8 个适配状态的 RMSE 中位数由 2.913 降至 0.595 deg/s，4 个留出状态由 2.662 降至 2.148 deg/s，分别降低 79.6% 和 19.3%。所有状态均采用 15 deg/s 阶跃和 8 s 评价时域。适配组与留出组的改善共同体现了源域控制知识在目标模型上的更新与复用。',
        '相同微调设置下，tiny 学生的适配后 RMSE 为 1.57 deg/s，small 为 0.59 deg/s，相差约 2.7 倍；源域蒸馏的中位比相应为 1.28 和 1.34。两档网络分别提供轻量共享策略与较大参数容量的实现选择，small 在本组目标域更新中取得更高跟踪精度。40 次更新中，每个适配状态被访问 5 次，构成该比较的共同更新预算。',
    ])
    set_paragraphs(soup, 's6', [
        '飞行品质为飞控大模型提供了组织控制知识的坐标。操纵效能、模态阻尼、响应时间和时延将不同对象转化为结构统一的控制任务；逐对象教师学习相应反馈策略，多轮蒸馏再将这些策略汇聚为共享参数。滚转通道结果表明，该框架能够将对象族上的控制经验转化为统一学生的多对象控制能力。',
        '知识汇聚的价值体现在精度、覆盖和实现规模的组合上。tiny 学生以 3,856 个参数实现多对象控制，在分布外扫描中获得高于固定增益 PID 的通过率；历史表征对照则支持保留与控制任务共同学习的时序编码。跟踪门限、操纵活动和恢复过程共同描述控制性能，长时响应进一步提供振荡衰减与残余误差的信息。',
        '目标域适配将这条路线延伸到 DHC-6 仿真记录辨识的模型。small 学生在有限更新预算下同时改善适配状态与留出状态，展示了共享控制知识的小样本更新能力。由品质对象族到目标模型的验证，为后续加入执行机构特性和多轴耦合提供了方法基础。',
    ])
    set_paragraphs(soup, 's62', [
        '后续验证围绕执行机构约束与完整控制回路展开。表 13 给出力限幅、速率限幅及执行机构滞后的计划对照。在固定对象和指令下比较冻结策略，可观察控制性能随接口条件的变化；采用相同预算更新策略，则可进一步评价适应这些条件的能力。记录指标包括教师达标率、跟踪误差、操纵变化、饱和比例以及目标域适配表现。',
        '力与副翼的桥接关系将用于同步描述源域和目标域的位置、速率范围。后续研究将进一步结合更高保真对象、滚转—偏航耦合和实时执行条件，扩展飞控大模型构建路线的验证范围。',
    ])
    set_paragraphs(soup, 's7', [
        '本文提出基于飞行品质的飞控大模型构建路线，并以滚转通道完成从对象组织、特权教师学习、多轮蒸馏到目标域适配的验证。飞行品质参数将对象差异组织为可学习任务，教师控制经验经 DAgger 汇聚为统一策略，实现了控制知识的离线积累与共享。',
        '统一学生以 3,856 个参数保留多对象控制能力，在 44 个蒸馏源对象上的 RMSE 中位比为 1.28。在 500 个分布外对象的 ±1 deg/s 测试中，通过率达到 86.8%，高于固定增益 PID 的 46.6%。历史表征对照支持使用任务驱动的时序编码，形成从观测历史到统一控制动作的部署形式。',
        'DHC-6 仿真适配进一步展示了共享策略的更新能力：small 学生经 40 次更新，将适配状态 RMSE 中位数降至 0.59 deg/s，留出状态降至 2.15 deg/s。上述证据共同支撑了基于飞行品质组织控制知识、通过教师—学生学习形成统一飞控策略的技术路线。',
    ])

    # Language consistency in supplementary prose and captions, preserving
    # numbers, model names, mathematical definitions and comparison groups.
    for node in list(soup.find_all(string=True)):
        if node.parent.name in ('script', 'style'):
            continue
        value = str(node)
        value = re.sub(r'(\d+)\s*架', r'\1 个', value)
        value = value.replace('每架运行', '每个对象运行').replace('二阶自由度 PID', '二自由度 PID')
        value = value.replace('，。', '。').replace('原始历史的 raw_ft', '原始历史微调分支')
        value = value.replace('原始历史的 原始历史微调', '原始历史微调分支')
        value = value.replace('raw 与', '原始历史分支与').replace('低于 raw', '低于原始历史分支')
        value = value.replace('raw 为', '原始历史分支为').replace('（raw）', '（原始历史）')
        if value != str(node):
            node.replace_with(value)
    # Reflect revised wording in the table of contents without changing order.
    for a in soup.select('.toc a[href^="#"]'):
        heading = soup.find(id=a['href'][1:])
        if heading and heading.name in ('h2', 'h3', 'h4'):
            a.string = heading.get_text(' ', strip=True)
    assert images_before == Counter(i['src'].split('?')[0] for i in soup.find_all('img'))
    assert data_before == Counter(tuple(c.get_text() for c in t.select('tbody td')) for t in soup.find_all('table'))
    assert equations_before == [str(e) for e in soup.select('.eq')]


def revise_cross_object_section(soup, figs):
    _, block = nodes(soup, 's532')
    p = [n for n in block if n.name == 'p']
    assert len(p) == 11, ('s532', len(p))
    contents = [
        '为直接检验逐对象控制经验的汇聚，采用 12 对象配对组比较专用教师与统一学生。三个品质等级各含 4 个对象，每组包含 12 个教师及由其蒸馏的一个 tiny 学生，共三组训练重复。全部控制器在固定权重下运行。',
        '每个教师依次控制全部 12 个对象，统一学生也控制相同对象。教师—对象矩阵的对角线表示自身对象控制，非对角线表示迁移到其他对象，末行表示统一学生。迁移教师使用当前被控对象的参数与状态，学生使用常规观测及历史。每格采用 18 条测试指令和 8 s 时域，按统一三项门限评价。',
        f'图 {figs["cross_object_matrix.svg"]} 可按行比较同一教师换对象后的变化，按列比较同一对象上不同控制器的表现。每格汇总三组重复，共 54 条测试。例如教师 1 在对象 1 上通过 46/54（85%），在对象 6 上通过 22/54（41%）。右侧圆点、方块和三角形表示三组训练重复，横线表示平均值。',
        '自身教师与统一学生各包含 648 个测试组合，迁移教师包含 7,128 个组合。响应非有限或角速度绝对值超过 300 deg/s 时记为失败；通过率按全部测试组合计算，RMSE 中位数按有限轨迹计算。',
        '统一学生通过率为 66.4%，接近自身教师的 68.2%，高于教师迁移后的 27.6%。这一结果直接体现了多教师知识汇聚的作用：一套共享策略覆盖多个源对象，保留了与逐对象教师整体接近的控制能力。三组重复中，学生分别为 87.0%、63.9%、48.1%，自身教师为 82.4%、61.1%、61.1%。',
        '配对组用于评价已蒸馏对象上的能力保留；主试验学生的新增对象评估进一步考察共享策略的对象覆盖。',
        '主试验 tiny 学生在 120 个新增同分布对象上通过 94/120（78.3%），三个等级分别通过 34/40、30/40、30/40，RMSE 中位数为 0.64 deg/s，90 百分位为 2.24 deg/s。这些对象从 4,500 个同分布对象池按每级 40 个抽取，与主试验 48 个评估对象不重叠。',
        '在新增对象中，RMSE 与滚转操纵效能的相关系数为 −0.47；滚转时间常数 <i>T</i><sub>R</sub> > 6 s 的 4 个对象中有 1 个通过。对象参数与性能的对应关系为下一阶段扩大训练覆盖提供了具体方向。',
        f'图 {figs["cross_object_responses.svg"]} 比较同一对象上的自身教师、迁移教师和统一学生。样例按训练阶段独立验证结果选择：第一组有 8 个教师达标，每个品质等级按原对象顺序取首个，得到对象 1、6、9及迁移教师 6、9、1。三类控制器采用相同指令、初始偏差和外力扰动，选样与验证指标见<a href="evidence/cross_object_20261025/illustration_protocol_v2.json">时域对照协议</a>。',
        '时域结果同时展示跟踪和恢复动态。长时响应中的振荡包络与末段残余响应，为整段误差门限提供补充信息；完整示例见<a href="paper/cross_object_diagnostic.svg">闭环诊断响应</a>。例如配对对象 2 的自身教师通过 4/18 条测试指令，其 30 s 扰动测试虽达到三项门限，后段振荡包络仍随时间增大。',
        None,
    ]
    for node, text in zip(p, contents):
        if text is not None:
            node.clear(); node.append(BeautifulSoup(text, 'html.parser'))
    # Keep each experiment together: paired matrix -> paired traces -> coverage.
    trace = soup.find(id='cross-object-time-response')
    trace.insert_before(p[8].extract())
    links = p[10]
    links.insert_after(p[7].extract()); links.insert_after(p[6].extract())
    p[5].extract()
    p[6].insert_before(p[5])


def polish_after_numbering(soup):
    p = soup.find(id='closed-loop-explanation')
    if p:
        n = fig_number(soup, 'closed_loop_simulation.svg')
        p.string = f'图 {n} 展示源域学生闭环仿真及响应曲线的生成过程。品质参数确定对象模型后，学生每 20 ms 根据观测和历史输出动作，经力接口与执行机构形成控制输入；仿真器以 1 ms 步长推进对象状态，并将新的角速度反馈至策略。按时间记录角速度与操纵量，即得到后文的时域响应。每条轨迹对应一组固定对象参数与固定策略权重。'
    captions = {
        'pipeline_framework.svg': '飞控大模型的构建与验证流程：品质对象族、特权 PPO 教师、多轮蒸馏与目标域适配。tiny 与 small 学生分别含 3,856 和 75,040 个参数；40 次目标域更新及 2.91→0.59 deg/s 的结果对应 small。',
        'closed_loop_simulation.svg': '学生控制器的闭环仿真。品质参数确定对象动态，学生根据观测和历史输出力速率动作，对象响应反馈至下一控制周期。',
        'teacher_student_compare.svg': '教师与统一学生在主试验对象 7 上的同指令时域对照。黑色点线为目标，蓝线为教师，橙线为学生；标题报告 2 s 后的跟踪 RMSE 及其比值。',
        'cross_object_matrix.svg': '教师跨对象控制与统一学生的多对象覆盖。每格为三组训练重复的平均通过率；右侧点为各组结果，横线为平均值。对象与其专用教师按 1–12 对应编号。',
        'cross_object_responses.svg': '验证达标教师与统一学生的闭环对照。三行受控对象为 1、6、9，迁移教师为 6、9、1；三列依次展示指令跟踪、初始偏差恢复和外力扰动抑制。黑色实线为外部指令，灰色点线为记录目标。',
        'teacher_training_curves.svg': '21 个达标教师的固定指令集验证结果：跟踪误差、操纵总变差及所选权重对应的训练步数。',
        'delay_closed_loop.svg': '历史表征的闭环对照。左：按输入时延分层的通过率。右：全延迟样本与短/中延迟样本再训练后的长延迟表现。留出集包含 120 个对象、960 条对象—指令轨迹。',
    }
    for figure in soup.find_all('figure'):
        filename = Path(figure.img['src'].split('?')[0]).name
        if filename in captions:
            number = int(re.match(r'图\s*(\d+)', figure.figcaption.get_text())[1])
            figure.figcaption.string = f'图 {number}　' + captions[filename]
    for image in soup.find_all('img'):
        if image.has_attr('alt'):
            image['alt'] = image.find_parent('figure').figcaption.get_text(' ', strip=True)
