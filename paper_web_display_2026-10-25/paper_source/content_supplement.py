"""Explain the archived sampling protocol and the evidence in each figure."""
from pathlib import Path
import html
import json
import re
import subprocess

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]


def math(tex):
    return (f'<span class="supplement-math" data-tex="{html.escape(tex, quote=True)}">'
            f'{html.escape(tex)}</span>')


def set_html(node, markup):
    node.clear()
    node.append(BeautifulSoup(markup, 'html.parser'))


def paragraphs(soup, ident):
    result = []
    for node in soup.find(id=ident).find_next_siblings():
        if node.name in ('h2', 'h3', 'h4'):
            break
        if node.name == 'p':
            result.append(node)
    return result


def add_paragraph(soup, after, markup, ident=None):
    node = soup.new_tag('p')
    if ident:
        node['id'] = ident
    set_html(node, markup)
    after.insert_after(node)
    return node


def supplement_methods(soup):
    object_text = paragraphs(soup, 's22')
    set_html(object_text[1],
        '对象参数向量 <i>θ</i> 由八项组成，零点频率与阻尼通过比值关系确定。'
        '各参数的定义、单位及动态作用见<a href="#model-parameter-table">参数说明表</a>，'
        '联合生成与筛选方式见第 3.1 节。')
    set_html(soup.find(id='model-parameter-note'),
        '前八项组成对象参数向量，后两项给出相应零点参数。'
        '操纵效能由响应灵敏度标定，零点阻尼比例系数由阻尼比关系构造。'
        '单位按输入力 N、输出角速度 rad/s 定义。')
    parameter_rows = soup.select('#model-parameter-table tbody tr')
    parameter_rows[0].find_all('td')[-1].string = '响应标定'
    parameter_rows[6].find_all('td')[-1].string = '比值构造'
    parameter_rows[9].find_all('td')[-1].string = '约束选取'

    task = paragraphs(soup, 's21')[0]
    set_html(task,
        '被控量为滚转角速度 <i>p</i>，外部指令为 <i>p</i><sub>cmd</sub>。'
        '主试验将外部指令平滑处理，并按当前对象的输入运输时延对齐，得到目标速率 '
        '<i>p</i><sub>ref</sub>。控制器根据目标与反馈产生动作，跟踪误差定义为 '
        '<i>e</i><sub>t</sub> = <i>p</i><sub>ref,t</sub> − <i>p</i><sub>t</sub>。'
        '同一对象上的师生对照使用相同的指令与目标序列；后文的跟踪误差均相对所记录的目标计算。')

    sampling = paragraphs(soup, 's31')
    set_html(sampling[0],
        '对象库按 IV 类飞机、A 类飞行阶段的滚转、螺旋、荷兰滚基准和灵敏度相关条件组织候选 [13]。'
        '采用九维扰动 Sobol 序列，先确定候选的品质区域，按目标分量设置品质梯度，其余分级分量取基础区域；'
        '随后生成模态、时延及灵敏度参数。'
        '对每个候选重新计算滚转、螺旋、荷兰滚和灵敏度分量的等级，按其中最差等级分层选入。'
        '最终得到 5,000 个对象，L1、L2、L3 各 1,500 个，另有 500 个分布外对象。')
    set_html(sampling[1],
        '其中，<i>θ</i><sup>(i)</sup> 为第 <i>i</i> 个对象的参数向量，'
        '<i>c</i><sub>i</sub> 表示候选品质区域，<i>q</i> 为条件生成与配额筛选形成的分布，'
        '<i>g</i><sub>i</sub> 为四项分量判据给出的源域分层标签。参数范围及构造方式见表 88。')

    rows = [
        (math(r'L_{F_a}'),
         '由 1 s 滚转角灵敏度反标；同分布筛选包络为 '+math(r'[0.04,\,0.47083875]'),
         '将标定后效能超出该包络的候选归入分布外。'),
        (math(r'T_R'),
         '按对数尺度映射；基础区间为 '+math(r'[0.18,\,1.0]')+' s，等级扩展上限为 1.4、10 s。',
         '慢滚转候选扩展至 '+math(r'(10,\,15]')+' s。'),
        (math(r'\lambda_s'),
         '按线性尺度映射；基础区间为 '+math(r'[-0.15,\,\ln 2/12]')+
         ' '+math(r'\mathrm{s}^{-1}')+'，等级扩展上限为 '+math(r'\ln 2/8')+'、'+math(r'\ln 2/4')+'。',
         '不稳定根扩展至 '+math(r'(\ln 2/4,\,0.30]')+' '+math(r'\mathrm{s}^{-1}')+'。'),
        (math(r'\zeta_d'),
         '与模态频率联合取值，候选范围为 '+math(r'[0.005,\,0.70]')+
         '；按阻尼比及阻尼与频率乘积的分级条件设置下限。',
         '沿用阻尼构造，允许频率超出常规分级范围。'),
        (math(r'\omega_d'),
         '按对数尺度映射；基础区间为 '+math(r'[1,\,6]')+' rad/s；'
         '频率项决定二级时取 '+math(r'[0.4,\,0.99]')+' rad/s，三级低阻尼候选取 '+math(r'[0.4,\,2]')+' rad/s。',
         '低频候选扩展至 '+math(r'[0.20,\,0.40)')+' rad/s。'),
        (math(r'R_\omega'),
         '按对数尺度映射到 '+math(r'[0.65,\,1.35]')+'。',
         '现有库的分支映射取值约为 '+math(r'[0.30,\,0.424]')+' 或 '+math(r'[2.121,\,3.0]')+'。'),
        (math(r'R_\zeta'),
         '先在线性尺度上选取 '+math(r'\zeta_\phi\in[0.1,\,0.7]')+'，再令 '+
         math(r'R_\zeta=\zeta_\phi/\zeta_d')+'。',
         '采用相同的比值构造。'),
        (math(r'\tau_p'),
         '按线性尺度映射到 '+math(r'[0.001,\,0.20]')+' s。',
         '长时延候选扩展至 '+math(r'(0.25,\,0.50]')+' s。'),
    ]
    table = soup.new_tag('table', attrs={'class': 'bt sampling-protocol-table', 'id': 'sampling-protocol-table'})
    caption = soup.new_tag('caption'); caption.string = '表 88　对象族的候选范围与构造方式'; table.append(caption)
    thead = soup.new_tag('thead'); tr = soup.new_tag('tr')
    for label in ['参数', '同分布候选的生成与筛选', '分布外扩展']:
        th = soup.new_tag('th', scope='col'); th.string = label; tr.append(th)
    thead.append(tr); table.append(thead); tbody = soup.new_tag('tbody')
    for row in rows:
        tr = soup.new_tag('tr')
        for value in row:
            td = soup.new_tag('td'); set_html(td, value); tr.append(td)
        tbody.append(tr)
    table.append(tbody)
    wrapper = soup.new_tag('div', attrs={'class': 'parameter-table-wrap'}); wrapper.append(table)
    sampling[1].insert_after(wrapper)
    gain = add_paragraph(soup, wrapper,
        '操纵效能的标定以 1 N 阶跃作用下的响应为基础：对角速度积分，取 1 s 时刻滚转角的绝对值与输入力幅值之比作为灵敏度。'
        '灵敏度目标按对数尺度从 0.85–3.38 deg/N 选取，二级候选扩展至 5.62 deg/N，'
        '分布外候选扩展至 8.43 deg/N。每次针对选定的分级分量扩展取值，再重新检查完整参数组合。',
        'gain-calibration-protocol')
    add_paragraph(soup, gain,
        '分布外划分同时考虑分级边界和参数包络，包括过大的滚转时间常数、不稳定螺旋根、'
        '低于 0.4 rad/s 的模态频率、高于 5.62 deg/N 的灵敏度，以及超出效能、零点频率比或时延包络的组合。'
        '5,000 个对象按对象编号划为 3,000 个训练候选、450 个验证对象、1,050 个同分布测试对象和 500 个分布外测试对象；'
        '教师训练、蒸馏及后续抽样使用这些对象池中的相应子集。',
        'ood-generation-protocol')

    method = paragraphs(soup, 's33')[0]
    set_html(method, str(method.decode_contents()).replace(
        '六轮 DAgger 数据聚合后', '经行为克隆初始化和五轮 DAgger 数据聚合后'))

    intro = paragraphs(soup, 's1')
    intro[0].string = (
        '不同对象的操纵效能、模态阻尼和输入时延决定了不同的反馈控制需求。'
        '飞控策略既要跟踪任务指令，也要处理对象变化带来的过冲、振荡与操纵需求变化。'
        '本文研究如何利用飞行品质对这些动态差异进行组织，并将多个对象上的控制经验用于同一套策略。')
    intro[1].string = (
        '逐对象训练可以分别学习各自的反馈关系，但得到的仍是多个专用控制器。'
        '要将这些经验用于统一控制，还需处理对象之间的状态分布差异，以及训练信息与部署观测的差异。'
        '本研究采用具有对象参数和内部状态信息的教师学习专用策略，'
        '再在学生实际访问的状态上标注动作，训练仅使用常规观测和历史的共享策略。')
    intro[2].string = (
        '滚转角速度通道用于检验这一构建方式。实验依次考察专用教师的控制效果、'
        '蒸馏后的跟踪精度、多对象及未参与蒸馏对象上的闭环表现，'
        '并通过历史表征对照和 DHC-6 辨识模型上的参数更新分析输入设计与目标域适配。')
    related = paragraphs(soup, 's11')
    related[-1].string = (
        '本文以统一权重的多对象控制为评价重点。同对象师生对照用于衡量蒸馏后的精度变化，'
        '专用教师跨对象测试用于比较直接复用与知识汇聚，'
        '新增对象测试及目标域更新用于考察共享策略在不同动态条件下的表现。')


def find_following_paragraph(figure):
    for node in figure.find_next_siblings():
        if node.name in ('h2', 'h3', 'h4'):
            break
        if node.name == 'p':
            return node
    return None


def finalize_content(soup):
    figures = {Path(f.img['src'].split('?')[0]).name: f for f in soup.find_all('figure')}
    assert len(figures) == 22
    numbers = {name: re.match(r'图\s*(\d+)', f.figcaption.get_text())[1] for name, f in figures.items()}
    student_table = soup.find(id='distillation-results-table')
    student_number = re.match(r'表\s*(\d+)', student_table.caption.get_text())[1]
    transfer_appendix = soup.find(id='results-transfer-details').get_text().split()[0]

    # The introductory paragraphs provide conditions; interpretations follow the plots.
    paragraphs(soup, 's3')[0].string = (
        '本文以面向品质参数化对象族学习的共享飞控策略为模型主体，统一学生承担策略部署，'
        '特权教师库提供逐对象控制经验。图 2 给出对象族、教师、统一学生与目标域模型之间的构建关系。')
    paragraphs(soup, 's51')[0].string = (
        '5,000 个对象在参数空间中的分布如图 5 所示，其中三个品质等级各 1,500 个，'
        '分布外对象为 500 个。各参数的边际分布见附录 B.1。')
    paragraphs(soup, 's52')[0].string = (
        '主试验选取 50 个源域对象开展逐对象教师训练，已有 46 个教师达到统一性能门限。'
        '图 6 给出三个品质等级代表对象的阶跃与多正弦响应，'
        '训练诊断和对象参数与误差的关系见附录 B.1。')
    paragraphs(soup, 's53')[0].string = (
        '学生先通过行为克隆学习教师动作，再进行五轮 DAgger 数据聚合。'
        f'图 {numbers["student_dagger.svg"]} 将行为克隆记为第 0 轮，随后记录第 1–5 轮聚合后的闭环结果。')
    paragraphs(soup, 's53')[0]['id'] = 'distillation-results'
    paragraphs(soup, 's531')[0].string = (
        f'图 {numbers["teacher_student_compare.svg"]} 比较主试验对象 7 上教师与学生在相同指令下的响应。'
        '误差标注采用 2 s 后的评价段。')
    paired = paragraphs(soup, 's532')
    paired[0].string = paired[0].get_text().replace(
        '共三次训练重复。', '共三次训练重复；12 个对象均参与相应学生的蒸馏。')

    student_caption = figures['student_network_tensor.svg'].figcaption
    student_caption.string = (
        f'图 {numbers["student_network_tensor.svg"]}　tiny 学生网络：32×5 历史经时序编码得到 16 维特征，'
        '与 9 维当前量和 7 维动作历史拼接为 32 维输入，经一个 32 宽残差块输出动作，共 3,856 个参数。')
    figures['topk4_diagnostic.svg'].figcaption.string = (
        f'图 {numbers["topk4_diagnostic.svg"]}　Top-k SAE 与原始历史的 180 s 闭环响应'
        '（时延对象 2，每分量 1 deg/s）。左：时域响应；右：功率谱。'
        'Top-k SAE 的低频功率较高，对应持续低频摆动与较大的 LOES 拟合残差。')
    figures['sampling_scatter.svg'].figcaption.string = (
        f'图 {numbers["sampling_scatter.svg"]}　5,000 个对象的参数空间投影。'
        '颜色区分三个品质等级与分布外对象，虚线标出所选分级边界。')
    figures['sampling_marginals.svg'].figcaption.string = (
        f'图 {numbers["sampling_marginals.svg"]}　5,000 个对象的参数边际分布。'
        '颜色表示选入等级，竖线标出相应分级或研究包络边界。')
    distribution_intro = paragraphs(soup, 'results-distribution-details')[0]
    distribution_intro.string = '图中给出八项对象参数的边际分布，配合正文的二维散点描述取值范围及其相互约束。'
    for p in soup.find_all('p'):
        if '从 IV-A 对象库的' in p.get_text():
            set_html(p, p.decode_contents().replace('从 IV-A 对象库的', '从第 3.1 节对象库的'))

    texts = {
        'closed_loop_simulation.svg':
            '学生每 20 ms 根据当前观测和历史输出动作，仿真器以 1 ms 步长推进对象状态。'
            '单条轨迹中，对象参数和策略权重固定，动作、施加力与角速度随反馈更新。'
            '按时间记录这些量就得到后文的响应曲线，其中的过冲和振荡取决于对象动态、力接口与控制策略的共同作用。',
        'pipeline_framework.svg':
            '逐对象教师将各自的控制任务转化为动作标签，蒸馏再把这些标签对应的反馈关系合并到共享权重。'
            '统一学生在源域对象上以固定权重运行；进入 DHC-6 环节后才利用适配状态更新参数。'
            '前一阶段检验跨对象控制，后一阶段考察已有策略在目标模型上的更新效果。',
        'teacher_network_tensor.svg':
            '教师的 12 维特权量包含对象参数和内部状态，使其能够直接利用当前对象的动态信息。'
            '历史分支同时保留过去的输入与响应。两类信息共同用于专用策略训练，'
            '其输出动作随后作为学生的监督标签。',
        'student_network_tensor.svg':
            '学生将 160 维历史编码为 16 维特征，再与当前量和动作历史合并后计算控制动作。'
            '部署输入中没有独立的对象参数向量或模型内部状态，跨对象响应的差异需要通过反馈历史体现。'
            '3,856 个参数对应的是一套共享策略，各对象使用同一组权重。',
        'sampling_scatter.svg':
            '时间常数—频率平面覆盖了快、慢滚转及低频模态，阻尼—频率平面则显示不同的振荡衰减条件。'
            '各等级在二维投影中仍有重叠，因为分级还涉及螺旋模态和灵敏度。'
            '部分分布外对象也落在图示坐标的常规范围内，它们的越界发生在其他参数或参数组合上。',
        'teacher_closed_loop.svg':
            '三个代表对象的阶跃跟踪 RMSE 为 0.39–0.76 deg/s，多正弦下为 0.60–1.02 deg/s。'
            '跟踪误差接近时，操纵变化仍有明显差别：二级阶跃样例的请求力持续调整，'
            '完整 30 s 的总变差达到 506.8 N。图内的 TV 是总变差，项目指标则使用除以时长后的变化率；'
            '两者分别描述整段操纵累积和单位时间内的操纵活动。',
        'student_dagger.svg':
            '追加学生访问状态上的教师标签后，误差比在前几轮下降较快，后期曲线趋于平缓。'
            'tiny 与 small 的源对象误差比分别达到 1.28、1.34；'
            '在 44 个源对象中分别有 37、38 个达标。留出对象的误差比为 1.68、1.41，两者均通过 3/4。'
            f'参数规模、控制覆盖与相对精度的完整统计见表 {student_number}。',
        'teacher_student_compare.svg':
            '正、负阶跃中，师生的主要上升过程接近，学生在 2 s 后的跟踪 RMSE 为教师的 2.3–3.3 倍。'
            '双脉冲与多正弦的比值为 1.5–1.9。'
            '逐指令响应给出瞬态与稳态行为，对象级中位统计则汇总多对象上的相对跟踪精度。',
        'cross_object_responses.svg':
            '对象 6、9 上，跨对象教师的阶跃响应出现较大过冲和摆动，统一学生与自身教师的响应更接近。'
            '初始偏差和外力扰动测试中，学生的后段残差也较小。'
            '对象 9 在 30 s 内仍可见残余偏差，恢复阶段的残余响应需与整段跟踪误差同时评价。',
        'dhc6_states.svg':
            '状态 5、8 在微调前有明显摆动，微调后振荡减弱，响应更接近目标；状态 2 的上升段也有所改善。'
            '留出的状态 9 在微调前后都低于最终目标速率。'
            '各配平状态上的更新效果通过瞬态形态和稳态误差共同描述。'
            f'逐状态误差及两档模型的对照见附录 {transfer_appendix}。',
        'delay_network.svg':
            '基线历史特征和替换前端的潜码均为 16 维，主要区别在于学习目标与训练方式。'
            '基线随控制任务联合优化，PCA、AE 和 SAE 替换前端在表征学习后冻结；'
            'SAE 辅助分支则在保留基线特征时增加潜码。后续控制结果据此比较历史信息的使用方式。',
        'delay_probe.svg':
            '1.2 s 前缀上，请求力—响应互相关的时延读出 R² 为 0.995，MAE 为 2.9 ms，'
            '输入与响应的时移包含较清晰的时延线索。16 维稠密 AE 的 R² 为 0.608，Top-k SAE 为 0.122。'
            '延长前缀后，Top-k SAE 与稠密 AE 的 R² 分别为 0.506、0.656。'
            '各表征的读出结果与后续闭环对照共同用于选择学生的历史编码方式。',
        'delay_closed_loop.svg':
            '保留基线时序编码的两种方案在长延迟层均通过 90.0%，Top-k SAE、稠密 AE 和 PCA 替换方案分别为 '
            '71.9%、63.7%、51.6%。将再训练数据限制为短、中时延后，Top-k SAE 的长延迟通过率降至 46.2%，'
            '而保留基线的方案仍为 90.0%。表征替换与训练对象覆盖共同影响了长时延控制。',
        'tau_e_eval.svg':
            '原始历史与 SAE-aux 在长时延层的可评等效延迟约为 0.30 s，高于本文采用的三级建议上限 0.25 s。'
            '两者分别有 73、102 个对象通过拟合门，Top-k SAE 为 0/120。'
            '总体中位数对应不同的可评对象集合，控制器间的延迟差异还需结合共同对象及拟合残差解释。',
        'tau_e_amplitude.svg':
            '激励幅值增加时，原始历史的可评对象数由 41 增至 91，SAE-aux 由 78 增至 111。'
            '在三档幅值共同可评的对象上，等效延迟中位数分别保持在 0.055–0.066 s 和 0.105–0.116 s。'
            '幅值对纳入统计的对象数影响较大，共同对象上的延迟量级变化较小。',
        'tau_e_seed.svg':
            '第二个种子下，原始历史的可评对象数由 73 变为 77，SAE-aux 由 102 变为 106。'
            '共同对象上的延迟中位数分别为 0.073/0.075 s、0.100/0.095 s，Top-k SAE 两个种子均无可评对象。'
            '这两次重复中，各表征方案的主要动态差异保持一致。',
        'topk4_diagnostic.svg':
            '相同激励下，Top-k SAE 分支在 180 s 内仍有持续摆动，低频段功率也高于原始历史分支。'
            '这一响应形态与其较大的频域拟合残差相对应。'
            '短时误差统计以外，长时曲线进一步显示了替换历史表征后出现的低频动态变化。',
        'sampling_marginals.svg':
            '同分布对象的时延位于 0.001–0.20 s，零点频率比位于 0.65–1.35；分布外部分延伸到这些区间之外。'
            '零点阻尼比例系数的跨度较大，是由零点阻尼除以模态阻尼得到的，低阻尼候选会产生较大的比例值。'
            '操纵效能的分布则受 1 s 灵敏度标定与效能包络共同约束。',
        'teacher_difficulty.svg':
            '达标教师中，操纵效能与跟踪 RMSE 的双对数相关系数为 −0.74，较低效能一侧的误差总体更大，'
            '未通过样例也集中在该区域。在当前力接口和训练设置下，低效能对象是误差较突出的部分；'
            '相近效能处仍有分散，其他模态和时延条件也随对象变化。',
        'teacher_training_curves.svg':
            '固定验证误差随训练总体下降，请求力总变差则呈现不同的变化过程。'
            '21 个存档教师中，11 个选择最后一个检查点，其余 10 个选择更早的权重。'
            '最终选用的控制器由验证表现确定，训练末步与所选权重并不总是相同。',
        'cross_object_matrix.svg':
            '教师 1 在自身对象 1 上通过 46/54 条测试，在对象 6 上为 22/54，显示同一专用策略换对象后的差异。'
            '学生末行给出一套共享权重在全部 12 个源对象上的结果。三次训练重复的学生整体通过率为 '
            '87.0%、63.9%、48.1%，其多对象控制表现也随训练重复变化；完整汇总见下表。',
        'dhc6_transfer.svg':
            '逐状态结果显示更新效果随配平状态变化：状态 12 的误差下降较明显，状态 9–11 的残余误差较大。'
            '留出组的中位误差与各状态的时域响应共同描述跨状态复用，'
            '并将振荡减弱和最终目标偏差区分开来。',
    }

    reuse_after = {
        'student_dagger.svg', 'dhc6_states.svg', 'delay_probe.svg',
        'delay_closed_loop.svg', 'tau_e_eval.svg', 'tau_e_amplitude.svg', 'tau_e_seed.svg',
    }
    # Move two existing lead-ins below their plots instead of repeating them.
    reuse_before = {'teacher_difficulty.svg', 'teacher_training_curves.svg'}
    for name, markup in texts.items():
        figure = figures[name]
        if name == 'closed_loop_simulation.svg':
            node = soup.find(id='closed-loop-explanation')
        elif name in reuse_after:
            node = find_following_paragraph(figure)
            assert node is not None, name
        elif name in reuse_before:
            node = figure.find_previous_sibling('p')
        else:
            node = soup.new_tag('p')
        if not node.has_attr('id'):
            node['id'] = 'analysis-' + Path(name).stem.replace('_', '-')
        node['class'] = list(node.get('class', [])) + ['figure-analysis']
        node['data-figure-source'] = name
        set_html(node, markup)
        figure.insert_after(node.extract() if node.parent else node)

    # Replace the remaining generic transition before the long response plot.
    diagnostic = figures['topk4_diagnostic.svg']
    before = diagnostic.find_previous_sibling('p')
    before.string = f'图 {numbers["topk4_diagnostic.svg"]} 给出同一时延对象上的长时响应与功率谱。'

    discussion = paragraphs(soup, 's6')
    discussion[0].string = (
        '多教师蒸馏将逐对象控制经验汇入共享策略。12 对象配对组中，统一学生的跟踪 RMSE 中位数为 '
        '0.96 deg/s，自身教师为 0.75 deg/s，跨对象教师为 3.28 deg/s。'
        '这组对照显示，知识汇聚使一套权重能够复用多个对象的反馈经验；'
        '专用教师读取新对象参数后的直接复用具有不同的控制表现。')
    discussion[1].string = (
        '学生的规模与精度呈现不同变化：small 的留出对象误差比较低，源对象上则略高于 tiny。'
        '历史表征对照也显示，较高的参数读出分数不必然对应更好的控制结果。'
        '网络规模和输入编码应结合实际闭环误差、操纵变化与恢复动态选择。')
    discussion[2].string = (
        'DHC-6 更新主要改善了适配状态的跟踪与振荡，留出状态的整体改善幅度较小，且不同状态之间存在差异。'
        '后续扩展需围绕目标动态的覆盖，以及执行机构特性和多轴耦合条件下的闭环表现展开。')

    for table in soup.find_all('table'):
        if table.find_parent(class_='parameter-table-wrap'):
            continue
        wrapper = soup.new_tag('div', attrs={'class': 'parameter-table-wrap responsive-paper-table'})
        table.wrap(wrapper)
    for image in soup.find_all('img'):
        image['alt'] = image.find_parent('figure').figcaption.get_text(' ', strip=True)
    assert len(soup.select('p.figure-analysis')) == 22


def render_supplement_math(soup):
    spans = soup.select('.supplement-math[data-tex]')
    expressions = [{'tex': span['data-tex'], 'display': False} for span in spans]
    rendered = json.loads(subprocess.run(
        ['node', str(ROOT / 'paper_source/render_math.cjs')], input=json.dumps(expressions),
        text=True, capture_output=True, check=True).stdout)
    for span, markup in zip(spans, rendered):
        span['class'] = ['math-inline', 'supplement-math']
        set_html(span, markup)
    return len(spans)
