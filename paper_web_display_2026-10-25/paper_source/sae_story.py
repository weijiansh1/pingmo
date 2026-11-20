"""Connect the existing representation experiments to the unified-controller design."""
from bs4 import BeautifulSoup


TITLES = {
    's61': '6.1 历史信息如何支持统一控制',
    's611': '6.1.1 表征对照设计',
    's612': '6.1.2 历史中的时延信息',
    's613': '6.1.3 历史压缩后的闭环性能',
    's614': '6.1.4 从输入时延到闭环等效延迟',
    's615': '6.1.5 表征选择对闭环动态的影响',
    's616': '6.1.6 激励幅值与训练重复检验',
}

PARAGRAPHS = {
    's61': [
        '统一学生在部署时不接收教师使用的对象参数，而是依靠当前观测和历史响应控制不同对象。第 5 节已检验其闭环能力；本节进一步考察这一输入设计：历史中包含哪些对象动态线索，显式提取这些线索能否改善统一控制？选择输入运输时延 <i>τ</i><sub>p</sub> 作为分析变量，是因为它直接改变控制作用与对象响应的时间关系，对反馈控制和振荡抑制都有影响。',
        '为此，在已蒸馏的 tiny 学生基础上比较原始历史、稠密编码和稀疏自编码器（SAE）。分析依次回答三个问题：时延信息能否从历史中读出，压缩表征能否维持闭环性能，以及表征改变后为何出现长时响应差异。SAE 在这里用于检验学生的历史输入设计。',
    ],
    's611': [
        '从 IV-A 对象库的 train_core、validation、id_test 三个互不重叠的划分中，按 <i>τ</i><sub>p</sub> 的短、中、长三层分别抽取 100、30、40 架，共 510 架。每架运行 8 条指令，包括 +1、+2、+5、+10、+21、+31 deg/s 阶跃、双脉冲和多正弦；由既有 tiny 学生闭环驱动，记录 32×5 历史、当前量、旧动作和动作，共 489,600 个窗口。留出集为 120 架、960 个对象—指令 episode。训练、验证和测试按对象划分。',
        '图 16 给出两种使用历史的方式：主线学生保留原始历史；表征分支将 160 维展平历史压缩为 16 维潜码。SAE 编码器为 160→64→16，解码器对称，分别采用 L1 稀疏和 top-k 稀疏约束。编码器训练后冻结，再接线性时延探针或策略头。对照包括原始历史、PCA-16、稠密 AE-16、36 维请求力—响应互相关和 5 维通道均值。',
        '时延探针以对象中预先采样的 <i>τ</i><sub>p</sub> 为标签，它对应输入前的纯运输延迟 <i>e</i><sup>−τ<sub>p</sub>s</sup>。探针仅使用截至当前时刻的历史，从 1.2 s 起对已有窗口作运行平均，再用线性回归预测时延。这一读出检验先判断历史是否包含相关信息；随后用闭环对照检验该信息能否被控制器有效利用。',
    ],
    's612': [
        '原始 tiny 学生在短、中、长延迟层的闭环通过率分别为 94.0%、93.5% 和 88.8%，中位 RMSE 分别为 0.343、0.363 和 0.438 deg/s，长延迟层表现较弱。表 6 和图 17 考察这一差异对应的时延信息是否能从历史中读出。',
        '请求力—响应互相关在 1.2 s 因果前缀上达到 R² = 0.995、MAE = 2.9 ms，说明历史中的输入—响应时移提供了清晰的时延线索。原始历史的线性读出 R² 为 0.567，稠密 AE-16 为 0.608，PCA-16 为 0.496。相同的 16 维输出下，L0≈7 的 L1-SAE 为 0.344，Top-k SAE（k=4）为 0.122，L0≈3.5 的 L1-SAE 为 0.075；当前稀疏编码保留的线性可读出时延信息更少。',
        '这一差异随观测时间变化：例如 Top-k SAE 的 R² 从 1.2 s 的 0.122 升至 2.0 s 的 0.506。因而输入设计既取决于压缩方式，也取决于信息何时可用。对学生控制器而言，接下来的关键是这些表征在持续反馈中的效果，而非单独追求探针分数。',
        '本组读出结果覆盖 510 架对象、0–0.2 s 时延范围，采用 16 维编码与线性探针。它支持使用历史获取对象动态信息，并为下一组闭环实验提供表征对照。',
    ],
    's613': [
        '在同一 510 架对象划分上，将表征用于控制：PCA-16、稠密 AE-16 和 Top-k SAE-16 替换原始历史，SAE-aux 则保留原始历史并附加 SAE 潜码。动作标签由既有 tiny 学生产生，检验的是对已蒸馏控制能力的保留。原始历史微调分支与 SAE-aux 继承 tiny 权重；替换分支训练新的策略头。各分支先用 BC 数据训练，替换分支和 SAE-aux 再通过 DAgger 追加自身访问状态：在 96 架源域对象上以 β = 0.5 采样，合并数据后再训练 8 个 epoch。',
        '表 7 和图 18 显示，保留原始历史的 raw_ft 在短、中、长延迟层通过率为 93.1%、95.3%、90.0%，SAE-aux 为 93.1%、95.6%、90.0%。替换历史后，PCA-16 为 60.0%、60.3%、51.6%，稠密 AE-16 为 69.1%、72.5%、63.7%，Top-k SAE-16 为 85.6%、79.7%、71.9%。附加稀疏潜码维持了基线表现，直接替换历史的三种方案均出现下降。',
        '探针分数与控制性能的排序也不同：Top-k SAE 的早期时延读出弱于稠密 AE，但在替换类中闭环通过率更高。由此可见，线性时延可读出性并不足以决定控制效果；策略还要利用瞬态、幅值和动作历史等信息。当前结果支持主线学生继续保留原始历史，SAE-aux 尚未显示通过率收益。',
        '进一步仅用短、中延迟样本（<i>τ</i><sub>p</sub> ≤ 0.15 s）再训练，再到长延迟留出对象上评估：raw 与 SAE-aux 均为 90.0%，PCA 为 47.5%，Top-k SAE 从全延迟训练的 71.9% 降至 46.2%。这一对照反映限制再训练样本后的变化；raw 与 SAE-aux 保留了既有 tiny 权重中的知识。',
        'PCA 与稠密 AE 各重复一次，长延迟通过率分别落在 51.6–55.6% 与 52.5–63.8%，两次运行均低于 raw；raw 与 SAE-aux 的 90% 保持稳定。下一步转向长时闭环响应，检验短时通过率之外的动态差异。',
    ],
    's614': [
        '闭环性能下降可能表现为滞后加大，也可能表现为持续振荡。为区分这两种情况，需要将对象输入时延 <i>τ</i><sub>p</sub> 与完整受控系统的等效时间延迟 <i>τ</i><sub>e</sub> 联系起来：前者在对象生成时采样，后者在控制器确定后由闭环输入输出拟合得到。同一对象采用不同控制器，可以产生不同的等效延迟。',
        'GJB 2874-1997 A5.4.1.1.5 对滚转轴等效时间延迟的建议上限为 1 级 0.10 s、2 级 0.20 s、3 级 0.25 s（表 A21，印刷 246–247 页，PDF 250–251 页），考察前置滤波器、SAS、舵机等综合作用；A4.9（印刷 68 页，PDF 72 页）说明时间延迟项用于改善高频拟合。本文据此分析各表征策略形成的闭环动态。',
        '拟合采用滚转轴 4 阶低阶等效系统（LOES），包含滚转、螺旋与荷兰滚模态。其分母为 (s−λ<sub>s</sub>)(s+1/T<sub>R</sub>)(s²+2ζ<sub>d</sub>ω<sub>d</sub>s+ω<sub>d</sub>²)，分子为 s(s²+2ζ<sub>φ</sub>ω<sub>φ</sub>s+ω<sub>φ</sub>²)，并带有延迟项 e<sup>−τ<sub>e</sub>s</sup>；对应 GJB A5.4.1.1.1（印刷 210–212 页，PDF 214–216 页）的滚转速率模态结构。',
        '两类时延在实验中的用途见表 9：<i>τ</i><sub>p</sub> 用于构造对象和分层分析，<i>τ</i><sub>e</sub> 用于描述控制策略形成的整体响应。对闭环频响先检查 4 阶 LOES 的拟合残差，再报告等效延迟，便可同时识别“响应滞后”和“响应已难以用该线性结构描述”两类现象。',
        '因此，前面的探针检验学生能否获取对象时延信息；下面的 LOES 分析则检验不同表征策略最终产生了怎样的闭环动态。二者共同服务于统一控制器的输入选择。',
    ],
    's615': [
        '在 120 架留出对象（短、中、长各 40 架）上，对原始历史、Top-k SAE-16 和 SAE-aux 注入 0.1–10 rad/s 的 18 分量多正弦滚转指令，记录 180 s 闭环响应。以 50 Hz 输入输出估计 H(jω) = P/P<sub>c</sub>，再拟合 4 阶滚转 LOES 和 <i>τ</i><sub>e</sub>，沿用各策略既有闭环及其记录信号。',
        '以幅值 VAF ≥ 0.95、相位 RMSE ≤ 10° 作为本实验的拟合门限。通过拟合门的对象按 A21 延迟阈值分档，其余单列为未通过拟合。表 10 和图 19 同时报告可评数和等效延迟分布。',
        '原始历史与 SAE-aux 的等效延迟随对象时延层增加：前者短、中、长层的可评中位数为 0.042、0.127、0.299 s，后者为 0.032、0.125、0.309 s。两者长延迟层均约 0.30 s，高于 A21 的 3 级建议上限 0.25 s，说明保留短时跟踪能力后，长延迟对象的整体响应仍有改进空间。',
        '更明显的差异出现在 Top-k SAE-16：原始历史有 73/120 架通过拟合，SAE-aux 有 102/120 架，Top-k SAE 为 0/120。其 VAF 中位为 0.824、相位 RMSE 中位为 35.1°。结合图 22 的 180 s 响应和功率谱，可见持续低频振荡；这一变化超出了单一延迟数值所能描述的范围。',
        '通过拟合门的对象中，原始历史按 1、2、3、超过 3 级阈值分档的数量为 43、15、4、11，SAE-aux 为 51、14、7、30。SAE-aux 的可评集合更大，因而两组总体中位数 0.073 s 与 0.102 s 对应不同对象集合；比较控制器的延迟差异还需考察共同可评对象。',
        '由此，闭环表征的判断需要同时看短时跟踪、长时振荡和频域拟合。SAE-aux 保留了原始历史的短时通过率，并提高了 LOES 可评数；Top-k SAE 直接替换历史后，则暴露出短时统计未充分体现的长时动态问题。下面检查这一现象对激励幅值与训练重复是否稳定。',
    ],
    's616': [
        '首先在同一 120 架对象、3 种策略上，将多正弦每分量幅值设为 0.5、1.0、2.0 deg/s，保持拟合协议与门限一致。表 11 和图 20 同时比较可评数及三档幅值共同可评对象上的等效延迟。',
        '随着幅值增加，原始历史可评数为 41、73、91，SAE-aux 为 78、102、111；在三档幅值共同可评的 40 架原始历史对象和 75 架 SAE-aux 对象上，<i>τ</i><sub>e</sub> 中位分别为 0.055–0.066 s、0.105–0.116 s，变化约 0.01 s。激励幅值明显影响可评覆盖率，共同对象上的延迟量级则较稳定。',
        'Top-k SAE-16 在三档幅值下均为 0/120 可评，VAF 中位分别为 0.732、0.824、0.807，相位 RMSE 中位分别为 52.6°、35.1°、40.9°。改变激励幅值后，其闭环响应仍难以由当前 4 阶 LOES 描述。',
        '将相位门限取为 8°、10°、12°，原始历史可评数为 53、73、84，<i>τ</i><sub>e</sub> 中位为 0.071、0.073、0.090 s；SAE-aux 为 81、102、110 架和 0.100、0.102、0.098 s；Top-k SAE 均为 0。门限影响纳入统计的对象数，但上述动态差异保持一致。',
        '其次，三种策略各用第二种子按原 BC/DAgger 配置训练，再进行同一闭环评价，结果见表 12 和图 21。',
        '原始历史可评数由 73 变为 77，<i>τ</i><sub>e</sub> 中位由 0.073 s 变为 0.078 s，共同可评 73 架的中位为 0.073 / 0.075 s，分档数量为 43/15/4/11 → 43/15/5/14。SAE-aux 可评数由 102 变为 106，中位为 0.102 / 0.106 s，共同可评 101 架的中位为 0.100 / 0.095 s，分档数量为 51/14/7/30 → 52/15/7/32。Top-k SAE 两个种子均为 0/120 可评。',
        '两个种子和三档幅值下，Top-k SAE 的长时动态问题均出现；图 22 给出同一对象上的响应实例。综合探针与闭环结果，历史确实提供了对象时延线索，但当前 SAE 压缩方式没有改善统一控制。主线因此保留原始历史：后续表征设计应以维持跟踪、抑制振荡和跨对象控制能力为目标，再检验可解释特征是否带来额外收益。',
    ],
}


def section_nodes(soup, ident):
    heading = soup.find(id=ident)
    result = []
    for node in heading.next_siblings:
        if getattr(node, 'name', None) in ('h2', 'h3', 'h4'):
            break
        if getattr(node, 'name', None):
            result.append(node)
    return heading, result


def rewrite_sae_story(soup):
    """Rewrite only SAE prose/headings; retain every measured table cell and figure."""
    before_data = [[td.get_text() for td in t.select('tbody td')] for t in soup.find_all('table')]
    before_images = [img['src'] for img in soup.find_all('img')]
    for ident, paragraphs in PARAGRAPHS.items():
        heading, nodes = section_nodes(soup, ident)
        old = [node for node in nodes if node.name == 'p']
        while len(old) < len(paragraphs):
            extra = soup.new_tag('p')
            old[-1].insert_after(extra)
            old.append(extra)
        assert len(old) == len(paragraphs), (ident, len(old), len(paragraphs))
        for paragraph, content in zip(old, paragraphs):
            paragraph.clear()
            paragraph.append(BeautifulSoup(content, 'html.parser'))
        heading.string = TITLES[ident]
        toc = soup.select_one(f'.toc a[href="#{ident}"]')
        if toc:
            toc.string = TITLES[ident]
    # Name the restriction at the stage where it actually applies: these
    # inherited tiny weights were not trained from scratch on short delays.
    _, nodes = section_nodes(soup, 's613')
    table = next(node for node in nodes if node.name == 'table')
    table.find_all('th')[-1].string = '仅短/中延迟再训练 → 长延迟'
    table.caption.string = '表 7　历史表征的闭环性能与短/中延迟再训练对照'
    for node in nodes:
        if node.name == 'figure':
            node.figcaption.string = node.figcaption.get_text().replace('仅用短/中延迟训练后', '仅用短/中延迟样本再训练后').replace('与全延迟训练的对比', '与全延迟样本再训练的对比')
    _, nodes = section_nodes(soup, 's614')
    next(n for n in nodes if n.name == 'table').caption.string = '表 9　对象输入时延与闭环等效延迟的定义及用途'
    _, nodes = section_nodes(soup, 's615')
    next(n for n in nodes if n.name == 'table').caption.string = '表 10　不同历史表征策略的闭环 LOES 拟合与等效延迟'
    for node in nodes:
        if node.name == 'figure':
            node.figcaption.string = node.figcaption.get_text().replace('GJB τe 首版评价', '不同历史表征策略的闭环等效延迟')
    assert before_data == [[td.get_text() for td in t.select('tbody td')] for t in soup.find_all('table')]
    assert before_images == [img['src'] for img in soup.find_all('img')]
