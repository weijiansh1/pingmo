"""Place the history-representation ablation inside the controller argument."""
from collections import Counter
import re

from bs4 import BeautifulSoup


METHOD = [
    '多教师蒸馏同时包含两种变化：将多个对象的控制经验合并到一套权重，并撤去教师使用的对象参数和内部状态。部署学生因而需要利用可观测的输入—响应历史区分对象动态。基线学生将 32×5 历史送入随控制任务联合训练的时序编码器，得到 16 维历史特征。历史中的对象信息能否被有效利用，是统一控制器输入设计需要检验的问题。',
    '为检验显式表征学习能否改善对象动态信息的提取，设置 PCA、稠密自编码器和稀疏自编码器（SAE）前端对照。SAE 检验少数活跃特征能否集中保留对象动态线索，稠密编码用于比较稀疏约束的影响。SAE 采用 160→64→16 编码器和对称解码器，以 L1 或 top-k 约束限制潜码中的活跃分量。替换分支用冻结前端输出的 16 维特征替代基线时序特征；辅助分支在保留基线特征时再附加 SAE 潜码。各替换前端的历史特征维度均为 16，用于比较不同表征学习方式。',
    '评价分为信息读出和闭环控制两步：以对象输入运输时延 <i>τ</i><sub>p</sub> 为标签，检验因果历史中能否读出对象动态线索；再将相同表征接入策略，比较跟踪与恢复过程。探针和控制结果共同用于判断输入设计。该消融在既有 tiny 学生上进行，动作标签由该学生产生；原始历史与辅助输入分支继承其权重，替换分支训练新的策略头。完整网络和训练设置见<a href="#s611">附录 A.1</a>。',
]

RESULTS = [
    '前述跨对象测试显示，统一学生保留了多对象控制能力，但其误差仍随对象动态变化。本节按第 3.3.1 节的设计，检验由历史重构学习得到的冻结表征，能否替代面向控制任务学习的时序特征。采用 510 架分层对象和 489,600 个历史窗口，按对象划分训练、验证与测试，其中 120 架对象用于留出评价。',
    '首先，历史中确实含有时延线索：1.2 s 因果前缀上的请求力—响应互相关达到 R² = 0.995、MAE = 2.9 ms；稠密 AE-16 的 R² 为 0.608，Top-k SAE（k=4）为 0.122。当前稀疏编码保留的线性可读出时延信息较少，完整探针对照见<a href="#s612">附录 A.2</a>。',
    '闭环对照见图 18。在长延迟层，保留原始历史的学生通过率为 90.0%，附加 SAE 潜码后仍为 90.0%；直接替换历史时，PCA-16、稠密 AE-16、Top-k SAE-16 分别为 51.6%、63.7%、71.9%。Top-k SAE 的闭环表现优于其他替换方案，但其时延探针分数更低，说明单个对象参数的可读出性不足以决定控制效果。',
    '这组结果支持当前统一学生保留由控制任务学习的历史编码分支：现有 SAE 替换方案损失了部分控制能力，辅助输入方案维持基线表现，尚未带来通过率收益。历史特征的取舍需要以闭环表现为依据。完整训练对照及长时闭环诊断见<a href="#s613">附录 A.3–A.6</a>。',
]

APPENDIX_TITLES = {
    's611': 'A.1 历史表征的数据与训练设置',
    's612': 'A.2 时延信息读出的完整结果',
    's613': 'A.3 历史表征的完整闭环对照',
    's614': 'A.4 输入时延与闭环等效延迟',
    's615': 'A.5 长时闭环频响与 LOES 诊断',
    's616': 'A.6 激励幅值与训练重复检查',
}


def paragraph(soup, content):
    p = soup.new_tag('p')
    p.append(BeautifulSoup(content, 'html.parser'))
    return p


def renumber_figures(soup):
    mapping = {}
    for number, figure in enumerate(soup.find_all('figure'), 1):
        caption = figure.find('figcaption')
        original = int(re.match(r'图\s*(\d+)', caption.get_text())[1])
        mapping[original] = number
    pattern = re.compile(r'图\s*(\d+)(?!\d)')
    def replace(match):
        old = int(match[1])
        return f'图 {mapping[old]}' if old in mapping else match[0]
    for node in list(soup.find_all(string=pattern)):
        if node.parent.name not in ('script', 'style'):
            node.replace_with(pattern.sub(replace, str(node)))
    for image in soup.find_all('img'):
        if image.has_attr('alt'):
            image['alt'] = pattern.sub(replace, image['alt'])
    return mapping


def integrate_sae(soup):
    """Keep all measured data; move only the argument and its supporting material."""
    images_before = Counter(i['src'].split('?')[0] for i in soup.find_all('img'))
    tables_before = Counter(tuple(c.get_text() for c in t.select('tbody td')) for t in soup.find_all('table'))

    start = soup.find(id='s61')
    end = soup.find(id='s62')
    block = []
    for node in [start, *start.next_siblings]:
        if node is end:
            break
        if getattr(node, 'name', None):
            block.append(node)
    assert sum(node.name == 'figure' for node in block) == 7
    closed_loop = next(n for n in block if n.name == 'figure'
                       and 'delay_closed_loop.svg' in n.img['src'])
    for node in block:
        node.extract()

    # Introduce the reason for the ablation at the student interface itself.
    transfer_method = soup.find(id='s34')
    method = soup.new_tag('h4', id='s331')
    method.string = '3.3.1 历史表征与稀疏编码对照'
    transfer_method.insert_before(method)
    for content in METHOD:
        transfer_method.insert_before(paragraph(soup, content))
    student_diagram = next(f for f in soup.find_all('figure')
                           if 'student_network_tensor.svg' in f.img['src'])
    student_diagram.figcaption.string = student_diagram.figcaption.get_text().replace(
        '延迟表征实验的 SAE 分支见图 16。', '')

    # The original template repeats s54 for PID and DHC-6. Give each its own
    # anchor before inserting the ablation at the end of the student results.
    duplicates = soup.find_all(id='s54')
    pid = next(h for h in duplicates if h.get_text().startswith('5.3.4'))
    pid['id'] = 's534'
    transfer_result = next(h for h in duplicates if h is not pid)
    alias = soup.new_tag('span', id='s61')
    transfer_result.insert_before(alias)
    result_heading = soup.new_tag('h3', id='s535')
    result_heading.string = '5.3.5 历史表征方式的消融'
    transfer_result.insert_before(result_heading)
    for index, content in enumerate(RESULTS):
        transfer_result.insert_before(paragraph(soup, content))
        if index == 2:
            transfer_result.insert_before(closed_loop)

    # Appendix material retains all six tables and every remaining original
    # figure. The main text has one compact result, rather than a second study.
    appendix = soup.new_tag('h2', id='appendix-history')
    appendix.string = '附录 A　历史表征对照与闭环动态诊断'
    soup.main.append(appendix)
    soup.main.append(paragraph(soup, '本附录给出第 3.3.1 节与第 5.3.5 节历史表征消融的完整设置、探针结果和闭环诊断，便于核对输入设计的依据。'))
    in_details = False
    for node in block:
        ident = node.get('id')
        if ident in APPENDIX_TITLES:
            in_details = True
            node.name = 'h3'
            node.string = APPENDIX_TITLES[ident]
        if in_details and node is not closed_loop:
            soup.main.append(node)
            if ident == 's611':
                soup.main.append(paragraph(soup, '基线的时序编码器与控制策略联合训练，将 160 维历史映射为 16 维特征，再与 9 维当前量、7 维旧动作拼接。替换前端也输出 16 维并冻结，只训练后续策略头；辅助前端则在基线特征之外增加 16 维潜码。该对照比较的是表征学习方式及其训练配置。'))

    # The overall discussion returns to the controller route. Detailed LOES
    # qualifications now accompany their measurements in the appendix.
    discussion = soup.find(id='s6').find_next_sibling('p')
    discussion.append(' 历史表征消融进一步表明，当前 SAE 压缩没有改善闭环性能，保留原始历史仍是本实现中更合适的输入选择。')
    end.string = '6.1 后续验证：执行机构与完整回路'
    conclusion = soup.find(id='s7').find_next_siblings('p')
    conclusion[0].string = conclusion[0].get_text().replace('逐对象教师给出上界', '逐对象教师提供监督').replace(
        '压缩不损精度', '小型学生保留主要控制能力')
    conclusion[1].clear()
    conclusion[1].append(BeautifulSoup(
        '历史表征对照检验了部署学生的输入设计：输入—响应历史包含可读出的对象时延线索，但当前 SAE 压缩没有提高闭环通过率，统一学生继续保留原始历史。长时响应中的残余振荡还说明，跟踪门限与抑振性能需要分别检验。', 'html.parser'))
    related = soup.find(id='s11')
    related_paragraphs = []
    for node in related.next_siblings:
        if getattr(node, 'name', None) in ('h2', 'h3'):
            break
        if getattr(node, 'name', None) == 'p':
            related_paragraphs.append(node)
    related_paragraphs[-1].clear()
    related_paragraphs[-1].string = '本文将飞行品质分级用于构造对象族，以逐对象强化学习教师形成控制经验，再经多轮蒸馏得到统一学生。评价围绕三个问题展开：统一策略能保留多少教师能力，输入历史能否支持不同对象的控制，以及性能随指令幅值和对象分布如何变化。容量、对象覆盖和历史表征对照用于分析学生的误差来源。'

    for node in list(soup.find_all(string=True)):
        text = str(node)
        for old, new in [('6.1.1', '附录 A.1'), ('6.1.2', '附录 A.2'), ('6.1.3', '附录 A.3'),
                         ('6.1.4', '附录 A.4'), ('6.1.5', '附录 A.5'), ('6.1.6', '附录 A.6')]:
            text = text.replace(old, new)
        if text != str(node) and node.parent.name not in ('script', 'style'):
            node.replace_with(text)

    figure_numbers = renumber_figures(soup)
    toc = soup.select_one('.toc')
    for a in list(toc.find_all('a')):
        a.decompose()
    abstract = soup.new_tag('a', href='#abstract'); abstract.string = '摘要'; toc.append(abstract)
    for heading in soup.main.find_all(['h2', 'h3', 'h4'], recursive=False):
        ident = heading.get('id')
        if not ident or (heading.name == 'h4' and ident != 's331') or ident in APPENDIX_TITLES:
            continue
        a = soup.new_tag('a', href='#' + ident)
        if heading.name in ('h3', 'h4'):
            a['class'] = 'sub'
        a.string = heading.get_text(' ', strip=True)
        toc.append(a)

    assert images_before == Counter(i['src'].split('?')[0] for i in soup.find_all('img'))
    assert tables_before == Counter(tuple(c.get_text() for c in t.select('tbody td')) for t in soup.find_all('table'))
    assert soup.find(id='s331') and soup.find(id='s535') and soup.find(id='appendix-history')
    return figure_numbers
