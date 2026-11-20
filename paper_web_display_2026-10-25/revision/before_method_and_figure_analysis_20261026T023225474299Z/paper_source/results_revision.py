"""Keep section 5 focused on the evidence for a shared flight-control policy."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

from bs4 import BeautifulSoup, Comment

from integrate_sae import renumber_figures


ROOT = Path(__file__).resolve().parents[1]


def section_nodes(soup, ident):
    result = []
    for node in soup.find(id=ident).next_siblings:
        if getattr(node, 'name', None) in ('h2', 'h3', 'h4'):
            break
        if getattr(node, 'name', None):
            result.append(node)
    return result


def paragraph(soup, markup):
    node = soup.new_tag('p')
    node.append(BeautifulSoup(markup, 'html.parser'))
    return node


def rebuild(soup, ident, content):
    for node in section_nodes(soup, ident):
        node.extract()
    anchor = soup.find(id=ident)
    for item in content:
        node = paragraph(soup, item) if isinstance(item, str) else item.extract()
        anchor.insert_after(node)
        anchor = node


def renumber_tables(soup):
    mapping = {}
    for number, table in enumerate(soup.find_all('table'), 1):
        mapping[int(re.match(r'表\s*(\d+)', table.caption.get_text())[1])] = number
    pattern = re.compile(r'表\s*(\d+)(?!\d)')
    for node in list(soup.find_all(string=pattern)):
        if node.parent.name not in ('script', 'style'):
            node.replace_with(pattern.sub(lambda m: f'表 {mapping[int(m[1])]}'
                                         if int(m[1]) in mapping else m[0], str(node)))
    return mapping


def remove_dagger_title():
    """Remove a numerical title contradicted by the first aggregated points."""
    source = ROOT.parent / 'flight_control_original_restored_2026-10-25/webpage/paper/student_dagger.svg'
    original = source.read_bytes()
    before = BeautifulSoup(original, 'xml')
    after = BeautifulSoup(original, 'xml')
    label = after.find(string=lambda node: isinstance(node, Comment)
                       and '一轮 DAgger' in str(node))
    assert label is not None
    old_title = str(label).strip()
    title_group = label.parent.find('g', recursive=False)
    uses = title_group.find_all('use')
    removed = len(uses)
    for use in uses:
        use.decompose()
    label.replace_with(Comment(' DAgger 蒸馏过程；说明见图注 '))
    assert [dict(p.attrs) for p in before.find_all('path')] == [dict(p.attrs) for p in after.find_all('path')]
    target = ROOT / 'webpage/paper/student_dagger.svg'
    target.write_text(str(after), encoding='utf-8')
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    (ROOT / 'revision/dagger_title_correction.json').write_text(json.dumps({
        'original_title': old_title,
        'reason': '首轮聚合的两个误差比点未同时低于 2；删除内部结论式标题，数值由曲线和正文表述。',
        'source_sha256': hashlib.sha256(original).hexdigest(),
        'display_sha256': digest,
        'all_curve_marker_axis_and_glyph_paths_unchanged': True,
        'removed_title_glyph_uses': removed,
    }, ensure_ascii=False, indent=2) + '\n')
    return digest


def revise_results_section(soup):
    images_before = Counter(Path(i['src'].split('?')[0]).name for i in soup.find_all('img'))
    tables_before = Counter(tuple(c.get_text() for c in t.select('tbody td')) for t in soup.find_all('table'))
    equations_before = [str(e) for e in soup.select('.eq')]
    math_before = Counter(a.get_text() for a in soup.select('annotation'))
    figures = {Path(f.img['src'].split('?')[0]).name: f for f in soup.find_all('figure')}
    number = {name: int(re.match(r'图\s*(\d+)', f.figcaption.get_text())[1])
              for name, f in figures.items()}
    old = {key: [n for n in section_nodes(soup, key) if n.name == 'p']
           for key in ['s52', 's532', 's534', 's54']}
    main_student_table = soup.find(id='distillation-results-table-wrap')
    paired_table = soup.find(id='cross-object-results')
    pid_table = soup.find(id='pid-student-comparison')
    dhc_tables = [n for n in section_nodes(soup, 's54') if n.name == 'table']
    assert len(dhc_tables) == 2
    dhc_tables[0]['id'] = 'dhc6-adaptation-summary'
    dhc_tables[1]['id'] = 'dhc6-capacity-comparison'

    # The historical aliases remain valid after the two scan subsections merge.
    pid_alias = soup.find(id='s534')
    pid_alias.name = 'span'
    pid_alias.clear()
    history_alias = soup.find(id='s61')
    history_alias.extract()
    soup.find(id='s535').insert_before(history_alias)

    # Move the complete history plot next to its detailed numerical comparison.
    detail_table = next(n for n in section_nodes(soup, 's613') if n.name == 'table')
    detail_table.insert_before(figures['delay_closed_loop.svg'].extract())

    # Supplementary material retains the numerical evidence and the protocols.
    appendix = soup.new_tag('h2', id='appendix-results')
    appendix.string = '附录 B　对象覆盖、教师学习与目标域补充结果'
    soup.main.append(appendix)
    for ident, title in [
        ('results-distribution-details', 'B.1 对象分布与教师学习诊断'),
        ('results-cross-object-details', 'B.2 跨对象评估与补充协议'),
        ('results-transfer-details', 'B.3 目标域逐状态结果与模型规模对照'),
        ('results-pid-comparison', 'B.4 固定增益 PID 对照'),
    ]:
        heading = soup.new_tag('h3', id=ident)
        heading.string = title
        soup.main.append(heading)
        if ident == 'results-distribution-details':
            content = [
                '对象参数的边际分布及教师选取位置如下，配合正文参数空间散点描述学习对象的覆盖。',
                figures['sampling_marginals.svg'],
                old['s52'][1], figures['teacher_difficulty.svg'],
                old['s52'][2], figures['teacher_training_curves.svg'],
            ]
        elif ident == 'results-cross-object-details':
            coverage = [p for p in old['s532'] if p.get_text().startswith((
                '主试验 tiny 学生在 120', '在新增对象中，'))]
            source_links = next(p for p in old['s532'] if '补充数据：' in p.get_text())
            content = [
                figures['cross_object_matrix.svg'], paired_table,
                '12 对象配对组包含三次训练重复，每次为 12 个专用教师及其统一学生。自身教师和学生各有 648 个对象—指令组合，跨对象教师有 7,128 个组合。响应非有限或角速度绝对值超过 300 deg/s 时记为失败；通过率按全部组合、RMSE 中位数按有限轨迹计算。',
                '三组重复中，学生通过率分别为 87.0%、63.9%、48.1%，自身教师分别为 82.4%、61.1%、61.1%。时域样例按训练阶段独立验证记录选择：第一组 8 个教师达标，每个品质等级依原对象顺序取首个，得到对象 1、6、9，跨对象教师为 6、9、1。选样及验证记录见<a href="evidence/cross_object_20261025/illustration_protocol_v2.json">时域对照协议</a>，完整诊断响应见<a href="paper/cross_object_diagnostic.svg">闭环诊断图</a>。',
                source_links,
                '主试验的补充对象覆盖采用另一批 120 个同分布对象，按对象汇总指标评价：',
                *coverage,
            ]
        elif ident == 'results-transfer-details':
            content = [
                'small 学生在 12 个 DHC-6 辨识模型上的逐状态更新结果如下；8 个状态用于适配，4 个状态用于留出评价。两张表分别汇总 small 的状态分组结果及 tiny、small 的模型规模对照。',
                figures['dhc6_transfer.svg'], *dhc_tables, old['s54'][1],
            ]
        else:
            content = [
                pid_alias,
                '在 300 个同分布对象和 500 个分布外对象上，比较统一学生与固定增益二自由度 PID。300 个同分布对象从主试验 48 个评估对象之外抽取；两者采用相同的 14 条输入配置与控制接口，每个对象—指令组合重复两次。学生权重与 PID 增益在扫描期间保持固定。',
                old['s534'][0],
                pid_table,
            ]
        for item in content:
            soup.main.append(paragraph(soup, item) if isinstance(item, str) else item.extract())

    rebuild(soup, 's51', [
        '5,000 个品质参数化对象覆盖操纵效能、模态和输入时延的差异，其中 4,500 个用于训练及同分布评估，500 个用于分布外评估。参数空间散点展示不同品质等级与数据划分的分布，参数边际分布见<a href="#results-distribution-details">附录 B.1</a>。',
        figures['sampling_scatter.svg'],
    ])
    rebuild(soup, 's52', [
        f'主试验选取 50 个源域对象开展逐对象教师训练，已有 46 个教师达到统一性能门限。图 {number["teacher_closed_loop.svg"]} 展示三个品质等级代表对象的阶跃与多正弦响应，同时给出角速度跟踪和操纵量变化，形成多教师蒸馏的控制经验来源。训练收敛曲线及对象参数与误差的关系见<a href="#results-distribution-details">附录 B.1</a>。',
        figures['teacher_closed_loop.svg'],
    ])
    rebuild(soup, 's53', [
        f'多轮 DAgger 将逐对象教师经验汇聚到一组共享权重中。图 {number["student_dagger.svg"]} 显示，学生与教师的误差比从初始行为克隆时的数十倍下降到后期约 1.3。tiny 学生以 3,856 个参数在 44 个蒸馏源对象中通过 37 个，在 4 个留出对象中通过 3 个。',
        figures['student_dagger.svg'],
        '两档学生的参数规模、达标数和师生闭环跟踪误差比见表 5。tiny 与 small 的源对象误差比分别为 1.28、1.34，留出对象误差比分别为 1.68、1.41。两档实现提供了不同的参数规模与相对跟踪精度组合。',
        main_student_table,
    ])
    rebuild(soup, 's531', [
        f'图 {number["teacher_student_compare.svg"]} 比较同一源对象上教师与学生的响应。六档阶跃的上升过程接近，学生在 2 s 后的跟踪 RMSE 为教师的 2.3–3.3 倍；双脉冲与多正弦上的比值为 1.5–1.9。该时域对照与表 5 的对象级统计共同呈现蒸馏后的瞬态过程和跟踪精度。',
        figures['teacher_student_compare.svg'],
    ])
    rebuild(soup, 's532', [
        '采用 12 对象配对组比较专用教师与统一学生，每个品质等级含 4 个对象，共三次训练重复。每个教师依次控制全部对象，学生控制相同对象；每个控制器在每个对象上测试 18 条指令，时域为 8 s。跨对象使用的教师读取当前被控对象的参数与状态，学生依靠常规观测和历史；所有权重固定。',
        '在有限轨迹上，统一学生的跟踪 RMSE 中位数为 0.96 deg/s，自身教师为 0.75 deg/s，教师跨对象使用时为 3.28 deg/s。统一学生以一组共享权重，在多个源对象上保持了与专用教师相近的跟踪误差量级；相较直接跨对象使用专用教师，共享策略获得了更低的跟踪误差。',
        f'图 {number["cross_object_responses.svg"]} 在同一被控对象及相同指令、初始偏差和外力扰动下，对照自身教师、跨对象教师与统一学生的跟踪和恢复过程。样例按训练阶段的独立验证记录选取，完整规则、逐控制器结果和补充对象覆盖见<a href="#results-cross-object-details">附录 B.2</a>。',
        figures['cross_object_responses.svg'],
    ])
    soup.find(id='s533').string = '5.3.3 对象覆盖与分布外控制性能'
    rebuild(soup, 's533', [
        '采用 300 个同分布对象和 500 个分布外对象评价统一学生的对象覆盖，前者从主试验 48 个评估对象之外抽取。学生权重保持固定，每个对象测试 14 条输入配置，各重复两次，共 22,400 条轨迹，记录跟踪误差、峰值误差与操纵变化。',
        '同分布对象上，±1 deg/s 和 ±31 deg/s 阶跃的跟踪 RMSE 中位数分别为 0.11、1.13 deg/s；分布外对象上相应为 0.21、2.93 deg/s。这组结果给出同一策略在不同对象分布及指令幅值下的跟踪误差水平。补充控制器对照见<a href="#results-pid-comparison">附录 B.4</a>。',
    ])
    soup.find(id='s535').insert_before(history_alias.extract())
    soup.find(id='s535').string = '5.3.4 历史表征对照'
    rebuild(soup, 's535', [
        '统一学生利用输入—响应历史获取对象动态线索。按第 3.3.1 节的输入设计与训练配置，保留基线时序编码的学生及 SAE 辅助输入方案在长延迟层均达到 90.0% 通过率，PCA-16、稠密 AE-16、Top-k SAE-16 替换方案分别为 51.6%、63.7%、71.9%。本实现据此采用与控制任务联合学习的时序编码，将历史映射为 16 维特征参与决策。完整闭环对照见<a href="#s613">附录 A.3</a>，信息读出与长时动态分析见<a href="#appendix-history">附录 A</a>。',
    ])
    rebuild(soup, 's54', [
        '在 DHC-6 ARX 目标模型上，small 学生经 40 次参数更新，8 个适配状态的 RMSE 中位数由 2.913 降至 0.595 deg/s，4 个留出状态由 2.662 降至 2.148 deg/s，分别降低 79.6% 和 19.3%。所有状态均采用 15 deg/s 阶跃和 8 s 评价时域。',
        figures['dhc6_states.svg'],
        '适配状态与留出状态的改善体现了源域控制知识在目标模型上的更新和跨状态复用。逐状态统计、两档模型的完整数值及共同更新预算见<a href="#results-transfer-details">附录 B.3</a>。',
    ])

    # The false internal title is redundant with the figure caption.
    digest = remove_dagger_title()
    dag = figures['student_dagger.svg']
    dag.img['src'] = f'paper/student_dagger.svg?v={digest[:12]}'
    if dag.img.parent.name == 'a':
        dag.img.parent['href'] = dag.img['src']
    dag.figcaption.string = f'图 {number["student_dagger.svg"]}　DAgger 轮次与师生闭环跟踪误差比。第 0 轮为行为克隆，随后在学生访问状态上追加教师标签。'
    matrix = figures['cross_object_matrix.svg']
    matrix.figcaption.string = f'图 {number["cross_object_matrix.svg"]}　专用教师跨对象控制与统一学生的对象覆盖。对角线为自身教师，非对角线为跨对象教师，末行为学生；每格汇总三次训练重复。右侧点为各次重复，横线为平均值。'
    paired_table.caption.clear()
    paired_table.caption.string = '表 6　配对教师—学生的闭环测试汇总。通过率按全部测试组合统计，RMSE 中位数按有限轨迹统计。'
    pid_table.caption.clear()
    pid_table.caption.string = '表 7　固定增益 PID 与统一学生在同一批对象上的对照。阶跃行括号内为 RMSE 中位数（deg/s）。'

    # Repair the subsection references before numbering the moved material.
    for text in list(soup.find_all(string=lambda value: value and '5.3.5' in value)):
        if text.parent.name not in ('script', 'style'):
            text.replace_with(str(text).replace('5.3.5', '5.3.4'))
    for text in list(soup.find_all(string=lambda value: value and '第 5.3.2 节的 120 对象扩展评估' in value)):
        text.replace_with(str(text).replace('第 5.3.2 节的 120 对象扩展评估', '附录 B.2 的 120 对象补充评估'))
    figure_mapping = renumber_figures(soup)
    table_mapping = renumber_tables(soup)
    for image in soup.find_all('img'):
        image['alt'] = image.find_parent('figure').figcaption.get_text(' ', strip=True)
    for link in list(soup.select('.toc a[href^="#"]')):
        target = soup.find(id=link['href'][1:])
        if target and target.name in ('h2', 'h3', 'h4'):
            link.string = target.get_text(' ', strip=True)
        elif target is None or target.name == 'span':
            link.decompose()
    appendix_link = soup.new_tag('a', href='#appendix-results')
    appendix_link.string = appendix.get_text(' ', strip=True)
    soup.select_one('.toc').append(appendix_link)

    omitted_figures = Counter(['student_amplitude.svg'])
    assert images_before == Counter(Path(i['src'].split('?')[0]).name for i in soup.find_all('img')) + omitted_figures
    assert tables_before == Counter(tuple(c.get_text() for c in t.select('tbody td')) for t in soup.find_all('table'))
    assert equations_before == [str(e) for e in soup.select('.eq')]
    assert math_before == Counter(a.get_text() for a in soup.select('annotation'))
    (ROOT / 'revision/results_number_mapping.json').write_text(json.dumps({
        'figures': figure_mapping, 'tables': table_mapping,
        'results_figures_before': 13, 'results_figures_after': 6,
        'figures_removed_from_paper': list(omitted_figures),
        'original_figure_files_retained_in_archive': True,
        'all_15_table_datasets_preserved': True,
    }, ensure_ascii=False, indent=2) + '\n')
