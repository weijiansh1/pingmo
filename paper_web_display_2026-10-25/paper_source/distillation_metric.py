"""Describe distillation using the recorded closed-loop results."""
import re

from bs4 import BeautifulSoup


DISTILLATION_RATIO_TEX = (
    r'R_{\mathrm D}=\operatorname*{median}_{i\in I}'
    r'\left(\frac{\operatorname{RMSE}_{p,i}^{\mathrm S}}'
    r'{\operatorname{RMSE}_{p,i}^{\mathrm T}}\right).'
)


def section_paragraphs(soup, ident):
    paragraphs = []
    for node in soup.find(id=ident).find_next_siblings():
        if node.name in ('h2', 'h3', 'h4'):
            break
        if node.name == 'p':
            paragraphs.append(node)
    return paragraphs


def set_content(paragraph, markup):
    paragraph.clear()
    paragraph.append(BeautifulSoup(markup, 'html.parser'))


def add_distillation_metric(soup):
    heading = soup.select_one('#s23')
    block = []
    for node in heading.find_next_siblings():
        if node.name in ('h2', 'h3', 'h4'):
            break
        block.append(node)
    assert block and not soup.select_one('#distillation-metric-equation')

    content = BeautifulSoup('''
<p id="distillation-metric">蒸馏后的相对跟踪精度采用师生闭环跟踪误差比评价。学生与对应教师在相同对象、指令和评价时段下分别运行，对各对象的跟踪 RMSE 比值取中位数：</p>
<div class="eq" id="distillation-metric-equation" data-source-equation="14"></div>
<p id="distillation-metric-definition">其中，<i>I</i> 为评价对象集合，上标 <i>S</i>、<i>T</i> 分别表示学生与对应对象的教师。<i>R</i><sub>D</sub> 以 1 为教师基准，数值越小表示学生相对教师的跟踪误差越低。蒸馏源对象与对象级留出集分别统计。</p>
<p id="distillation-metric-interpretation">误差比反映对象间的典型相对精度，绝对 RMSE 和门限通过率用于判断实际误差与达标覆盖，时域响应则用于观察瞬态过程和残余振荡。教师误差接近零时，误差比对较小的绝对差异也较敏感，需结合绝对误差解释。</p>
''', 'html.parser')
    anchor = block[-1]
    for node in list(content.children):
        if getattr(node, 'name', None):
            anchor.insert_after(node)
            anchor = node

    method = section_paragraphs(soup, 's33')
    assert len(method) == 3
    method[-1].string = (
        '多轮聚合在学生访问的状态上持续追加教师标签，扩展训练样本的状态覆盖。'
        '训练中的动作均方误差约束师生动作差异；蒸馏后的控制能力通过冻结权重的闭环测试评价，'
        '指标见第 2.3 节。'
    )

    table = next(t for t in soup.find_all('table')
                 if '两种规模学生的源域结果' in t.caption.get_text())
    table_number = re.match(r'表\s*(\d+)', table.caption.get_text())[1]
    table['id'] = 'distillation-results-table'
    table['class'] = list(table.get('class', [])) + ['distillation-results-table']
    wrapper = soup.new_tag('div', attrs={
        'class': 'parameter-table-wrap', 'id': 'distillation-results-table-wrap'})
    table.wrap(wrapper)
    headers = table.find_all('th')
    headers[-2].string = '源对象误差比'
    headers[-1].string = '留出对象误差比'

    result = section_paragraphs(soup, 's53')
    assert len(result) == 2
    result[0]['id'] = 'distillation-results'
    result[0].string = (
        '六轮 DAgger 后，tiny 和 small 学生在蒸馏源对象上的师生闭环跟踪误差比分别为 1.28、1.34，'
        '在留出对象上分别为 1.68、1.41。两种学生保留了多对象跟踪能力，但中位误差比均高于 1，'
        '表明统一策略相对专用教师存在跟踪精度损失。'
    )
    result[1].string = (
        '学生评估集共 48 个对象，三个品质等级各 16 个；其中 44 个用于 DAgger，'
        '4 个作为对象级留出。每个对象采用 30 条加权指令与 64 条并行确定性轨迹评价，'
        f'源对象与留出对象分别统计，结果见表 {table_number}。'
    )
    coverage = soup.new_tag('p', id='distillation-performance-retention')
    coverage.string = (
        'tiny 学生含 3,856 个参数，在 44 个源对象中有 37 个达标；small 含 75,040 个参数，有 38 个达标。'
        '两者在 4 个留出对象中均有 3 个达标。small 的留出对象误差比低于 tiny，源对象上则略高。'
        '在本组测试中，增加网络规模未带来一致的跟踪精度改进。'
    )
    wrapper.insert_after(coverage)

    figure = next(f for f in soup.find_all('figure')
                  if f.img['src'].split('?')[0].endswith('teacher_student_compare.svg'))
    figure_number = re.match(r'图\s*(\d+)', figure.figcaption.get_text())[1]
    response = section_paragraphs(soup, 's531')
    assert len(response) == 1
    set_content(response[0], (
        f'图 {figure_number} 在同一源对象、相同指令下比较教师与学生的闭环响应。'
        '六档阶跃的上升过程接近，学生在 2 s 后的跟踪 RMSE 为教师的 2.3–3.3 倍；'
        '双脉冲与多正弦对应的比值为 1.5–1.9。学生保留了主要响应形态，'
        f'跟踪精度仍低于教师。图中比较单个对象的不同指令，表 {table_number} 则汇总对象级误差比的中位数。'
    ))

    paired = next(p for p in section_paragraphs(soup, 's532')
                  if p.get_text().startswith('统一学生通过率为 66.4%'))
    paired.string = (
        '在 12 对象配对组中，统一学生通过率为 66.4%，自身教师为 68.2%，迁移教师为 27.6%。'
        '学生在已蒸馏对象上的整体通过率接近自身教师，高于迁移教师。'
        '三组重复中，学生通过率分别为 87.0%、63.9%、48.1%，自身教师为 82.4%、61.1%、61.1%。'
        '第三组学生的达标覆盖有所下降，表明蒸馏后的能力保留仍随训练重复变化。'
    )
