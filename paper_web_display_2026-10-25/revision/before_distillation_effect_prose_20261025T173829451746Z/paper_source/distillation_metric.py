"""Define the distillation metric already reported in the student results."""
from bs4 import BeautifulSoup


DISTILLATION_RATIO_TEX = (
    r'R_{\mathrm D}=\operatorname*{median}_{i\in I}'
    r'\left(\frac{\operatorname{RMSE}_{p,i}^{\mathrm S}}'
    r'{\operatorname{RMSE}_{p,i}^{\mathrm T}}\right).'
)


def add_distillation_metric(soup):
    heading = soup.select_one('#s23')
    block = []
    for node in heading.find_next_siblings():
        if node.name in ('h2', 'h3', 'h4'):
            break
        block.append(node)
    assert block and not soup.select_one('#distillation-metric-equation')

    content = BeautifulSoup('''
<p id="distillation-metric">蒸馏效果采用学生相对教师的跟踪误差比衡量。在相同对象、指令与评价时段下，先计算各对象的学生与教师 RMSE 之比，再取对象间中位数，记为蒸馏性能比：</p>
<div class="eq" id="distillation-metric-equation" data-source-equation="14"></div>
<p id="distillation-metric-definition">其中，<i>I</i> 为评价对象集合，上标 <i>S</i>、<i>T</i> 分别表示学生与对应对象的教师。<i>R</i><sub>D</sub> 以 1 为教师基准，数值越小表示学生相对教师的跟踪误差越低。蒸馏源对象与留出对象分别统计。</p>
''', 'html.parser')
    anchor = block[-1]
    for node in list(content.children):
        if getattr(node, 'name', None):
            anchor.insert_after(node)
            anchor = node

    result = soup.select_one('#s53')
    intro = result.find_next_sibling('p')
    intro.string = intro.get_text().replace('RMSE 中位比', '蒸馏性能比')
    protocol = intro.find_next_sibling('p')
    original = '通过数和相对误差，其中中位比按各对象的学生 RMSE 除以对应教师 RMSE 后取中位数。'
    assert original in protocol.get_text()
    protocol.string = protocol.get_text().replace(original, '通过数和蒸馏性能比。')
    table = next(t for t in soup.find_all('table')
                 if '两种规模学生的源域结果' in t.caption.get_text())
    headers = table.find_all('th')
    headers[-2].string = '源对象蒸馏性能比'
    headers[-1].string = '留出对象蒸馏性能比'
