"""Add the authorized teacher-to-other-plant comparison to section 5.3.2."""
from pathlib import Path
import json
import re

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent / 'cross_object_control_2026-10-25'


def add_cross_object_comparison(soup):
    summary_path = STUDY / 'summary.json'
    if not summary_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    audit = json.loads((STUDY / 'replication_audit.json').read_text())
    assert audit['all_reproduced']
    for name in ('cross_object_matrix.svg', 'cross_object_responses.svg'):
        assert (ROOT / 'webpage/paper' / name).exists()
    # Insert after original Figure 10; synchronize all original references to
    # Figures 12 onward before adding the new references and captions.
    # Original Figure 11's statistics are integrated into the prose below.
    def shift(match):
        n = int(match[1])
        return f'图 {n + 1}' if 12 <= n <= 22 else match[0]
    pattern = re.compile(r'图\s*(\d+)(?!\d)')
    for node in list(soup.find_all(string=pattern)):
        if node.parent.name not in ('script', 'style'):
            node.replace_with(pattern.sub(shift, str(node)))
    for image in soup.find_all('img'):
        if image.has_attr('alt'):
            image['alt'] = pattern.sub(shift, image['alt'])
    heading = soup.find(id='s532')
    existing = []
    for node in heading.next_siblings:
        if getattr(node, 'name', None) in ('h2', 'h3', 'h4'):
            break
        if getattr(node, 'name', None):
            existing.append(node)
    paragraphs = [node for node in existing if node.name == 'p']
    assert len(paragraphs) == 4
    coverage_prose = [
        '对象覆盖进一步通过第 5.3 节主试验的 tiny 学生检验：从 4,500 个非分布外对象中按品质等级各抽取 40 个，共 120 个，与原 48 对象评估集不重叠，抽样种子为 20261023。按相同三项门限，通过 94/120（78.3%），三个等级分别通过 34/40、30/40、30/40；RMSE 中位数为 0.64 deg/s，90 百分位为 2.24 deg/s。',
        '这组新增对象也揭示了统一控制的薄弱区间：RMSE 与滚转操纵效能的相关系数为 −0.47；滚转时间常数 <i>T</i><sub>R</sub> > 6 s 的 4 个对象中仅 1 个通过。结合前面的配对对照，当前路线已能用统一权重保留多对象控制能力，进一步提高可靠性仍需关注低操纵效能、慢滚转响应和训练重复之间的差异。',
    ]
    for paragraph in paragraphs[:3]:
        paragraph.decompose()
    paragraphs[-1].clear()
    paragraphs[-1].string = '图 13 补充展示同一主试验 tiny 学生在 train_core-0000、train_core-0421、train_core-0701 上的正、负阶跃与多正弦响应，体现统一权重在不同品质对象上的瞬态差异。'
    old_coverage_figure = next(node for node in existing if node.name == 'figure'
                               and 'student_library_eval.svg' in node.find('img')['src'])
    old_coverage_figure.decompose()
    def percent(group):
        return f"{summary['groups'][group]['pass_rate'] * 100:.1f}%"
    names = {'own_teacher': '教师控制自身训练对象', 'foreign_teacher': '教师控制其他对象', 'student': '统一学生控制同组对象'}
    table_rows = []
    for group in names:
        d = summary['groups'][group]
        table_rows.append(f"<tr><td>{names[group]}</td><td>{d['passed_cases']}/{d['cases']}</td><td>{percent(group)}</td><td>{d['rmse_median_finite_deg_s']:.2f}</td><td>{d['failed_cases']}</td></tr>")
    html = f'''
<p>为比较专用教师的跨对象迁移与统一学生的多对象能力保留，补充采用 12 个源对象的配对教师—学生组，三个品质等级各 4 个对象，进行三次训练重复。每组包含 12 个逐对象教师和一个蒸馏全部 12 位教师的 tiny 学生，采用该组冻结权重开展对照。</p>
<p>将每个教师依次用于全部 12 个对象，构成教师—对象矩阵：对角线为自身对象，非对角线为跨对象控制；统一学生作为末行，控制相同对象。迁移教师保持在 A 上训练所得权重，在 B 上接收 B 的真实参数与当前状态；学生仅接收普通观测与历史。每格使用同样的 18 条测试指令、8 s 闭环，按 RMSE ≤ 1.5 deg/s、峰值误差 ≤ 5 deg/s、请求力总变差率 ≤ 12 N/s 计通过。</p>
<figure class="w100" id="cross-object-matrix" data-source="evidence/cross_object_20261025/summary.json">
<a class="figure-zoom" href="paper/cross_object_matrix.svg" target="_blank"><img src="paper/cross_object_matrix.svg" alt="图 11　控制器与受控对象的通过率矩阵。黑框为教师自身对象，非对角线为教师跨对象控制，末行为统一学生；右侧展示三次训练重复的通过率。"/></a>
<figcaption>图 11　专用教师跨对象控制与统一学生的多对象控制。矩阵每格为三次训练重复的平均通过率，右侧点标出各次结果，横线为平均值；各次训练重复的完整矩阵见补充数据。</figcaption>
</figure>
<table class="bt" id="cross-object-results"><caption>表 0　配对教师—学生组的闭环指令测试汇总</caption><thead><tr><th>控制方式</th><th>通过 / 测试组合</th><th>通过率</th><th>RMSE 中位 (deg/s)</th><th>超限或非有限终止</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p class="small">汇总覆盖三次训练重复。自身教师与学生各含 648 个测试组合，跨对象教师含 7,128 个；非有限响应或角速度绝对值超过 300 deg/s 的轨迹按失败计入通过率，RMSE 中位数由有限轨迹计算。各格等权，全部测试组合保留。</p>
<p>教师由自身对象迁移到其他对象后，通过率从 {percent('own_teacher')} 降至 {percent('foreign_teacher')}；统一学生为 {percent('student')}，接近自身教师的整体水平，并高于迁移教师。三次重复中，学生分别为 87.0%、63.9%、48.1%，自身教师为 82.4%、61.1%、61.1%。蒸馏后形成了一套能覆盖多个源对象的控制器，但不同训练重复间仍有明显差异，第三次重复的学生还存在较大的能力保留损失。</p>
<p>配对组的 12 个对象均参与相应学生的蒸馏，这一比较检验蒸馏后的多对象能力保留；新增对象则进一步检验统一权重的对象覆盖。</p>
<p>{coverage_prose[0]}</p>
<p>{coverage_prose[1]}</p>
<figure class="w100" id="cross-object-time-response" data-source="evidence/cross_object_20261025/protocol.json">
<a class="figure-zoom" href="paper/cross_object_responses.svg" target="_blank"><img src="paper/cross_object_responses.svg" alt="图 12　同一受控对象上，自身教师、外对象教师和统一学生对阶跃指令、初始偏差与外力扰动的响应。三个对象与外对象教师在评估前确定。"/></a>
<figcaption>图 12　同一对象上的三类控制器闭环响应。每行固定受控对象 B，比较其自身教师、在另一对象 A 上训练的教师和统一学生；三列依次为指令跟踪、初始偏差恢复和外力扰动抑制。展示第一组训练重复，预先按各品质等级最小对象编号选择受控对象。</figcaption>
</figure>
<p>时域对照进一步区分跟踪与振荡抑制：在对象 0401 上，外对象教师的响应振荡逐渐增大，而自身教师与学生保持有界；对象 0750 的自身教师在 30 s 恢复测试中也出现振荡增长，学生则衰减回零。这些差异说明，多对象控制能力需要同时用指令测试和长时恢复测试检验。</p>
<p class="small">补充数据：<a href="evidence/cross_object_20261025/controller_plant_results.csv">逐控制器—对象数值</a>；<a href="paper/cross_object_matrix_seed_20261031.svg">重复 1 矩阵</a>、<a href="paper/cross_object_matrix_seed_20261101.svg">重复 2 矩阵</a>、<a href="paper/cross_object_matrix_seed_20261102.svg">重复 3 矩阵</a>。</p>
'''
    new = BeautifulSoup(html, 'html.parser')
    anchor = heading
    for node in list(new.contents):
        anchor.insert_after(node.extract()); anchor = node
    return True
