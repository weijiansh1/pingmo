"""Add the authorized teacher-to-other-plant comparison to section 5.3.2."""
from pathlib import Path
import json

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
    # Replace original Figures 11 and 12 with the paired comparison below.
    # Integrate the 120-object statistics into prose and remove the redundant
    # student-only time traces. Later original figure numbers remain valid.
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
    for paragraph in paragraphs:
        paragraph.decompose()
    removed = []
    for node in existing:
        if node.name == 'figure':
            source = node.find('img')['src'].split('?')[0]
            if source in ('paper/student_library_eval.svg', 'paper/student_grid.svg'):
                removed.append(source)
                node.decompose()
    assert set(removed) == {'paper/student_library_eval.svg', 'paper/student_grid.svg'}
    def percent(group):
        return f"{summary['groups'][group]['pass_rate'] * 100:.1f}%"
    names = {'own_teacher': '教师控制自身训练对象', 'foreign_teacher': '教师控制其他对象', 'student': '统一学生控制同组对象'}
    table_rows = []
    for group in names:
        d = summary['groups'][group]
        table_rows.append(f"<tr><td>{names[group]}</td><td>{d['passed_cases']}/{d['cases']}</td><td>{percent(group)}</td><td>{d['rmse_median_finite_deg_s']:.2f}</td><td>{d['failed_cases']}</td></tr>")
    html = f'''
<p>为比较专用教师的跨对象迁移与统一学生的多对象能力保留，补充采用 12 个源对象的既有配对教师—学生权重，三个品质等级各 4 个对象，共三组训练重复。每组包含 12 个逐对象教师和一个蒸馏全部 12 位教师的 tiny 学生，评估时全部权重冻结。矩阵纳入该组全部教师训练实例。</p>
<p>将每个教师依次用于全部 12 个对象，构成教师—对象矩阵：对角线为自身对象，非对角线为跨对象控制；统一学生作为末行，控制相同对象。迁移教师保持在 A 上训练所得权重，在 B 上接收 B 的真实参数与当前状态；学生仅接收普通观测与历史。每格使用同样的 18 条测试指令、8 s 闭环，按 RMSE ≤ 1.5 deg/s、峰值误差 ≤ 5 deg/s、请求力总变差率 ≤ 12 N/s 计通过。</p>
<figure class="w100" id="cross-object-matrix" data-source="evidence/cross_object_20261025/summary.json">
<a class="figure-zoom" href="paper/cross_object_matrix.svg" target="_blank"><img src="paper/cross_object_matrix.svg" alt="图 11　控制器与受控对象的通过率矩阵。黑框为教师自身对象，非对角线为教师跨对象控制，末行为统一学生；右侧展示三次训练重复的通过率。"/></a>
<figcaption>图 11　专用教师跨对象控制与统一学生的多对象控制。矩阵每格为三次训练重复的平均通过率，右侧点标出各次结果，横线为平均值；各次训练重复的完整矩阵见补充数据。</figcaption>
</figure>
<p>图 11 的行固定控制器、列固定受控对象；黑框为教师控制自身对象，非对角线为同一教师迁移到其他对象，最后一行为统一学生。每格汇总三组训练重复的 18 条指令，共 54 条测试。例如教师 1 控制对象 1 时通过 46/54（85%），控制对象 6 时通过 22/54（41%）。横向可读同一教师换对象后的变化，纵向可比较同一对象下不同控制器的表现。右图的圆点、方块和三角形分别表示三组训练重复，横线为平均值。</p>
<table class="bt" id="cross-object-results"><caption>表 0　配对教师—学生组的闭环指令测试汇总</caption><thead><tr><th>控制方式</th><th>通过 / 测试组合</th><th>通过率</th><th>RMSE 中位 (deg/s)</th><th>超限或非有限终止</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p class="small">汇总覆盖三次训练重复。自身教师与学生各含 648 个测试组合，跨对象教师含 7,128 个；非有限响应或角速度绝对值超过 300 deg/s 的轨迹按失败计入通过率，RMSE 中位数由有限轨迹计算。各格等权，全部测试组合保留。</p>
<p>教师由自身对象迁移到其他对象后，通过率从 {percent('own_teacher')} 降至 {percent('foreign_teacher')}；统一学生为 {percent('student')}，接近自身教师的整体水平，并高于迁移教师。三次重复中，学生分别为 87.0%、63.9%、48.1%，自身教师为 82.4%、61.1%、61.1%。蒸馏后形成了一套能覆盖多个源对象的控制器，但不同训练重复间仍有明显差异，第三次重复的学生还存在较大的能力保留损失。</p>
<p>配对组的 12 个对象均参与相应学生的蒸馏，这一比较检验蒸馏后的多对象能力保留；新增对象则进一步检验统一权重的对象覆盖。</p>
<p>{coverage_prose[0]}</p>
<p>{coverage_prose[1]}</p>
<figure class="w100" id="cross-object-time-response" data-source="evidence/cross_object_20261025/illustration_protocol_v2.json">
<a class="figure-zoom" href="paper/cross_object_responses.svg" target="_blank"><img src="paper/cross_object_responses.svg" alt="图 12　同一受控对象上，自身教师、外对象教师和统一学生对阶跃指令、初始偏差与外力扰动的响应。三个对象与外对象教师在评估前确定。"/></a>
<figcaption>图 12　验证达标教师的跨对象闭环对照。采用第一组训练重复，按原独立验证门限筛选，每个品质等级取原对象顺序中的首个达标教师，得到对象 1、6、9；迁移教师依次为教师 6、9、1。每行固定受控对象，比较自身教师、迁移教师和统一学生；三列依次为指令跟踪、初始偏差恢复和外力扰动抑制。黑色实线为外部指令，灰色点线为实验记录中的滤波目标。</figcaption>
</figure>
<p>时域对照先要求教师在各自训练对象上通过原独立验证，再考察迁移到其他对象后的变化。第一组 12 个教师中有 8 个满足该门限，图 12 按原对象顺序每级取一个，同时保留所选教师和学生在完整测试时域中的响应。选样依据及全部验证指标见<a href="evidence/cross_object_20261025/illustration_protocol_v2.json">时域对照协议</a>。</p>
<p>该组也存在教师自身能力不足的情况，<a href="paper/cross_object_diagnostic.svg">原选样的完整诊断图</a>保留这些失败响应。例如配对对象 2 的自身教师仅通过 4/18 条测试指令；其 30 s 扰动测试虽满足现有三项门限，后段振荡仍在增大。这说明当前通过率主要衡量给定时域内的跟踪和控制活动，抑振性能还需要长时恢复过程与末段残余振荡共同检验。</p>
<p class="small">补充数据：<a href="evidence/cross_object_20261025/controller_plant_results.csv">逐控制器—对象数值</a>；<a href="paper/cross_object_matrix_seed_20261031.svg">重复 1 矩阵</a>、<a href="paper/cross_object_matrix_seed_20261101.svg">重复 2 矩阵</a>、<a href="paper/cross_object_matrix_seed_20261102.svg">重复 3 矩阵</a>。</p>
'''
    new = BeautifulSoup(html, 'html.parser')
    anchor = heading
    for node in list(new.contents):
        anchor.insert_after(node.extract()); anchor = node
    return True
