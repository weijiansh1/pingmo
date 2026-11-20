"""Readable manuscript identifiers; source records and model inputs stay intact."""
from pathlib import Path
import csv
import hashlib
import json
import re

from bs4 import BeautifulSoup, Comment
from fontTools.ttLib import TTFont
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath, TextToPath

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'webpage'
ORIGINAL = ROOT.parent / 'flight_control_original_restored_2026-10-25/webpage'
STUDY = ROOT.parent / 'cross_object_control_2026-10-25'
BASE = ROOT / 'paper_source/identifier_sources'
BASE.mkdir(exist_ok=True)
STATE = ROOT / 'revision/readable_identifier_assets.json'
MAIN_IDS = ['train_core-0174', 'train_core-0236', 'train_core-0467',
            'train_boundary-1488', 'train_core-0175', 'train_boundary-1388', 'train_core-0421']
PAIR_IDS = [p['plant_id'] for p in json.loads((STUDY / 'protocol.json').read_text())['source_objects']]
DELAY_IDS = ['id_test-3941', 'id_test-3458']
FLIGHT_STATES = ['V060_H2500', 'V060_H3000', 'V060_H3500', 'V070_H2500',
                 'V070_H3000', 'V070_H3500', 'V080_H2500', 'V080_H3000',
                 'V050_H2500', 'V050_H3000', 'V050_H3500', 'V080_H3500']
LABELS = {**{v: f'主试验对象 {i}' for i, v in enumerate(MAIN_IDS, 1)},
          **{v: f'配对对象 {i}' for i, v in enumerate(PAIR_IDS, 1)},
          **{v: f'时延对象 {i}' for i, v in enumerate(DELAY_IDS, 1)},
          **{v: f'状态 {i}' for i, v in enumerate(FLIGHT_STATES, 1)}}
PAIR_SUFFIX = {v.split('-')[-1]: i for i, v in enumerate(PAIR_IDS, 1)}
SOURCE_ID = re.compile(r'(?:train_core|train_boundary|validation|id_test|ood_test)-\d{4}|V\d{3}_H\d{4}')
PREPARED = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rewrite(text, filename=''):
    result = str(text)
    for original, label in LABELS.items():
        result = re.sub(r'对象\s*' + re.escape(original), label, result)
        result = result.replace(original, label)
    result = result.replace('train_core、validation、id_test', '训练集、验证集和留出测试集')
    result = result.replace('raw_ft', '原始历史微调')
    result = result.replace('抽样种子为 20261023', '采用固定种子抽样')
    result = result.replace('抽样种子为20261023', '采用固定种子抽样')
    result = re.sub(r'\s*·\s*seed\s+\d{8}', '', result)
    if filename.startswith('cross_object_matrix'):
        for suffix, number in PAIR_SUFFIX.items():
            if result.strip() == suffix:
                result = str(number)
            elif result.strip() == 'T' + suffix:
                result = f'教师 {number}'
        result = result.replace('受控对象 B', '受控对象编号').replace('专用教师的训练对象 A', '教师编号')
    if filename in ('cross_object_responses.svg', 'cross_object_diagnostic.svg'):
        result = re.sub(r'外对象教师\s+A=(\d{4})',
                        lambda m: f'迁移教师 {PAIR_SUFFIX[m[1]]}', result)
        result = re.sub(r'B=(\d{4})', lambda m: f'对象 {PAIR_SUFFIX[m[1]]}', result)
    for suffix, number in PAIR_SUFFIX.items():
        result = re.sub(r'对象\s+' + suffix + r'(?!\d)', f'配对对象 {number}', result)
    result = result.replace('3 个 V050 留出状态', '留出状态 9–11')
    if filename == 'delay_closed_loop.svg':
        result = result.replace('(b) 跨时延：训练分布对长延迟留出的影响', '(b) 再训练样本范围与长延迟表现')
        result = result.replace('全延迟训练', '全延迟再训练').replace('仅短/中延迟训练', '仅短/中延迟再训练')
        result = result.replace('闭环对照：延迟分层与跨时延迁移（120 架留出）', '历史表征的闭环对照（120 架留出）')
    return result


def shape_records(svg):
    """All non-text geometry and its presentation must be identical."""
    result = []
    for tag in svg.find_all(['path', 'use', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon']):
        if any(re.match(r'text_\d+$', p.get('id', '')) for p in tag.parents if getattr(p, 'attrs', None)):
            continue
        result.append((tag.name, dict(tag.attrs)))
    return result


def font_for(text):
    path = ROOT / 'paper_source/identifier-font.ttf'
    chars = set(text + ''.join(chr(i) for i in range(32, 127)))
    if not path.exists() or not {ord(c) for c in chars}.issubset(TTFont(path).getBestCmap()):
        from prepare_display_fonts import font
        binary, url = font(''.join(sorted(chars)))
        path.write_bytes(binary)
        (ROOT / 'revision/identifier_font.json').write_text(json.dumps({
            'source': url, 'sha256': sha(binary), 'characters': len(chars)}, ensure_ascii=False, indent=2) + '\n')
    return FontProperties(fname=str(path), size=100)


def path_string(text, font):
    path = TextPath((0, 0), text, prop=font, size=100)
    pieces = []
    commands = {MplPath.MOVETO: 'M', MplPath.LINETO: 'L', MplPath.CURVE3: 'Q', MplPath.CURVE4: 'C'}
    for vertices, code in path.iter_segments(curves=True, simplify=False):
        if code == MplPath.CLOSEPOLY:
            pieces.append('z')
        else:
            pieces.append(commands[code] + ' '.join(f'{v:.7g}' for v in vertices))
    return ' '.join(pieces)


def redraw_text(svg, comment, text, font):
    original = str(comment).strip()
    group = comment.find_next_sibling('g')
    assert group is not None and group.parent is comment.parent
    uses = group.find_all('use')
    assert uses, original
    transform = group.get('transform', '')
    # Tick labels are right aligned. Preserve their original anchor after
    # replacing a long source ID with a shorter manuscript number.
    tick = next((p for p in comment.parents if re.match(r'[xy]tick_\d+$', p.get('id', ''))), None)
    centered = original.startswith(('Top-k SAE 与原始历史的 180 s', 'DHC-6 迁移：', '闭环对照：'))
    if tick is not None or centered:
        last = uses[-1]
        position = re.search(r'translate\(([-\d.]+)', last.get('transform', ''))
        last_x = float(position[1]) if position else float(last.get('x', 0))
        measure = TextToPath()
        old_width = last_x + measure.get_text_width_height_descent(original[-1], font, False)[0]
        new_width = measure.get_text_width_height_descent(text, font, False)[0]
        transform += f' translate({(old_width - new_width) * (.5 if centered else 1):.7g} 0)'
    # Keep glyph definitions: other unedited labels may refer to them.
    for node in list(group.children):
        if getattr(node, 'name', None) == 'defs':
            continue
        node.extract()
    path = svg.new_tag('path', d=path_string(text, font))
    path['data-display-label'] = text
    group['transform'] = transform
    group.append(path)
    comment.replace_with(Comment(' ' + text + ' '))


def prepare_assets(soup):
    global PREPARED
    if PREPARED is not None:
        return PREPARED
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    records = {}
    paths = {i['src'].split('?')[0] for i in soup.find_all('img')}
    paths.update(a['href'].split('?')[0] for a in soup.find_all('a', href=True)
                 if a['href'].startswith('paper/') and a['href'].split('?')[0].endswith('.svg'))
    paths.update(f'paper/cross_object_matrix_seed_{s}.svg' for s in (20261031, 20261101, 20261102))
    pending = []
    font_text = []
    for relative in sorted(paths):
        current = WEB / relative
        if not current.exists():
            current = ORIGINAL / relative
        raw = current.read_bytes()
        source = BASE / Path(relative).name
        prior = previous.get(relative)
        if prior and sha(raw) == prior['display_sha256']:
            raw = source.read_bytes()
        svg = BeautifulSoup(raw, 'xml')
        changes = []
        for comment in svg.find_all(string=lambda t: isinstance(t, Comment)):
            old = str(comment).strip(); new = rewrite(old, Path(relative).name)
            if new != old:
                changes.append((comment, old, new)); font_text.extend([old, new])
        if changes:
            source.write_bytes(raw)
            pending.append((relative, raw, svg, changes))
    font = font_for(''.join(font_text)) if pending else None
    for relative, raw, svg, changes in pending:
        geometry = shape_records(svg)
        for comment, old, new in changes:
            redraw_text(svg, comment, new, font)
        assert shape_records(svg) == geometry, f'Non-text geometry changed: {relative}'
        target = WEB / relative; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(svg), encoding='utf-8')
        records[relative] = {'source_sha256': sha(raw), 'display_sha256': sha(target.read_bytes()),
                             'all_nontext_geometry_unchanged': True,
                             'changes': [{'old': old, 'new': new} for _, old, new in changes]}
    STATE.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    mapping = {
        'numbering_scope': 'Each experiment group starts at 1; labels are consistent in prose and figures.',
        'main_objects': [{'number': i, 'source_id': v, 'label': LABELS[v]} for i, v in enumerate(MAIN_IDS, 1)],
        'paired_objects_and_teachers': [{'number': i, 'source_id': v, 'object_label': f'对象 {i}',
                                          'teacher_label': f'教师 {i}'} for i, v in enumerate(PAIR_IDS, 1)],
        'delay_objects': [{'number': i, 'source_id': v, 'label': LABELS[v]} for i, v in enumerate(DELAY_IDS, 1)],
        'dhc6_states': [{'number': i, 'source_id': v, 'label': LABELS[v]} for i, v in enumerate(FLIGHT_STATES, 1)],
    }
    (WEB / 'evidence/display_identifier_mapping.json').write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + '\n')
    source_csv = STUDY / 'controller_plant_results.csv'
    with source_csv.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row['seed'] = '重复 ' + str([20261031, 20261101, 20261102].index(int(row['seed'])) + 1)
        row['controller'] = '统一学生' if row['controller'] == 'student' else '教师 ' + str(PAIR_IDS.index(row['controller']) + 1)
        row['target_plant'] = '对象 ' + str(PAIR_IDS.index(row['target_plant']) + 1)
    with (WEB / 'evidence/cross_object_20261025/controller_plant_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    PREPARED = records
    return records


def apply_identifiers(soup):
    prepare_assets(soup)
    for node in list(soup.find_all(string=True)):
        if node.parent.name in ('script', 'style') or isinstance(node, Comment):
            continue
        result = rewrite(str(node))
        if result != str(node):
            node.replace_with(result)
    for image in soup.find_all('img'):
        if image.has_attr('alt'):
            image['alt'] = rewrite(image['alt'])
    caption = soup.select_one('#cross-object-matrix figcaption')
    if caption:
        caption.append(' 对象与其专用教师按 1–12 对应编号。')
    metrics = soup.find(id='s23')
    paragraph = soup.new_tag('p', id='pass-rate-definition')
    paragraph.string = '同时满足三项门限的测试记为达标，通过率为达标测试数占全部测试数的比例。教师—学生对照与历史表征消融按对象—指令测试组合统计；第 5.3.2 节的 120 对象扩展评估按每个对象的归档汇总指标统计。'
    metrics.find_next_sibling('p').insert_after(paragraph)
    link = soup.select_one('a[href="evidence/cross_object_20261025/controller_plant_results.csv"]')
    if link:
        wrapper = soup.new_tag('span')
        wrapper.append('；')
        mapping = soup.new_tag('a', href='evidence/display_identifier_mapping.json')
        mapping.string = '编号对照'; wrapper.append(mapping)
        link.insert_after(wrapper)


def version_assets(soup):
    for tag in soup.find_all(['img', 'a']):
        attribute = 'src' if tag.name == 'img' else 'href'
        relative = tag.get(attribute, '').split('?')[0]
        if relative in (PREPARED or {}):
            tag[attribute] = relative + '?v=labels-' + PREPARED[relative]['display_sha256'][:12]
