"""Build the corrected report from its authoritative Markdown, offline.

Run: python paper_source/dhc6_paper.py
No omitted _report_review data are needed to render the checked manuscript.
"""
from pathlib import Path
import html
import json
import re
import subprocess

import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'webpage/dhc6_report.md'
MATH_PATTERN = re.compile(r'\$\$(?P<display>.*?)\$\$|\\\((?P<inline>.*?)\\\)', re.S)


def render_manuscript(source):
    """Protect both display and inline TeX from Markdown before rendering."""
    formulas = [
        {'tex': (m.group('display') if m.group('display') is not None else m.group('inline')).strip(),
         'display': m.group('display') is not None}
        for m in MATH_PATTERN.finditer(source)
    ]
    rendered = json.loads(subprocess.run(
        ['node', str(ROOT / 'paper_source/render_math.cjs')],
        input=json.dumps(formulas), text=True, capture_output=True, check=True,
    ).stdout)
    placeholders = []

    def replace_math(match):
        number = len(placeholders)
        display = formulas[number]['display']
        tag = 'div' if display else 'span'
        token = f'<{tag} id="paper-math-{number}"></{tag}>'
        css_class = 'eq' if display else 'math-inline'
        replacement = f'<{tag} class="{css_class}">' + rendered[number] + f'</{tag}>'
        placeholders.append((token, replacement))
        return '\n\n' + token + '\n\n' if display else token

    protected = MATH_PATTERN.sub(replace_math, source)
    protected = re.sub(r'\^\{\[([^}]+)\]\}', r'<sup class="citation">[\1]</sup>', protected)
    converter = markdown.Markdown(extensions=['tables', 'toc', 'fenced_code', 'sane_lists'])
    body = converter.convert(protected)
    for token, replacement in placeholders:
        if token not in body:
            raise ValueError('Math placeholder lost during Markdown conversion: ' + token)
        body = body.replace(token, replacement)
    if 'paper-math-' in body:
        raise ValueError('Unresolved mathematical placeholder')
    return body, converter.toc, formulas


def main():
    source = SOURCE.read_text(encoding='utf-8')
    body, toc, formulas = render_manuscript(source)
    soup = BeautifulSoup(body, 'html.parser')
    if soup.h1:
        soup.h1['class'] = 'title'
    for table in soup.find_all('table'):
        table['class'] = 'bt'
    missing = []
    for image in soup.find_all('img'):
        path = SOURCE.parent / image['src']
        if not path.is_file():
            missing.append(str(path))
        image['loading'] = 'lazy'
    if missing:
        raise FileNotFoundError('Missing report images: ' + ', '.join(missing))
    style = (ROOT / 'paper_source/report_style.css').read_text(encoding='utf-8')
    font_style = ROOT / 'paper_source/chinese_font.css'
    if font_style.is_file():
        style += '\n' + font_style.read_text(encoding='utf-8')
    title = soup.h1.get_text() if soup.h1 else '飞控研究修订报告'
    output = ('<!DOCTYPE html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<link rel="icon" href="data:,">'
        '<link rel="stylesheet" href="vendor/katex/katex.min.css">'
        '<title>' + html.escape(title) + '</title><style>' + style + '</style></head>'
        '<body><div class="paper-shell"><aside class="toc" aria-label="章节导航">'
        + toc + '</aside><main>' + str(soup) + '</main></div></body></html>\n')
    for name in ('dhc6_report.html', 'dhc6_paper.html'):
        (SOURCE.parent / name).write_text(output, encoding='utf-8')
    counts = {'display': sum(item['display'] for item in formulas),
              'inline': sum(not item['display'] for item in formulas)}
    (ROOT / 'revision/math_manifest.json').write_text(json.dumps(
        {'counts': counts, 'expressions': formulas}, ensure_ascii=False, indent=2) + '\n')
    print(f'Built {SOURCE.parent / "dhc6_report.html"}; {len(soup.find_all("img"))} images; math: {counts}')


if __name__ == '__main__':
    main()
