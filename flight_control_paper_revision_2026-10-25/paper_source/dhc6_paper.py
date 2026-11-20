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


def main():
    source = SOURCE.read_text(encoding='utf-8')
    formulas = re.findall(r'\$\$\s*(.*?)\s*\$\$', source, flags=re.S)
    rendered = json.loads(subprocess.run(
        ['node', str(ROOT / 'paper_source/render_math.cjs')],
        input=json.dumps(formulas), text=True, capture_output=True, check=True,
    ).stdout)
    blocks = []
    def math_placeholder(match):
        number = len(blocks)
        blocks.append('<div class="eq">' + rendered[number] + '</div>')
        return f'\n\n<div id="paper-equation-{number}"></div>\n\n'
    source = re.sub(r'\$\$\s*(.*?)\s*\$\$', math_placeholder, source, flags=re.S)
    source = re.sub(r'\^\{\[([^}]+)\]\}', r'<sup>[\1]</sup>', source)
    converter = markdown.Markdown(extensions=['tables', 'toc', 'fenced_code', 'sane_lists'])
    body = converter.convert(source)
    for number, block in enumerate(blocks):
        body = body.replace(f'<div id="paper-equation-{number}"></div>', block)
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
        + converter.toc + '</aside><main>' + str(soup) + '</main></div></body></html>\n')
    for name in ('dhc6_report.html', 'dhc6_paper.html'):
        (SOURCE.parent / name).write_text(output, encoding='utf-8')
    print(f'Built {SOURCE.parent / "dhc6_report.html"}; {len(soup.find_all("img"))} images and {len(formulas)} equations verified')


if __name__ == '__main__':
    main()
