"""Export a frozen copy of the current webpage as a paper without navigation."""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'flight_control_original_restored_2026-10-25/webpage'
OUT = ROOT / 'webpage/downloads'
BASE_URL = 'http://127.0.0.1:8320/webpage/'
NAME = '基于飞行品质的飞控大模型构建与滚转通道验证_无侧栏.pdf'
CLI = '/home/swj/.codex/skills/playwright/scripts/playwright_cli.sh'
CONFIG = ROOT.parent / 'paper_web_proxy_2026-10-25/output/playwright/original_display.config.json'

PRINT_CSS = r'''
@page { size:A4; }
@media print {
  html,body {margin:0 !important;padding:0 !important;background:white !important;}
  html {scroll-behavior:auto !important;}
  body {width:178mm;max-width:178mm;font-size:10pt;line-height:1.55;color:#16233a;}
  .toc,aside,nav {display:none !important;}
  .paper-shell {display:block !important;width:100% !important;max-width:none !important;
    padding:0 !important;margin:0 !important;}
  main {width:100% !important;max-width:none !important;min-width:0;
    padding:0 !important;margin:0 !important;box-shadow:none !important;}
  h1.title {font-size:18pt;line-height:1.4;margin:0 0 8pt;}
  .meta {font-size:8pt;margin-bottom:12pt;}
  .abstract {font-size:9.5pt;line-height:1.65;padding:9pt 2pt;}
  .keywords {font-size:9pt;margin:8pt 0 13pt;}
  h2 {font-size:13pt;margin:16pt 0 7pt;padding-bottom:4pt;break-after:avoid;}
  h3 {font-size:11.5pt;margin:10pt 0 6pt;break-after:avoid;}
  h4 {font-size:10.5pt;margin:10pt 0 5pt;break-after:avoid;}
  p {margin:0 0 6pt;orphans:2;widows:2;}
  figure {margin:10pt auto;break-inside:avoid;break-after:auto !important;}
  figure img {width:auto !important;max-width:100% !important;height:auto !important;
    max-height:160mm !important;object-fit:contain;}
  figcaption {font-size:8.5pt;line-height:1.5;margin-top:5pt;}
  .figure-analysis {font-size:9.5pt;line-height:1.55;margin-top:7pt;
    break-before:auto !important;break-inside:auto;}
  .eq {font-size:10.5pt;padding:7pt 3pt;gap:7pt;margin:8pt 0;
    break-inside:avoid;overflow:visible !important;}
  .eq .math-content {overflow:visible !important;}
  .eq .eqno {font-size:9pt;}
  .parameter-table-wrap,.responsive-paper-table {overflow:visible !important;
    width:100%;max-width:100%;break-inside:auto;}
  table.bt,table.bt.parameter-table,table.bt.observation-table,
  table.bt.sampling-protocol-table,table.bt.distillation-results-table,
  .responsive-paper-table table.bt {width:100% !important;min-width:0 !important;
    max-width:100%;font-size:8.5pt;line-height:1.5;margin:9pt 0 6pt;break-inside:auto;}
  table.bt th,table.bt td {padding:4pt;white-space:normal !important;}
  table.bt caption {font-size:8.5pt;line-height:1.5;break-after:avoid;}
  thead {display:table-header-group;break-after:avoid;}
  tr {break-inside:avoid;}
  ol.refs {font-size:8.5pt;line-height:1.55;}
  ol.refs li {break-inside:avoid;}
  a {color:inherit;text-decoration:none;}
  a::after {content:none !important;}
  footer {font-size:8pt;margin-top:14pt;}
}
'''


def freeze_source():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    archive = ROOT / 'revision' / f'pdf_export_{stamp}'
    archive.mkdir(parents=True)
    raw = (ROOT / 'webpage/dhc6_report.html').read_bytes()
    (archive / 'source.html').write_bytes(raw)
    if (OUT / NAME).exists():
        shutil.copy2(OUT / NAME, archive / 'previous_export.pdf')
    soup = BeautifulSoup(raw, 'html.parser')
    for node in soup.select('.toc'):
        node.decompose()
    main = soup.main.extract()
    soup.body.clear()
    soup.body.append(main)
    base = soup.new_tag('base', href=BASE_URL)
    soup.head.insert(0, base)
    assets = []
    for image in soup.find_all('img'):
        original_src = image['src']
        rel = urlsplit(original_src).path
        path = ROOT / 'webpage' / rel
        if not path.exists():
            path = ORIGINAL / rel
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or 'image/svg+xml'
        image['data-source-src'] = original_src
        image['src'] = f'data:{mime};base64,' + base64.b64encode(data).decode()
        assets.append({'source': original_src, 'sha256': hashlib.sha256(data).hexdigest()})
    # Use the exact stylesheet at export time; its font files are content named.
    font_link = next(link for link in soup.find_all('link', href=True)
                     if 'paper-fonts.css' in link['href'])
    font_css = (ROOT / 'webpage/vendor/paper-fonts.css').read_text()
    font_css = font_css.replace('url("fonts/', 'url("' + BASE_URL + 'vendor/fonts/')
    style = soup.new_tag('style')
    style.string = font_css
    font_link.replace_with(style)
    style = soup.new_tag('style', id='pdf-print-layout')
    style.string = PRINT_CSS
    soup.head.append(style)
    assert len(soup.find_all('figure')) == 22
    assert len(soup.find_all('table')) == 16
    assert len(soup.select('.eq')) == 13
    assert not soup.select('.toc')
    frozen = OUT / ('paper_print_snapshot_' + hashlib.sha256(raw).hexdigest()[:12] + '.html')
    frozen.write_text(str(soup))
    (archive / 'print.css').write_text(PRINT_CSS)
    record = {
        'archive': str(archive), 'source_html_sha256': hashlib.sha256(raw).hexdigest(),
        'frozen_html': str(frozen), 'image_assets': assets,
        'print_css_sha256': hashlib.sha256(PRINT_CSS.encode()).hexdigest(),
    }
    return frozen, record


def export(frozen):
    env = dict(os.environ)
    env['PATH'] = '/home/swj/.nvm/versions/node/v22.23.2/bin:' + env['PATH']
    env['LD_LIBRARY_PATH'] = '/tmp/pw-libs/usr/lib/x86_64-linux-gnu:/home/swj/.local/syslib/usr/lib/x86_64-linux-gnu:' + env.get('LD_LIBRARY_PATH', '')
    cli = [CLI, '-s=paper-pdf-clean']

    def call(args):
        result = subprocess.run(cli + args, env=env, capture_output=True, text=True, timeout=60)
        if result.returncode or '### Error' in result.stdout:
            raise RuntimeError(result.stdout[-3500:] + result.stderr[-1500:])
        return result.stdout

    footer = '<div style="width:100%;text-align:center;font-size:8px;color:#667085"><span class="pageNumber"></span></div>'
    code = r'''async(page)=>{
      await page.goto(URL,{waitUntil:'networkidle',timeout:25000});
      await page.emulateMedia({media:'print'});
      await page.evaluate(()=>document.fonts.ready);
      await page.evaluate(()=>Promise.all([...document.images].map(i=>i.decode())));
      const fitted=await page.evaluate(()=>{
        const changed=[];
        for(const el of document.querySelectorAll('.eq .math-content')){
          const before=parseFloat(getComputedStyle(el).fontSize);
          if(el.scrollWidth>el.clientWidth+1){
            el.style.fontSize=(before*el.clientWidth/el.scrollWidth*.99)+'px';
            changed.push({equation:el.parentElement.querySelector('.eqno').innerText,
              before,after:parseFloat(el.style.fontSize)});
          }
        }
        return changed;
      });
      const verification=await page.evaluate(()=>({
        title:document.title,tocCount:document.querySelectorAll('.toc').length,
        figures:document.querySelectorAll('figure').length,
        tables:document.querySelectorAll('table').length,
        equations:document.querySelectorAll('.eq').length,
        analyses:document.querySelectorAll('.figure-analysis').length,
        mathErrors:document.querySelectorAll('.katex-error').length,
        brokenImages:[...document.images].filter(i=>!i.naturalWidth).map(i=>i.dataset.sourceSrc),
        pageWidth:document.body.getBoundingClientRect().width,
        figureDimensions:[...document.querySelectorAll('figure')].map(f=>({
          caption:f.querySelector('figcaption').innerText,
          width:f.getBoundingClientRect().width,height:f.getBoundingClientRect().height}))
      }));
      if(verification.tocCount||verification.mathErrors||verification.brokenImages.length)
        throw new Error(JSON.stringify(verification));
      await page.pdf({path:PDF,format:'A4',printBackground:true,preferCSSPageSize:true,
        displayHeaderFooter:true,headerTemplate:'<div></div>',footerTemplate:FOOTER,
        margin:{top:'14mm',right:'16mm',bottom:'16mm',left:'16mm'}});
      return {...verification,fitted};
    }'''.replace('URL', json.dumps(BASE_URL + 'downloads/' + frozen.name))
    code = code.replace('PDF', json.dumps(str(OUT / NAME))).replace('FOOTER', json.dumps(footer))
    call(['open', 'about:blank', '--config', str(CONFIG)])
    try:
        result = call(['run-code', code])
        return json.loads(result.split('### Result\n', 1)[1].split('\n### ', 1)[0].strip())
    finally:
        call(['close'])


def main():
    OUT.mkdir(exist_ok=True)
    frozen, record = freeze_source()
    record['browser'] = export(frozen)
    path = OUT / NAME
    reader = PdfReader(path)
    texts = [page.extract_text() or '' for page in reader.pages]
    assert all(re.sub(r'\s+', '', text) for text in texts)
    assert '本文目录' not in re.sub(r'\s+', '', ''.join(texts))
    record.update(file=str(path), pages=len(reader.pages), bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  page_text_lengths=[len(text) for text in texts])
    (ROOT / 'revision/clean_pdf_export.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'file':str(path),'pages':len(reader.pages),'bytes':path.stat().st_size}, ensure_ascii=False))


if __name__ == '__main__':
    main()
