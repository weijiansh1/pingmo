"""Export the current paper and its technical review without editing the paper.

Run from the overlay directory: python paper_source/export_current_pdf.py
The live local server supplies the existing SVGs and fonts on their own origin.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.request import urlopen

from bs4 import BeautifulSoup
from fontTools.ttLib import TTFont
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "webpage/downloads"
PAPER_NAME = "基于飞行品质的飞控大模型构建与滚转通道验证_当前稿.pdf"
REVIEW_NAME = "当前稿技术合理性核查.pdf"
BASE_URL = "http://127.0.0.1:8320/webpage/"
CONFIG = ROOT.parent / "paper_web_proxy_2026-10-25/output/playwright/original_display.config.json"

PRINT_CSS = """
@page {size:A4;}
@media print {
  html,body {margin:0 !important;padding:0 !important;background:white !important;}
  body {font-size:10.5pt;line-height:1.65;color:#16233a;}
  .paper-shell {display:block !important;max-width:none !important;margin:0 !important;padding:0 !important;}
  .toc {display:none !important;}
  main {padding:0 !important;box-shadow:none !important;min-width:0;max-width:100%;}
  h1.title {font-size:18pt;line-height:1.45;margin-bottom:10pt;}
  .meta {font-size:8pt;margin-bottom:12pt;}
  .abstract {font-size:9.5pt;line-height:1.7;padding:10pt 2pt;}
  .keywords {font-size:9pt;margin:9pt 0 15pt;}
  h2 {font-size:13pt;margin:20pt 0 8pt;break-after:avoid;}
  h3 {font-size:11.5pt;margin:16pt 0 7pt;break-after:avoid;}
  h4 {font-size:10.5pt;margin:12pt 0 6pt;break-after:avoid;}
  p {margin:0 0 8pt;orphans:3;widows:3;}
  figure {margin:12pt auto;break-inside:avoid;}
  figure img {width:auto !important;max-width:100% !important;height:auto;max-height:185mm;object-fit:contain;}
  figcaption {font-size:8.5pt;line-height:1.6;margin-top:6pt;}
  .figure-analysis {font-size:9.5pt;break-before:auto;}
  .eq {font-size:10.5pt;padding:9pt 4pt;gap:8pt;margin:10pt 0;overflow:visible;break-inside:avoid;}
  .eq .math-content {overflow:visible;}
  .eq .eqno {font-size:9pt;}
  .parameter-table-wrap,.responsive-paper-table {overflow:visible !important;max-width:100%;}
  table.bt,table.bt.parameter-table,table.bt.observation-table,
  table.bt.sampling-protocol-table,table.bt.distillation-results-table,
  .responsive-paper-table table.bt {
    width:100% !important;min-width:0 !important;max-width:100%;
    font-size:8.2pt;line-height:1.55;margin:12pt 0 6pt;
  }
  table.bt th,table.bt td {padding:4pt;white-space:normal !important;}
  table.bt caption {font-size:8.5pt;line-height:1.6;}
  thead {display:table-header-group;}
  tr {break-inside:avoid;}
  table {break-inside:auto;}
  ol.refs {font-size:8.5pt;line-height:1.65;}
  ol.refs li {break-inside:avoid;}
  a {color:inherit;text-decoration:none;}
  a::after {content:none !important;}
  footer {font-size:8pt;margin-top:20pt;}
}
"""

REVIEW_CSS = """
body {margin:0;background:white;color:#16233a;font-family:PaperChinese,Georgia,serif;
      font-size:10.5pt;line-height:1.75;}
main {max-width:100%;}
h1 {font-size:19pt;margin:0 0 16pt;line-height:1.5;}
h2 {font-size:13pt;margin:20pt 0 8pt;border-bottom:1px solid #c9d3e2;padding-bottom:5pt;}
h3 {font-size:11.5pt;margin:15pt 0 7pt;}
h1,h2,h3 {break-after:avoid;}
p {margin:0 0 9pt;orphans:3;widows:3;text-align:justify;}
code {font-family:PaperChinese,monospace;font-size:8.5pt;overflow-wrap:anywhere;}
a {color:#245a9e;text-decoration:none;}
table {width:100%;border-collapse:collapse;font-size:8.2pt;line-height:1.65;}
thead {display:table-header-group;}
th,td {padding:5pt;border-bottom:1px solid #d8dfe9;text-align:left;vertical-align:top;}
th {border-top:1.5px solid #16233a;border-bottom:1px solid #16233a;}
tr {break-inside:avoid;}
tbody tr:last-child td {border-bottom:1.5px solid #16233a;}
"""


def prepare_review() -> dict:
    text = (ROOT / "revision/当前稿技术合理性核查.md").read_text()
    body = markdown.markdown(text, extensions=["tables", "sane_lists"])
    css = (ROOT / "webpage/vendor/paper-fonts.css").read_text()
    covered = set()
    for name in set(re.findall(r'fonts/([^"\)]+)', css)):
        ft = TTFont(ROOT / "webpage/vendor/fonts" / name)
        covered.update(ft.getBestCmap())
        ft.close()
    required = {ord(c) for c in BeautifulSoup(body, "html.parser").get_text() if "\u3400" <= c <= "\u9fff"}
    missing = "".join(chr(c) for c in sorted(required - covered))
    extra_css = ""
    if missing:
        from prepare_display_fonts import font
        data, source = font(missing)
        name = "review-extra-" + hashlib.sha256(data).hexdigest()[:12] + ".ttf"
        (OUT / name).write_bytes(data)
        extra_css = '@font-face{font-family:PaperChinese;src:url("' + name + '") format("truetype");font-display:swap;unicode-range:'
        extra_css += ",".join(f"U+{ord(c):X}" for c in missing) + ";}"
    else:
        source = None
    document = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                '<title>当前稿技术合理性核查</title><link rel="stylesheet" href="../vendor/paper-fonts.css">'
                '<style>' + REVIEW_CSS + extra_css + '</style></head><body><main>' + body + '</main></body></html>')
    (OUT / "technical_review.html").write_text(document)
    return {"extra_cjk_characters": missing, "extra_font_source": source}


def run_browser() -> dict:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "/tmp/pw-libs/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    cli = ["npx", "--yes", "--package", "node@22", "--package", "@playwright/cli", "playwright-cli", "-s=current-pdf-export"]

    def call(args, timeout=55):
        result = subprocess.run(cli + args, env=env, capture_output=True, text=True, timeout=timeout)
        if result.returncode or "### Error" in result.stdout:
            raise RuntimeError(result.stdout[-3500:] + result.stderr[-1200:])
        return result.stdout

    footer = '<div style="width:100%;text-align:center;font-size:8px;color:#667085;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>'
    code = """async(page)=>{
const specs=SPECS;
const output=[];
for (const spec of specs) {
  await page.setViewportSize({width:1440,height:1000});
  const errors=[];const onError=e=>errors.push(String(e));page.on('pageerror',onError);
  await page.goto(spec.url,{waitUntil:'networkidle',timeout:25000});
  await page.evaluate(()=>document.fonts.ready);
  if(spec.style)await page.addStyleTag({content:spec.style});
  await page.emulateMedia({media:'print'});
  await page.evaluate(()=>document.fonts.ready);
  const verification=await page.evaluate(()=>({
    title:document.title,figures:document.querySelectorAll('figure').length,
    tables:document.querySelectorAll('table').length,equations:document.querySelectorAll('.katex-display').length,
    mathErrors:document.querySelectorAll('.katex-error').length,
    brokenImages:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.src),
    fontsReady:document.fonts.status
  }));
  if(verification.mathErrors||verification.brokenImages.length||errors.length)throw new Error(JSON.stringify({verification,errors}));
  await page.pdf({path:spec.path,format:'A4',printBackground:true,preferCSSPageSize:true,
    displayHeaderFooter:true,headerTemplate:'<div></div>',footerTemplate:FOOTER,
    margin:{top:'16mm',right:'16mm',bottom:'18mm',left:'16mm'}});
  output.push({...verification,file:spec.path,pageErrors:errors});
  page.off('pageerror',onError);
  await page.emulateMedia({media:'screen'});
}
return output;
}""".replace("SPECS", json.dumps([
        {"url":BASE_URL + "dhc6_report.html","path":str(OUT / PAPER_NAME),"style":PRINT_CSS},
        {"url":BASE_URL + "downloads/technical_review.html","path":str(OUT / REVIEW_NAME),"style":""},
    ], ensure_ascii=False)).replace("FOOTER", json.dumps(footer))
    call(["open", "about:blank", "--config", str(CONFIG)])
    try:
        output = call(["run-code", code])
        result = output.split("### Result\n", 1)[1].split("\n### ", 1)[0].strip()
        return json.loads(result)
    finally:
        call(["close"])


def main():
    OUT.mkdir(exist_ok=True)
    review_fonts = prepare_review()
    results = run_browser()
    from pypdf import PdfReader
    for entry in results:
        path = Path(entry["file"])
        reader = PdfReader(path)
        text = [p.extract_text() or "" for p in reader.pages]
        entry.update(pages=len(reader.pages),bytes=path.stat().st_size,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     page_text_lengths=[len(t) for t in text],
                     text="\n".join(text))
    assert results[0]["figures"] == 22 and results[0]["tables"] == 16 and results[0]["equations"] == 13
    record = {"source_html_sha256":hashlib.sha256((ROOT / "webpage/dhc6_report.html").read_bytes()).hexdigest(),
              "review_fonts":review_fonts,"exports":results,"training_or_simulation_run":False}
    (ROOT / "revision/current_pdf_export.json").write_text(json.dumps(record,ensure_ascii=False,indent=2) + "\n")
    for item in results:
        print(item["file"],item["pages"],"pages",item["bytes"],"bytes")


if __name__ == "__main__":
    main()
