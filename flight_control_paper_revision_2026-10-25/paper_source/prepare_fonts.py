"""Fetch a licensed Google Fonts text subset once; manuscript builds stay offline."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import hashlib
import json
import re
import ssl

from fontTools.ttLib import TTFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    text = (ROOT / 'webpage/dhc6_report.md').read_text()
    chars = sorted({c for c in text if ord(c) > 127})
    chunks = [''.join(chars[i:i + 450]) for i in range(0, len(chars), 450)]
    target = ROOT / 'webpage/vendor/fonts'
    target.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context(cafile='/etc/ssl/certs/ca-certificates.crt')

    def fetch(item):
        index, chunk = item
        url = 'https://fonts.googleapis.com/css2?' + urlencode({
            'family': 'Noto Sans SC', 'display': 'swap', 'text': chunk,
        })
        with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}),
                     context=context, timeout=30) as response:
            stylesheet = response.read().decode()
        font_url = re.search(r'url\(([^)]+)\)', stylesheet).group(1)
        with urlopen(font_url, context=context, timeout=30) as response:
            data = response.read()
        name = f'noto-sc-subset-{index:02}.ttf'
        path = target / name
        path.write_bytes(data)
        font = TTFont(path)
        required = {ord(c) for c in chunk if '\u4e00' <= c <= '\u9fff'}
        assert required.issubset(font.getBestCmap()), 'Missing Chinese glyphs'
        ranges = ','.join(f'U+{ord(c):X}' for c in chunk)
        format_name = font.flavor or 'truetype'
        rule = ('@font-face{font-family:PaperChinese;'
                f'src:url(vendor/fonts/{name}) format("{format_name}");'
                'font-weight:400;font-style:normal;font-display:swap;'
                f'unicode-range:{ranges};}}')
        return rule, {'file': name, 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'font_source': font_url, 'chinese_glyphs': len(required)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, enumerate(chunks)))
    css = '\n'.join(rule for rule, _ in results)
    css += '\nbody,h1,h2,h3,h4,table.bt,.toc{font-family:PaperChinese,Georgia,"Noto Sans CJK SC",serif;}\n'
    (ROOT / 'paper_source/chinese_font.css').write_text(css)
    manifest = {'family': 'Noto Sans SC', 'license': 'SIL OFL 1.1',
                'source': 'https://fonts.googleapis.com/css2?family=Noto+Sans+SC',
                'subsets': [record for _, record in results]}
    (ROOT / 'revision/font_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f'Prepared {len(results)} local subsets covering {len(chars)} non-ASCII characters')


if __name__ == '__main__':
    main()
