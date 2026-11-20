"""Remove the parenthetical title note requested by the author, retaining all curves."""
from pathlib import Path
import hashlib
import json
import re

from bs4 import BeautifulSoup, Comment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'flight_control_original_restored_2026-10-25/webpage/paper/teacher_closed_loop.svg'
TARGET = ROOT / 'webpage/paper/teacher_closed_loop.svg'


def trim_title(source, target, letter, number):
    original = source.read_bytes()
    before = BeautifulSoup(original, 'xml')
    after = BeautifulSoup(original, 'xml')
    comment = after.find(string=lambda t: isinstance(t, Comment) and str(t).strip().startswith(letter + ' ·'))
    assert comment is not None
    old_title = str(comment).strip()
    new_title = old_title.split('（', 1)[0]
    group = comment.parent.find('g', recursive=False)
    uses = group.find_all('use')
    opening = next(i for i, u in enumerate(uses) if u.get('xlink:href', '').endswith('-ff08'))
    title_advance = float(re.search(r'translate\(([\d.]+)', uses[opening].get('transform', ''))[1])
    for u in uses[opening:]:
        u.decompose()
    comment.replace_with(Comment(' ' + new_title + ' '))
    # Keep the shortened heading centered at its existing font size.
    width = float(after.svg['viewBox'].split()[2])
    scale = float(re.search(r'scale\(([\d.]+)', group['transform'])[1])
    x = (width - title_advance * scale) / 2
    group['transform'] = re.sub(r'translate\([\d.]+ ', f'translate({x:.6f} ', group['transform'])
    assert [p.get('d') for p in before.find_all('path')] == [p.get('d') for p in after.find_all('path')]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(after), encoding='utf-8')
    (ROOT / f'revision/figure{number}_title_edit.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(original).hexdigest(),
        'display_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'old_title': old_title, 'new_title': new_title,
        'all_curve_and_glyph_paths_unchanged': True,
        'removed_title_glyphs': len(uses) - opening,
    }, ensure_ascii=False, indent=2) + '\n')
    print(new_title)


def main():
    trim_title(SOURCE, TARGET, 'B', 8)
    source = SOURCE.with_name('teacher_training_curves.svg')
    target = TARGET.with_name('teacher_training_curves.svg')
    trim_title(source, target, 'A', 7)


if __name__ == '__main__':
    main()
