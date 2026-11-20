"""Replot all three frozen seeds, without smoothing or new control experiments."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'webpage/evidence/controller_route_report_summary.json'
OUT = ROOT / 'webpage/paper'
COLORS = ['#0072B2', '#A44D00', '#006B53']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contrast(hex_color):
    rgb = [int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5)]
    lin = [x / 12.92 if x <= .04045 else ((x + .055) / 1.055) ** 2.4 for x in rgb]
    return 1.05 / (.05 + sum(w*x for w, x in zip((.2126, .7152, .0722), lin)))


def main():
    source = json.loads(SOURCE.read_text())
    seeds = ['20261031', '20261101', '20261102']
    arms = ['frozen_source_student', 'adapted_source_student', 'scratch_student_same_budget']
    groups = ['adaptation_condition', 'heldout_condition']
    data = {'protocol': 'controller-route-v1', 'source_sha256': digest(SOURCE),
            'seeds': seeds, 'transfer': {}, 'distillation': {}}
    manifest = {'destination': 'Provisional research manuscript/web; no journal specified',
                'width_mm': 178, 'source': str(SOURCE), 'source_sha256': digest(SOURCE),
                'generator_sha256': digest(Path(__file__)),
                'uncertainty': 'All three policy seeds shown; no CI or episode-independence assumption',
                'transformations': 'pass percentages = passed / all declared episodes * 100; no filtering or smoothing',
                'missing': 'No missing plotted observations; failed cases stay in pass-rate denominators',
                'contrast_to_white': {c: contrast(c) for c in COLORS}, 'figures': []}
    for seed in seeds:
        row = source['seeds'][seed]
        data['transfer'][seed] = {
            group: {arm: {key: row['transfer'][arm][group][key]
                          for key in ['passed_episodes', 'episodes', 'pass_rate']}
                    for arm in arms} for group in groups}
        data['distillation'][seed] = row['student_rounds']
        assert len(row['student_rounds']) == 7
        for group in groups:
            for arm in arms:
                item = data['transfer'][seed][group][arm]
                assert abs(item['pass_rate'] - item['passed_episodes']/item['episodes']) < 1e-12
    (ROOT / 'webpage/evidence/story_figure_data.json').write_text(json.dumps(data, indent=2)+'\n')

    def save(fig, name):
        files = {}
        for ext in ['svg', 'pdf', 'png']:
            path = OUT / f'{name}.{ext}'
            fig.savefig(path, dpi=300, facecolor='white', transparent=False)
            files[ext] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
        with Image.open(OUT / f'{name}.png') as im:
            pixels, dpi = list(im.size), list(im.info['dpi'])
        page = PdfReader(OUT / f'{name}.pdf').pages[0]
        manifest['figures'].append({'name': name, 'files': files, 'png_pixels': pixels,
            'png_dpi': dpi, 'pdf_points': [float(page.mediabox.width), float(page.mediabox.height)]})
        plt.close(fig)

    style = {'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.labelsize': 8.5, 'pdf.fonttype': 42, 'svg.fonttype': 'path',
             'axes.edgecolor': '#283443', 'text.color': '#283443', 'axes.labelcolor': '#283443'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(178/25.4, 89/25.4), sharey=True, layout='constrained')
        arm_styles = [('#586574', 'o', ':'), (COLORS[0], 's', '-'), (COLORS[1], '^', '--')]
        labels = ['Frozen', 'Pretrained + adapt', 'Scratch + adapt']
        for ax, group, title in zip(axes, groups, ['(a) 8 adaptation conditions', '(b) 4 held-out conditions']):
            for arm, label, (color, marker, ls) in zip(arms, labels, arm_styles):
                y = [100*data['transfer'][seed][group][arm]['pass_rate'] for seed in seeds]
                ax.plot(range(3), y, label=label, color=color, marker=marker,
                        linestyle=ls, markersize=5, linewidth=1.4)
            ax.set(xticks=range(3), xticklabels=['S1', 'S2', 'S3'], ylim=(0,100),
                   xlim=(-.35, 2.35), yticks=range(0,101,20), xlabel='Policy seed', title=title)
            ax.grid(axis='y', color='#e2e6ec', linewidth=.7)
            ax.set_axisbelow(True)
        axes[0].set_ylabel('Episodes passing all gates (%)')
        axes[0].legend(frameon=False, loc='upper left', fontsize=7.6)
        for x, seed in enumerate(seeds):
            pair=data['transfer'][seed]['heldout_condition']
            diff=100*(pair['adapted_source_student']['pass_rate']-pair['scratch_student_same_budget']['pass_rate'])
            axes[1].text(x, 91, f'+{diff:.1f} pp', ha='center', fontsize=8, color='#283443')
        save(fig,'transfer_budget')

        fig, ax = plt.subplots(figsize=(178/25.4, 85/25.4), layout='constrained')
        for seed, color, marker, ls, short in zip(seeds,COLORS,['o','s','^'],['-','--',':'],['S1','S2','S3']):
            rows=data['distillation'][seed]
            x=[r['round'] for r in rows]; y=[r['validation_score_deg_s'] for r in rows]
            ax.plot(x,y,color=color,marker=marker,linestyle=ls,linewidth=1.5,label=short)
            j=int(np.argmin(y))
            ax.plot(x[j],y[j],marker='o',markersize=12,markerfacecolor='none',
                    markeredgecolor=color,markeredgewidth=1.3,linestyle='none')
        ax.set(xticks=range(7),xticklabels=['BC','1','2','3','4','5','6'],ylim=(0,12),
               xlabel='New closed-loop collection round',ylabel='Validation mean RMSE (deg/s)')
        ax.grid(axis='y',color='#e2e6ec',linewidth=.7);ax.set_axisbelow(True)
        ax.legend(frameon=False,title='Policy seed',loc='upper right')
        save(fig,'distillation_rounds')
    manifest['matplotlib_version'] = matplotlib.__version__
    (ROOT/'revision/figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Rendered two figures from all frozen seeds; JSON data and export provenance saved.')


if __name__=='__main__':
    main()
