"""Summarize every frozen controller/plant pair and export reproducible figures."""
from pathlib import Path
import csv
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from fontTools.ttLib import TTFont
import numpy as np

ROOT = Path(__file__).resolve().parent
DISPLAY = ROOT.parent / 'paper_web_display_2026-10-25'
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def data():
    p = read(ROOT / 'protocol.json')
    assert read(ROOT / 'status.json')['state'] == 'complete'
    assert read(ROOT / 'replication_audit.json')['all_reproduced']
    records = [read(f) for f in sorted((ROOT / 'results').rglob('metrics.json'))]
    assert len(records) == p['controller_plant_pairs']
    ids = [p['plant_id'] for p in p['source_objects']]
    matrices = np.full((3, 13, 12), np.nan)
    groups = {g: [] for g in ('own_teacher', 'foreign_teacher', 'student')}
    by_seed = {seed: {g: [] for g in groups} for seed in p['seeds']}
    table = []
    for r in records:
        i = ids.index(r['teacher_source']) if r['controller_kind'] == 'teacher' else 12
        j = ids.index(r['target_plant'])
        k = p['seeds'].index(r['seed'])
        command = [a for a in r['rows'] if a['duration_s'] == 8 and a['category'] == 'command']
        assert len(command) == 18
        rate = sum(a['passed'] for a in command) / 18
        matrices[k, i, j] = rate
        group = 'student' if i == 12 else ('own_teacher' if i == j else 'foreign_teacher')
        groups[group].extend(command)
        by_seed[r['seed']][group].extend(command)
        table.append({'seed': r['seed'], 'controller': 'student' if i == 12 else ids[i],
                      'target_plant': ids[j], 'group': group,
                      'passed_cases': sum(a['passed'] for a in command), 'cases': 18,
                      'pass_rate': rate, 'failed_cases': sum(a['failed'] for a in command)})
    assert np.isfinite(matrices).all()
    def stats(rows):
        finite = [r['rmse_deg_s'] for r in rows if r['rmse_deg_s'] is not None]
        return {'cases': len(rows), 'passed_cases': sum(r['passed'] for r in rows),
                'pass_rate': sum(r['passed'] for r in rows) / len(rows),
                'failed_cases': sum(r['failed'] for r in rows),
                'rmse_median_finite_deg_s': float(np.median(finite)), 'finite_rmse_cases': len(finite)}
    summary = {'source_objects': ids, 'seeds': p['seeds'], 'groups': {g: stats(v) for g, v in groups.items()},
               'by_seed': {seed: {g: stats(v) for g, v in group.items()} for seed, group in by_seed.items()},
               'matrix_mean_pass_rate': matrices.mean(0).tolist(), 'matrix_pass_rate_by_seed': matrices.tolist(),
               'analysis': 'Primary: all 18 command cases per target at 8 s. No cases excluded; deterministic trajectories, not independent population samples.',
               'trace_examples': p['trace_examples_predeclared'], 'trace_seed': p['trace_seed']}
    write(ROOT / 'summary.json', summary)
    with (ROOT / 'controller_plant_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    return p, ids, matrices, summary


def font():
    path = ROOT / 'figure-font.ttf'
    chars = set(''.join(chr(i) for i in range(32, 127)) + '−±·×（）%，：；→。、')
    chars.update(c for c in Path(__file__).read_text() if 0x4e00 <= ord(c) <= 0x9fff)
    if not path.exists() or not {ord(c) for c in chars}.issubset(TTFont(path).getBestCmap()):
        sys.path.insert(0, str(DISPLAY / 'paper_source'))
        from prepare_display_fonts import font as fetch_font
        binary, url = fetch_font(''.join(sorted(chars)))
        path.write_bytes(binary)
        write(ROOT / 'font_manifest.json', {'source': url, 'sha256': hashlib.sha256(binary).hexdigest()})
    return FontProperties(fname=str(path))


def save(fig, name, fp):
    for text in fig.findobj(Text):
        size = text.get_fontsize()
        text.set_fontproperties(fp)
        text.set_fontsize(size)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def matrix_axis(ax, values, ids, title, fontsize=9):
    im = ax.imshow(values * 100, cmap='Blues', vmin=0, vmax=100, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(12), [p.split('-')[-1] for p in ids], rotation=45, ha='right')
    ax.set_yticks(range(13), ['T' + p.split('-')[-1] for p in ids] + ['统一学生'])
    for i in range(13):
        for j in range(12):
            v = values[i, j] * 100
            rgb = np.asarray(im.cmap(im.norm(v))[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ [.2126, .7152, .0722])
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=fontsize,
                    color='white' if luminance <= .179 else 'black')
    for i in range(12):
        ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor='#000000', linewidth=1.25))
    ax.axhline(11.5, color='black', linewidth=1.4)
    for pos in (3.5, 7.5):
        ax.axhline(pos, color='#64748b', lw=.8); ax.axvline(pos, color='#64748b', lw=.8)
    ax.set_xlabel('受控对象 B · 按 L1 / L2 / L3 各四个排列')
    ax.set_ylabel('专用教师的训练对象 A')
    ax.set_title(title, loc='left', pad=12, fontsize=13)
    return im


def plot_matrices(p, ids, matrices, summary, fp):
    fig = plt.figure(figsize=(13.6, 8.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.15], left=.095, right=.97,
                           bottom=.15, top=.86, wspace=.38)
    ax = fig.add_subplot(grid[0, 0])
    im = matrix_axis(ax, matrices.mean(0), ids, '(a) 控制器 × 受控对象 · 三次训练重复的平均通过率')
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.047, pad=.16, aspect=40)
    cbar.set_label('指令测试通过率 (%)'); cbar.set_ticks([0, 25, 50, 75, 100])
    right = fig.add_subplot(grid[0, 1])
    groups = ['own_teacher', 'foreign_teacher', 'student']
    colors = ['#1d4ed8', '#64748b', '#b45309']
    for k, seed in enumerate(p['seeds']):
        vals = [summary['by_seed'][seed][g]['pass_rate'] * 100 for g in groups]
        for i, v in enumerate(vals):
            right.plot(i + (k - 1) * .13, v, marker=['o', 's', '^'][k], color=colors[i],
                       markersize=7, linestyle='none')
    for i, g in enumerate(groups):
        mean = summary['groups'][g]['pass_rate'] * 100
        right.plot([i - .28, i + .28], [mean, mean], color=colors[i], lw=2)
        right.text(i, mean + 3.5,
                   f'{mean:.1f}%', ha='center', fontsize=10)
    right.set(ylim=(0, 105), xlim=(-.55, 2.55), ylabel='通过率 (%)')
    right.set_yticks([0, 25, 50, 75, 100])
    right.set_xticks(range(3), ['教师\n自身对象', '教师\n其他对象', '统一\n学生'])
    right.grid(axis='y', color='#b0b0b0', alpha=.22)
    right.set_title('(b) 每次训练重复的整体结果', loc='left', fontsize=13, pad=12)
    handles = [Line2D([], [], color='#111827', marker=m, ls='none', label=f'重复 {i + 1}')
               for i, m in enumerate(['o', 's', '^'])]
    right.legend(handles=handles, loc='lower right', frameon=False, prop=fp)
    fig.text(.02, .982, '专用教师的跨对象迁移与统一学生的多对象控制', fontsize=18, va='top')
    fig.text(.02, .925, '对角线：教师控制自身训练对象；非对角线：冻结教师迁移到其他对象；末行：同组统一学生。', fontsize=11)
    fig.text(.02, .025, '每个对象 18 条测试指令、8 s 闭环；黑框标出教师自身对象。学生蒸馏同组全部 12 个教师。', fontsize=10, color='#334155')
    save(fig, 'cross_object_matrix', fp)
    for k, seed in enumerate(p['seeds']):
        fig, ax = plt.subplots(figsize=(10, 8.4))
        fig.subplots_adjust(left=.14, right=.93, bottom=.17, top=.9)
        im = matrix_axis(ax, matrices[k], ids, f'训练重复 {k + 1} · seed {seed}')
        fig.colorbar(im, ax=ax, label='指令测试通过率 (%)', fraction=.04, pad=.04)
        save(fig, f'cross_object_matrix_seed_{seed}', fp)


def plot_traces(p, fp):
    fig, axes = plt.subplots(3, 3, figsize=(15.6, 11.2))
    fig.subplots_adjust(left=.065, right=.98, bottom=.09, top=.85, hspace=.55, wspace=.25)
    modes = [('8s', 'test-11-step', '指令 +15 deg/s'),
             ('30s', 'initial_+5', '初始偏差 +5 deg/s'),
             ('30s', 'disturbance_+1', '外力扰动 +1 N')]
    # test-11-step is the positive 15 deg/s case in the frozen 18-command ordering.
    seed = p['trace_seed']
    for row, example in enumerate(p.get('trace_examples_for_display', p['trace_examples_predeclared'])):
        target, foreign = example['target'], example['foreign_teacher']
        own_metrics = read(ROOT / f'results/seed_{seed}/teacher_{target}/{target}/metrics.json')
        own_passed = own_metrics['summary_commands_8s']['passed_episodes']
        ctrls = [(f'teacher_{target}', '#1d4ed8', '-', '自身教师'),
                 (f'teacher_{foreign}', '#64748b', '--', '外对象教师'),
                 ('student_unified', '#b45309', '-.', '统一学生')]
        for col, (horizon, case_id, title) in enumerate(modes):
            ax = axes[row, col]
            reference = None
            expected_input = None
            for name, color, ls, label in ctrls:
                folder = ROOT / f'results/seed_{seed}/{name}/{target}'
                with np.load(folder / f'closed_loop_{horizon}.npz') as tr:
                    index = tr['case_ids'].tolist().index(case_id)
                    time = tr['time']; output = np.degrees(tr['p'][index])
                    reference = np.degrees(tr['reference'][index])
                    command = np.degrees(tr['command'][index])
                    if expected_input is None:
                        expected_input = (time.copy(), command.copy(), reference.copy())
                    else:
                        assert all(np.array_equal(a, b) for a, b in zip(expected_input, (time, command, reference)))
                    ax.plot(time, output, color=color, ls=ls, lw=1.6, label=label)
                    failure = tr['first_failure_s'][index]
                    if np.isfinite(failure):
                        valid = np.flatnonzero(np.isfinite(output))
                        ax.plot(time[valid[-1]], output[valid[-1]], marker='x', color=color, ms=7)
                        ax.text(.98, .06 + .09 * ctrls.index((name, color, ls, label)),
                                label + f' 终止 {failure:.1f} s', color=color, transform=ax.transAxes,
                                ha='right', fontsize=8)
            ax.step(time, command, where='post', color='#111827', lw=1.0, label='外部指令')
            ax.plot(time, reference, color='#444444', ls=':', lw=1.2, label='记录目标')
            own_note = f'自身教师指令通过 {own_passed}/18 · ' if col == 0 else ''
            ax.set_title(f'L{row + 1} · B={target.split("-")[-1]} · {title}\n{own_note}外对象教师 A={foreign.split("-")[-1]}',
                         loc='left', fontsize=11)
            ax.set_xlim(0, 8 if horizon == '8s' else 30)
            ax.grid(True, color='#b0b0b0', alpha=.22, lw=.8)
            ax.set_ylabel('p (deg/s)')
            ax.set_xlabel('时间 (s)')
    handles = [Line2D([], [], color='#111827', lw=1.0, label='外部指令'),
               Line2D([], [], color='#444444', ls=':', lw=1.2, label='记录目标'),
               Line2D([], [], color='#1d4ed8', lw=1.6, label='自身教师'),
               Line2D([], [], color='#64748b', ls='--', lw=1.6, label='外对象教师'),
               Line2D([], [], color='#b45309', ls='-.', lw=1.6, label='统一学生')]
    fig.legend(handles=handles, ncol=5, frameon=False, loc='upper center',
               bbox_to_anchor=(.5, .93), prop=FontProperties(fname=fp.get_file(), size=12))
    fig.text(.02, .98, '验证达标教师的跨对象闭环对照', fontsize=18, va='top')
    fig.text(.02, .02, '每级取原独立验证达标的首个教师；完整对象结果见矩阵。阶跃展示完整 8 s，恢复测试展示完整 30 s。', fontsize=10, color='#334155')
    save(fig, 'cross_object_responses', fp)


def main():
    p, ids, matrices, summary = data()
    illustration = ROOT / 'illustration_protocol_v2.json'
    if illustration.exists():
        p = dict(p)
        p['trace_examples_for_display'] = read(illustration)['examples']
    fp = font()
    with plt.rc_context({'svg.fonttype': 'path', 'pdf.fonttype': 42, 'font.size': 10,
                         'axes.unicode_minus': False, 'axes.spines.top': True, 'axes.spines.right': True}):
        if '--traces-only' not in sys.argv:
            plot_matrices(p, ids, matrices, summary, fp)
        plot_traces(p, fp)
    write(ROOT / 'figure_manifest.json', {
        'source_protocol_sha256': hashlib.sha256((ROOT / 'protocol.json').read_bytes()).hexdigest(),
        'summary_sha256': hashlib.sha256((ROOT / 'summary.json').read_bytes()).hexdigest(),
        'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__},
        'matrix': 'Each cell: mean of three paired-seed pass fractions, all 18 command cases. Individual repetitions shown as points and separate matrices.',
        'traces': 'Illustration v2: first originally validation-qualified teacher in each level, in frozen object order; selection uses training validation records, not test results. First seed; full 8 s / 30 s samples; no smoothing or interpolation. Original failed examples archived as cross_object_diagnostic.',
        'illustration_protocol_sha256': hashlib.sha256(illustration.read_bytes()).hexdigest() if illustration.exists() else None,
        'exports': [{'file': str(f.relative_to(ROOT)), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest()}
                    for f in sorted(OUT.iterdir())]})
    print(json.dumps(summary['groups'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
