"""Plot recorded controller-route-v1 trajectories; never run or train a policy.

The selection rule is in revision/time_series_selection.json. Main figures use
identifier-based examples; the atlas includes every declared object and seed.
"""
from pathlib import Path
import hashlib
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'flight_control_corrected_2026-10-24/experiments/controller_route_20261024'
OUT = ROOT / 'webpage/paper'
SELECTION = json.loads((ROOT / 'revision/time_series_selection.json').read_text())
PROTOCOL = json.loads((BASE / 'protocol.json').read_text())
SEEDS = SELECTION['seeds']
COLORS = {'teacher': '#0072B2', 'student': '#B55C00', 'pid': '#6A3D9A',
          'uncontrolled': '#666666', 'reference': '#161616'}
SEED_STYLES = ['-', '--', '-.']
CACHE, SOURCES, EXPORTED, MAIN_RECORDS = {}, {}, {}, []
MANIFEST = []
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
         'axes.labelsize': 8.5, 'axes.spines.top': False, 'axes.spines.right': False,
         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
         'pdf.fonttype': 42, 'svg.fonttype': 'path', 'savefig.transparent': False}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace(policy, seed, plant, case, horizon=8, target=False):
    if target:
        path = BASE / 'adaptation' / f'seed_{seed}' / f'{policy}_{plant}.npz'
    else:
        directory = 'baselines' if policy in ('pid', 'uncontrolled') else f'{policy}_seed_{seed}'
        path = BASE / 'evaluations' / directory / plant / f'{policy}_{horizon}s.npz'
    if path not in CACHE:
        with np.load(path, allow_pickle=False) as z:
            CACHE[path] = {k: z[k] for k in z.files}
        SOURCES[str(path.relative_to(BASE))] = digest(path)
    z = CACHE[path]
    i = int(case.split('-')[1]) if target else z['case_ids'].tolist().index(case)
    # Radians become deg/s; no smoothing, decimation or missing-value filling.
    result = {'time': z['time'], 'p': np.degrees(z['p'][i]),
              'reference': np.degrees(z['reference'][i]),
              'command': np.degrees(z['command'][i]), 'force': z['requested'][i],
              'first_failure_s': float(z['first_failure_s'][i])}
    return result


def export_trace(trace, policy, seed, plant, case, horizon):
    key = f'{policy}__{seed}__{plant}__{case}__{horizon}s'
    if key not in EXPORTED:
        EXPORTED[key] = trace
        finite = np.isfinite(trace['p']).all() and not np.isfinite(trace['first_failure_s'])
        start = 1.0 if case.startswith('test-') else (2.0 if case.startswith('disturbance_') else 0.)
        active = trace['time'] >= start
        record = {'key': key, 'policy': policy, 'seed': seed, 'object': plant,
                  'case': case, 'horizon_s': horizon,
                  'metric_start_s': start,
                  'rmse_deg_s': float(np.sqrt(np.mean((trace['p'][active]-trace['reference'][active])**2))) if finite else None,
                  'first_failure_s': trace['first_failure_s'] if np.isfinite(trace['first_failure_s']) else None}
        if case.startswith(('initial_', 'disturbance_')) and finite:
            release = 2.3 if case.startswith('disturbance_') else 0.
            record['post_release_rms_deg_s'] = float(np.sqrt(np.mean(trace['p'][trace['time'] >= release]**2)))
        MAIN_RECORDS.append(record)


def curve(ax, trace, policy, field='p', *, style='-', label=None):
    ax.plot(trace['time'], trace[field], color=COLORS[policy], linestyle=style,
            linewidth=1.35, label=label)


def ref(ax, trace):
    ax.plot(trace['time'], trace['reference'], color=COLORS['reference'],
            linestyle=':', linewidth=1.35)


def finish_axes(axes, horizon, force_rows=()):
    for row, row_axes in enumerate(np.atleast_2d(axes)):
        for ax in row_axes:
            ax.set_xlim(0, horizon)
            ax.grid(alpha=.2, linewidth=.5)
            if row in force_rows:
                ax.set_ylim(-23, 23)
                ax.set_yticks([-22, 0, 22])
        if row not in force_rows:
            lo = min(ax.get_ylim()[0] for ax in row_axes)
            hi = max(ax.get_ylim()[1] for ax in row_axes)
            for ax in row_axes:
                ax.set_ylim(lo, hi)
    for ax in np.atleast_2d(axes)[-1]:
        ax.set_xlabel('Time (s)')


def policy_legend(fig, policies, labels=None):
    handles = [Line2D([], [], color=COLORS[p], linestyle=':' if p == 'reference' else ('--' if p == 'teacher' else '-'),
                      linewidth=1.5, label=(labels or {}).get(p, p.capitalize())) for p in policies]
    fig.legend(handles=handles, loc='outside upper center', ncols=len(handles), frameon=False)


def seed_legend(fig, policy, extra=()):
    handles = [Line2D([], [], color=COLORS[policy], linestyle=ls, label=f'{policy.capitalize()} S{i+1}')
               for i, ls in enumerate(SEED_STYLES)]
    handles += [Line2D([], [], color=COLORS[p], linestyle=':' if p == 'reference' else '-', label='Fixed PID' if p=='pid' else p.capitalize()) for p in extra]
    fig.legend(handles=handles, loc='outside upper center', ncols=len(handles), frameon=False)


def save(fig, name, description):
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white')
    MANIFEST.append({'name': name, 'description': description, 'inches': list(fig.get_size_inches()),
                     'outputs': {ext: digest(OUT/f'{name}.{ext}') for ext in ('svg','pdf','png')}})
    plt.close(fig)


def comparison_grid(plant, *, include_teacher=True, atlas=False):
    fig, axes = plt.subplots(4, 3, figsize=(7.0, 6.8), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios':[1.35, .8, 1.35, .8]})
    for col, seed in enumerate(SEEDS):
        for j, case in enumerate(SELECTION['commands']):
            student = read_trace('student', seed, plant, case)
            if not atlas:
                export_trace(student, 'student', seed, plant, case, 8)
            ref(axes[2*j,col], student)
            for policy in (('teacher','student') if include_teacher else ('pid','student')):
                trace = read_trace(policy, seed, plant, case)
                if not atlas:
                    export_trace(trace, policy, seed, plant, case, 8)
                for field, row in [('p',2*j), ('force',2*j+1)]:
                    curve(axes[row,col], trace, policy, field, style='--' if policy=='teacher' else '-')
            axes[0,col].set_title(f'S{col+1}: {seed}')
    for j,label in enumerate(['+15 step','Multisine']):
        axes[2*j,0].set_ylabel(f'{label}\nRoll rate (deg/s)')
        axes[2*j+1,0].set_ylabel('Force (N)')
    finish_axes(axes,8,force_rows=(1,3))
    policy_legend(fig,['reference','teacher','student'] if include_teacher else ['reference','pid','student'], {'pid':'Fixed PID'})
    if atlas:
        fig.legends[0].set_title(f'{plant} | all 3 seeds | recorded 8 s test',prop={'size':10})
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2,3,figsize=(7.0,3.8),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.4,1]})
        for col,(level,plant) in enumerate(SELECTION['source_by_partial_level'].items()):
            for j,seed in enumerate(SEEDS):
                trace=read_trace('teacher',seed,plant,'test-11-step')
                export_trace(trace,'teacher',seed,plant,'test-11-step',8)
                curve(axes[0,col],trace,'teacher',style=SEED_STYLES[j])
                curve(axes[1,col],trace,'teacher','force',style=SEED_STYLES[j])
                if j==0:ref(axes[0,col],trace)
            axes[0,col].set_title(f'{level}: {plant}')
        axes[0,0].set_ylabel('Roll rate (deg/s)');axes[1,0].set_ylabel('Force (N)')
        finish_axes(axes,8,force_rows=(1,));seed_legend(fig,'teacher',('reference',))
        save(fig,'teacher_time_domain','Lowest identifier within each source partial-quality level; all three seeds, +15 deg/s step; full response and force.')

        fig=comparison_grid(SELECTION['source_for_teacher_student'])
        save(fig,'teacher_student_time_domain','Same L2 source object, same commands, and each paired seed; step and multisine responses and force.')

        fig, axes=plt.subplots(4,2,figsize=(7.0,6.6),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.35,.8,1.35,.8]})
        for col,group in enumerate(('id','ood')):
            plant=SELECTION[group]
            for j,case in enumerate(SELECTION['commands']):
                for k,seed in enumerate(SEEDS):
                    trace=read_trace('student',seed,plant,case)
                    export_trace(trace,'student',seed,plant,case,8)
                    curve(axes[2*j,col],trace,'student',style=SEED_STYLES[k])
                    curve(axes[2*j+1,col],trace,'student','force',style=SEED_STYLES[k])
                    if k==0:ref(axes[2*j,col],trace)
                trace=read_trace('pid',None,plant,case)
                export_trace(trace,'pid',None,plant,case,8)
                curve(axes[2*j,col],trace,'pid');curve(axes[2*j+1,col],trace,'pid','force')
            axes[0,col].set_title(f'{group.upper()}: {plant}')
        for j,label in enumerate(['+15 step','Multisine']):
            axes[2*j,0].set_ylabel(f'{label}\nRoll rate (deg/s)');axes[2*j+1,0].set_ylabel('Force (N)')
        finish_axes(axes,8,force_rows=(1,3));seed_legend(fig,'student',('pid','reference'))
        save(fig,'student_unseen_time_domain','Lowest ID and OOD identifiers; all student seeds and fixed PID; two test commands with full responses and force.')

        fig,axes=plt.subplots(3,2,figsize=(7.0,5.5),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.2,1.2,.8]})
        plant=SELECTION['ood']
        for col,case in enumerate(SELECTION['recovery']):
            for k,seed in enumerate(SEEDS):
                trace=read_trace('student',seed,plant,case,30)
                export_trace(trace,'student',seed,plant,case,30)
                curve(axes[0,col],trace,'student',style=SEED_STYLES[k]);curve(axes[1,col],trace,'student',style=SEED_STYLES[k]);curve(axes[2,col],trace,'student','force',style=SEED_STYLES[k])
            trace=read_trace('uncontrolled',None,plant,case,30)
            export_trace(trace,'uncontrolled',None,plant,case,30)
            curve(axes[0,col],trace,'uncontrolled');curve(axes[2,col],trace,'uncontrolled','force')
            axes[0,col].set_title('Initial +5 deg/s' if col==0 else '+1 N pulse: 2.0–2.3 s')
            for ax in axes[:,col]:
                if col==1:ax.axvspan(2,2.3,color='#aaaaaa',alpha=.3,linewidth=0)
        axes[0,0].set_ylabel('Full response\nRoll rate (deg/s)');axes[1,0].set_ylabel('Student detail\nRoll rate (deg/s)');axes[2,0].set_ylabel('Force (N)')
        finish_axes(axes,30,force_rows=(2,));seed_legend(fig,'student',('uncontrolled',))
        save(fig,'recovery_time_domain','Same fixed OOD object as tracking; full response, a separately labeled student-only detail row, and force. All three seeds and the uncontrolled full response are retained; missing divergent tail stays missing.')

        fig,axes=plt.subplots(2,2,figsize=(7.0,4.1),sharex=True,layout='constrained',gridspec_kw={'height_ratios':[1.4,1]})
        target_styles=[('frozen_source_student','#6A3D9A','--','Frozen'),('adapted_source_student','#B55C00','-','Adapted'),('scratch_student_same_budget','#0072B2','-.','Scratch')]
        for col,(group,condition) in enumerate(SELECTION['target_conditions'].items()):
            for arm,color,ls,label in target_styles:
                trace=read_trace(arm,SELECTION['target_seed'],condition,'test-11-step',target=True)
                export_trace(trace,arm,SELECTION['target_seed'],condition,'test-11-step',8)
                for row,field in enumerate(('p','force')):
                    axes[row,col].plot(trace['time'],trace[field],color=color,ls=ls,lw=1.4)
                if arm=='frozen_source_student':ref(axes[0,col],trace)
            axes[0,col].set_title(('Adaptation' if col==0 else 'Held-out')+f': {condition}')
        axes[0,0].set_ylabel('Roll rate (deg/s)');axes[1,0].set_ylabel('Force (N)')
        finish_axes(axes,8,force_rows=(1,))
        fig.legend(handles=[Line2D([],[],color=COLORS['reference'],ls=':',label='Reference')]+[Line2D([],[],color=c,ls=ls,label=l) for _,c,ls,l in target_styles],loc='outside upper center',ncols=4,frameon=False)
        save(fig,'target_time_domain','First condition identifier in each target group; middle listed seed S2, +15 deg/s step; all three target arms at the fixed endpoint.')

        atlas_dir=OUT/'time_domain_atlas';atlas_dir.mkdir(exist_ok=True)
        atlas_index=[]
        with PdfPages(OUT/'time_domain_atlas.pdf',metadata={'Title':'Controller-route-v1: all 60 objects, three seeds','Author':'Pingmo research'}) as pdf:
            for group in ('source','id_test','ood_test'):
                for row in sorted(PROTOCOL[group+'_objects'],key=lambda r:r['plant_id']):
                    plant=row['plant_id'];fig=comparison_grid(plant,include_teacher=group=='source',atlas=True)
                    fig.savefig(atlas_dir/f'{plant}.svg',facecolor='white')
                    pdf.savefig(fig,facecolor='white');plt.close(fig)
                    atlas_index.append({'page':len(atlas_index)+1,'group':group,'object':plant,'partial_level':row['gjb_level'],'svg':f'paper/time_domain_atlas/{plant}.svg'})
        assert len(atlas_index)==60

    payload={}
    index=[]
    for i,(key,trace) in enumerate(EXPORTED.items()):
        prefix=f'trace_{i:03d}'
        for field,value in trace.items():payload[prefix+'__'+field]=value
        index.append({'prefix':prefix,'key':key})
    np.savez_compressed(ROOT/'webpage/evidence/time_series_selected.npz',**payload)
    data={'selection':SELECTION,'trace_index':index,'selected_case_metrics':MAIN_RECORDS,'atlas':atlas_index,
          'source_root':str(BASE),'source_sha256':SOURCES,'transformations':['Radians to degrees for p, reference, command.','Original 50 Hz samples, including t=0 and terminal sample.','No smoothing, decimation, interpolation, or zero fill.','Shared output limits per comparison row; force limits fixed at +/-23 N.'],
          'figure_generator_sha256':digest(Path(__file__)),'figures':MANIFEST,'journal_format':'General research manuscript; no target journal specified.'}
    (ROOT/'webpage/evidence/time_series_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    cards=[]
    for row in atlas_index:
        cards.append(f'<details id="{row["object"]}"><summary>{row["page"]}. {row["object"]} · {row["group"]} · partial L{row["partial_level"]}</summary><img loading="lazy" src="{row["svg"]}" alt="{row["object"]} 的三个种子、阶跃与多正弦的滚转速率和请求力时域对照"></details>')
    gallery='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>全部对象时域验证</title><link rel="stylesheet" href="vendor/paper-fonts.css"><style>body{max-width:1000px;margin:32px auto;padding:0 20px;font:16px/1.7 PaperChinese,sans-serif;color:#16233a}details{border:1px solid #ccd5df;padding:12px;margin:10px 0}summary{cursor:pointer}img{max-width:100%;height:auto}a{color:#245a9e}</style><h1>全部 60 个对象的时域验证</h1><p><a href="dhc6_report.html">返回论文</a> · <a href="paper/time_domain_atlas.pdf">下载 60 页图册</a> · <a href="evidence/time_series_manifest.json">对象清单与数据来源</a></p><p>controller-route-v1：12 个源对象、24 个 ID、24 个 OOD，三个种子全部保留。每个对象显示 +15 deg/s 阶跃和正多正弦测试的完整 8 s 轨迹及请求力；源对象对照对应教师，未见对象对照固定 PID。选取的两类波形用于展开时域表现，完整通过率仍以全部 18 条测试指令计算。图中断线保留原始截尾，不补零、不平滑；打开对象即可查看。</p>'''+''.join(cards)+'</html>'
    (ROOT/'webpage/time_domain_gallery.html').write_text(gallery)
    print(json.dumps({'main_time_figures':len(MANIFEST),'atlas_objects':len(atlas_index),'selected_traces':len(index),'source_files':len(SOURCES)},ensure_ascii=False))


if __name__=='__main__':main()
