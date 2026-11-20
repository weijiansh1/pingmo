"""Match Figure 8's visual style while retaining every original Figure 10 path.

The original SVG is the source: no model, simulation, or new metric is run.
Only the layout, axes, typography and line styling are changed.
"""
from pathlib import Path
import copy
import hashlib
import json
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from fontTools.ttLib import TTFont
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'flight_control_original_restored_2026-10-25/webpage/paper'
SOURCE=ORIGINAL/'teacher_student_compare.svg'
REFERENCE=ORIGINAL/'teacher_closed_loop.svg'
NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
def tag(name):return '{'+NS+'}'+name
def comments(element):return [n.text.strip() for n in element.iter() if n.tag is ET.Comment and n.text]
def load_svg(path):return ET.parse(path,parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))).getroot()
def sha(data):return hashlib.sha256(data).hexdigest()

def main():
    original=load_svg(SOURCE)
    clips={e.attrib['id']:e.find(tag('rect')) for e in original.iter(tag('clipPath'))}
    groups=[e for e in original.iter(tag('g')) if re.fullmatch(r'axes_\d+',e.attrib.get('id',''))]
    assert len(groups)==8
    roles={'#7a8798':'reference','#2f6fd0':'teacher','#e0761f':'student'}
    panels=[]
    for index,group in enumerate(groups):
        curves={}
        for path in group.iter(tag('path')):
            stroke=re.search(r'(?:^|;)\s*stroke:\s*(#[0-9a-f]+)',path.attrib.get('style',''))
            if stroke and stroke[1] in roles and 'clip-path' in path.attrib:
                assert 'transform' not in path.attrib
                curves[roles[stroke[1]]]=copy.deepcopy(path)
        assert set(curves)=={'reference','teacher','student'}
        clip_id=re.search(r'#([^\)]+)',curves['teacher'].attrib['clip-path'])[1]
        rect={k:float(v) for k,v in clips[clip_id].attrib.items()}
        ticks=[]
        for tick in group.iter(tag('g')):
            if not tick.attrib.get('id','').startswith('ytick_'):continue
            labels=comments(tick)
            marker=next(e for e in tick.iter(tag('use')) if 'y' in e.attrib)
            ticks.append((float(labels[0].replace('−','-')),float(marker.attrib['y']),labels[0]))
        slope,intercept=np.polyfit([x[0] for x in ticks],[x[1] for x in ticks],1)
        ymin=(rect['y']+rect['height']-intercept)/slope
        ymax=(rect['y']-intercept)/slope
        labels=comments(group)
        teacher_label=next(x for x in labels if x.startswith('教师 '))
        student_label=next(x for x in labels if x.startswith('学生 '))
        title=['阶跃 +10.5 deg/s','阶跃 +20.9 deg/s','阶跃 +31.4 deg/s',
               '阶跃 −10.5 deg/s','阶跃 −20.9 deg/s','阶跃 −31.4 deg/s','双脉冲','多正弦'][index]
        panels.append({'source_index':index,'title':title,'teacher_label':teacher_label,'student_label':student_label,
                       'curves':curves,'rect':rect,'ticks':ticks,'ylim':(ymin,ymax)})
    font_path=ROOT/'paper_source/figure10-font.ttf'
    font_text='教师与学生的闭环时域对照 参考 教师 学生 阶跃 双脉冲 多正弦 时间 对象 级品质 为 后的跟踪误差 全库 架中位比 八组指令 均保留完整时域' + ''.join(chr(i) for i in range(32,127))+' −·；：×（），。'
    if not font_path.exists() or not {ord(c) for c in font_text}.issubset(TTFont(font_path).getBestCmap()):
        from prepare_display_fonts import font
        data,url=font(''.join(sorted(set(font_text))))
        font_path.write_bytes(data)
        (ROOT/'revision/figure10_font.json').write_text(json.dumps({'source':url,'sha256':sha(data)},indent=2)+'\n')
    font=FontProperties(fname=str(font_path))
    order=[0,3,1,4,2,5,6,7]
    with plt.rc_context({'svg.fonttype':'path','axes.unicode_minus':False,'font.size':12.5}):
        fig,axes=plt.subplots(4,2,figsize=(15.6,13.8),sharex='col')
        fig.subplots_adjust(left=.075,right=.975,bottom=.065,top=.875,hspace=.66,wspace=.23)
        for ax,index in zip(axes.ravel(),order):
            panel=panels[index]
            ax.set_xlim(0,8);ax.set_ylim(*panel['ylim']);ax.set_xticks([0,2,4,6,8])
            ax.set_yticks([r[0] for r in panel['ticks']])
            ax.set_yticklabels([r[2] for r in panel['ticks']])
            ax.set_ylabel('p (deg/s)',fontproperties=font,fontsize=15)
            ax.set_title(panel['title']+'\nRMSE '+panel['teacher_label']+' · '+panel['student_label'],
                         fontproperties=font,fontsize=14,loc='left',pad=7,color='#000000')
            ax.grid(True,color='#b0b0b0',alpha=.22,linewidth=.8);ax.set_axisbelow(True)
            ax.tick_params(axis='both',labelsize=12.5,colors='#000000')
            for label in [*ax.get_xticklabels(),*ax.get_yticklabels()]:label.set_fontproperties(font);label.set_fontsize(12.5)
            for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#000000');spine.set_linewidth(.8)
        for ax in axes[-1]:ax.set_xlabel('时间 (s)',fontproperties=font,fontsize=13)
        handles=[Line2D([],[],color='#111827',ls=':',lw=1.4,label='参考'),
                 Line2D([],[],color='#1d4ed8',lw=2,label='教师'),
                 Line2D([],[],color='#b45309',lw=2,label='学生')]
        fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,
                   prop=FontProperties(fname=str(font_path),size=14),bbox_to_anchor=(.5,.963))
        fig.text(.01,.988,'教师与学生的闭环时域对照',fontproperties=font,fontsize=18.6,weight='bold',va='top')
        fig.text(.01,.018,'对象 train_core-0421（2 级品质）；RMSE 为 2 s 后的跟踪误差，全库 44 架中位比 1.28×；八组指令均保留完整时域。',
                 fontproperties=font,fontsize=11.5,color='#475569')
        fig.canvas.draw()
        boxes=[]
        height=fig.get_figheight()*72
        for ax in axes.ravel():
            b=ax.get_position();boxes.append({'x':b.x0*fig.get_figwidth()*72,'y':height*(1-b.y1),
                'width':b.width*fig.get_figwidth()*72,'height':b.height*height})
        template=ROOT/'revision/figure10_axes_template.svg';fig.savefig(template,facecolor='white');plt.close(fig)
    result=load_svg(template)
    definitions=result.find(tag('defs'))
    new_axes=[e for e in result.iter(tag('g')) if re.fullmatch(r'axes_\d+',e.attrib.get('id',''))]
    verification=[]
    styles={'reference':'fill:none;stroke:#111827;stroke-width:1.4;stroke-dasharray:1.4,2.2;stroke-linecap:butt',
            'teacher':'fill:none;stroke:#1d4ed8;stroke-width:2;stroke-linecap:round',
            'student':'fill:none;stroke:#b45309;stroke-width:2;stroke-linecap:round'}
    for new_index,(ax,index,box) in enumerate(zip(new_axes,order,boxes),1):
        panel=panels[index];old=panel['rect'];sx=box['width']/old['width'];sy=box['height']/old['height']
        tx=box['x']-sx*old['x'];ty=box['y']-sy*old['y']
        clip=ET.SubElement(definitions,tag('clipPath'),{'id':f'figure10_plot_{new_index}'})
        ET.SubElement(clip,tag('rect'),{k:f'{v:.12g}' for k,v in box.items()})
        outer=ET.SubElement(ax,tag('g'),{'clip-path':f'url(#figure10_plot_{new_index})','id':f'original_curves_{index+1}'})
        inner=ET.SubElement(outer,tag('g'),{'transform':f'matrix({sx:.12g} 0 0 {sy:.12g} {tx:.12g} {ty:.12g})'})
        curves=[]
        for role in ['reference','teacher','student']:
            path=copy.deepcopy(panel['curves'][role]);original_d=path.attrib['d']
            path.attrib.pop('clip-path',None);path.set('style',styles[role]);path.set('vector-effect','non-scaling-stroke')
            path.set('data-role',role);path.set('data-original-path-sha256',sha(original_d.encode()));inner.append(path)
            assert path.attrib['d']==panel['curves'][role].attrib['d']
            curves.append({'role':role,'path_sha256':sha(original_d.encode())})
        verification.append({'source_panel':index+1,'display_panel':new_index,'title':panel['title'],
            'teacher_annotation':panel['teacher_label'],'student_annotation':panel['student_label'],
            'xlim':[0,8],'source_clip_rect':old,'display_clip_rect':box,'curves':curves})
    result.insert(0,ET.Element(tag('title')));result[0].text='图 10：与图 8 一致的两列时域图样式'
    output=ROOT/'webpage/paper/teacher_student_compare.svg'
    ET.ElementTree(result).write(output,encoding='utf-8',xml_declaration=True)
    loaded=load_svg(output);all_paths=[p for p in loaded.iter(tag('path')) if p.attrib.get('data-role')]
    assert len(all_paths)==24
    original_hashes=sorted(sha(p.attrib['d'].encode()) for panel in panels for p in panel['curves'].values())
    assert sorted(sha(p.attrib['d'].encode()) for p in all_paths)==original_hashes
    manifest={'source_svg':str(SOURCE),'source_sha256':sha(SOURCE.read_bytes()),'style_reference':str(REFERENCE),
        'reference_sha256':sha(REFERENCE.read_bytes()),'output_sha256':sha(output.read_bytes()),
        'original_panels':8,'display_layout':[4,2],'curves_retained':24,'all_curve_path_coordinates_unchanged':True,
        'metrics':'Copied original SVG annotations verbatim; no recomputation.',
        'data_method':'Original plotted path coordinates retained and affine-mapped to new axes; not a recovery of missing raw samples.',
        'style':['Two columns','Black full axis frames','Light gray grid','Left-aligned two-line titles','Shared top legend','Bottom protocol note'],
        'panels':verification}
    (ROOT/'revision/figure10_style_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Figure 10 restyled: 8 panels, all 24 original curve paths and annotations preserved.')
if __name__=='__main__':main()
