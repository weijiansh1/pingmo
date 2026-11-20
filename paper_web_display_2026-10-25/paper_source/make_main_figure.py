"""Redraw Figure 1 with the user-corrected teacher scope and recorded metrics."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Circle

ROOT=Path(__file__).resolve().parents[1]
font=FontProperties(fname=str(ROOT/'paper_source/diagram-font.ttf'))
ink='#18324F';muted='#45596D';blue='#17649B';edge='#91ADC4'
panels=[
 ('品质模型对象库','5,000 个对象',['8 维品质相关参数','L1 / L2 / L3 / OOD','响应 · 阻尼 · 时延'],r'$G_\theta(s)$'),
 ('特权强化学习教师','50 个专用策略',['46 个通过项目门限','普通观测 + 特权信息','Actor–Critic / PPO'],'每对象 614 万步'),
 ('多教师蒸馏','统一学生控制器',['BC → 6 轮 DAgger','学生运行 → 教师标注','累计数据 → 再训练'],'tiny 3,856\nsmall 75,040'),
 ('目标机型适配','DHC-6 仿真 / ARX',['8 个适配 + 4 个留出','small / 40 步 Adam','适配误差（deg/s）'],'2.91 → 0.59'),
]
with plt.rc_context({'svg.fonttype':'path','pdf.fonttype':42,'font.size':10.5}):
 fig,ax=plt.subplots(figsize=(8.0,3.9));fig.subplots_adjust(left=.015,right=.985,bottom=.015,top=.985)
 ax.set(xlim=(0,100),ylim=(0,50));ax.axis('off')
 def text(x,y,s,size=10.5,weight='normal',color=ink,**kwargs):
  return ax.text(x,y,s,fontproperties=font,fontsize=size,fontweight=weight,color=color,ha='center',va='center',**kwargs)
 text(50,47.2,'基于品质对象族构建统一飞控控制器',14,weight='bold')
 centers=[12.5,37.5,62.5,87.5];width=22.2
 for index,(x,(title,subtitle,body,tail)) in enumerate(zip(centers,panels),1):
  ax.add_patch(FancyBboxPatch((x-width/2,9.8),width,32,boxstyle='round,pad=.3,rounding_size=1.0',facecolor='#F4F8FC',edgecolor=edge,linewidth=1))
  ax.add_patch(Circle((x-width/2+2.6,39.1),1.6,facecolor=blue,edgecolor='none'))
  text(x-width/2+2.6,39.05,str(index),9.5,color='white')
  text(x+1,39.1,title,10.5,weight='bold')
  text(x,33.9,subtitle,11.2,weight='bold',color=blue)
  for y,line in zip([28.7,24.4,20.1],body):text(x,y,line,9.7,color=muted)
  ax.plot([x-8.9,x+8.9],[16.8,16.8],color='#D0DDE8',linewidth=.8)
  text(x,13.15,tail,11.5 if index in (1,4) else 9.7,weight='bold',linespacing=1.35)
 for x in centers[:-1]:
  ax.add_patch(FancyArrowPatch((x+width/2+.5,31.8),(x+25-width/2-.5,31.8),arrowstyle='-|>',mutation_scale=10,linewidth=1.2,color=blue))
 ax.add_patch(FancyBboxPatch((1.1,1.5),97.8,5.8,boxstyle='round,pad=.25,rounding_size=.7',facecolor='#EAF2F8',edgecolor='none'))
 text(50,4.4,'部署反馈：观测与历史 → 学生策略 → 力速率动作 → 执行机构与对象',10.3)
 out=ROOT/'webpage/paper';out.mkdir(parents=True,exist_ok=True)
 for suffix in ['svg','pdf','png']:fig.savefig(out/f'pipeline_framework.{suffix}',dpi=240,facecolor='white')
 plt.close(fig)
 svg=out/'pipeline_framework.svg';data=svg.read_text();at=data.index('> ',data.index('<svg')) if '> ' in data[data.index('<svg'):data.index('<metadata>')] else data.index('>',data.index('<svg'))
 at=data.index('>',data.index('<svg'))+1
 data=data[:at]+'\n<title>品质对象库、特权教师、多轮蒸馏与目标适配</title><desc>50 个对象分别训练专用教师，已有 46 个通过项目门限的记录。学生包含 tiny 3856 参数和 small 75040 参数；图中 40 步适配及 2.91 到 0.59 deg/s 对应 small。</desc>'+data[at:]
 svg.write_text(data)
 (ROOT/'revision/main_figure_data.json').write_text(json.dumps({'source':'Original delivery manuscript and Figure 1; teacher-object count corrected to 50 by the user on 2026-10-25','panels':panels,'change':'Teacher scope aligned to 50 as requested; recorded pass count and other metrics retained. No new experimental results.'},ensure_ascii=False,indent=2)+'\n')
 print('Rendered main figure SVG, PDF and PNG')
