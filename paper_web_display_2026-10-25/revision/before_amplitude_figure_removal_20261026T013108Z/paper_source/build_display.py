"""Apply math typography and Figure 1 presentation to the original HTML only."""
from pathlib import Path
from bs4 import BeautifulSoup,Tag
import ast,base64,hashlib,json,re,subprocess
from sae_story import rewrite_sae_story
from cross_object_section import add_cross_object_comparison
from display_identifiers import apply_identifiers,version_assets
from integrate_sae import integrate_sae
from closed_loop_explanation import add_closed_loop_explanation
from editorial_revision import revise_manuscript,polish_after_numbering
from observation_prose import rewrite_observation_prose
from distillation_metric import add_distillation_metric,DISTILLATION_RATIO_TEX
from results_revision import revise_results_section

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent/'flight_control_original_restored_2026-10-25/webpage'
EQUATIONS=[
None, # The fixed reference-model equation was removed at the user's request.
r'\begin{gathered}o_t=\operatorname{Norm}\!\bigl([p_{\mathrm{cmd}},p_{\mathrm{ref}},p,e,\int e\,dt,\dot p,\dot p_{\mathrm{ref}},F_{\mathrm{req}},F_{\mathrm{applied}},h_{t-31:t},a_{t-7:t-1}]\bigr)\in\mathbb R^{176},\\[0.4em]o_t^{\mathrm T}=[o_t,z_t]\in\mathbb R^{188},\qquad o_t^{\mathrm S}=o_t.\end{gathered}',
r'G_\theta(s)=L_{F_a}\,\frac{s(s^2+2\zeta_\phi\omega_\phi s+\omega_\phi^2)}{(s-\lambda_s)(s+1/T_R)(s^2+2\zeta_d\omega_ds+\omega_d^2)}\,e^{-\tau_ps}.',
r'\begin{aligned}\operatorname{RMSE}_p&=\sqrt{\frac1M\sum_{t=1}^M(p_{\mathrm{ref},t}-p_t)^2},\\\operatorname{Peak}&=\max_t|p_{\mathrm{ref},t}-p_t|,\\\operatorname{TV}_F&=\frac1T\sum_{t=1}^M|F_{\mathrm{req},t}-F_{\mathrm{req},t-1}|.\end{aligned}',
r'\theta^{(i)}\sim q(\theta\mid c_i),\qquad g_i=\operatorname{Grade}_{\mathrm{GJB}}(\theta^{(i)},c_i).',
r'\begin{gathered}F_{\mathrm{req},t+1}=\operatorname{clip}\!\left(F_{\mathrm{req},t}+(88\,\mathrm{N/s})a_t\Delta t,-22\,\mathrm N,22\,\mathrm N\right),\\\Delta t=0.02\,\mathrm s.\end{gathered}',
r'r_t=-50\left[\left(\frac{e_t}{p_{\mathrm{scale}}}\right)^2\Delta t+0.02\left(\frac{F_{\mathrm{applied},t}}{22\,\mathrm N}\right)^2\Delta t+0.02a_t^2\right].',
r'\begin{aligned}\delta_t&=r_t+\gamma V(s_{t+1})-V(s_t),\\\widehat A_t&=\delta_t+\gamma\lambda\widehat A_{t+1},\qquad\gamma=0.995,\quad\lambda=0.95.\end{aligned}',
r'\begin{gathered}\mathcal L_{\mathrm{clip}}(w)=-\mathbb E_t\!\left[\min\!\left(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right],\\\epsilon=0.2.\end{gathered}',
r'\begin{aligned}\mathcal L_{\mathrm{total}}&=\mathcal L_{\mathrm{clip}}+0.5\,\mathbb E[(V_v(s_t)-R_t)^2]-0.001\,\mathcal H(\pi_w),\\R_t&=\widehat A_t+V_v(s_t).\end{aligned}',
r'\begin{aligned}D_0&=\{(o_t,\mu_{\mathrm T}(o_t,z_t))\},\\D_{k+1}&=D_k\cup\{(o_t^{\mathrm S,k},\mu_{\mathrm T}(o_t^{\mathrm S,k},z_t))\}.\end{aligned}',
r'w_{\mathrm S}^{*}=\operatorname*{arg\,min}_w\frac1{|D_K|}\sum_{(o,u_{\mathrm T})\in D_K}\|\mu_{\mathrm S}(o;w)-u_{\mathrm T}\|^2.',
r'\begin{aligned}e_t(W)&=p_{\mathrm{ref},t}-p_t(W;\widehat G_{\mathrm{ARX}}),\\\mathcal L_{\mathrm{DHC6}}(W)&=\frac1{N_{t\ge2\,\mathrm s}}\sum_{t\ge2\,\mathrm s}e_t(W)^2,\\W_{k+1}&=\operatorname{AdamUpdate}(W_k,\nabla_W\mathcal L_{\mathrm{DHC6}};\eta=3\times10^{-5}).\end{aligned}',
DISTILLATION_RATIO_TEX,
]
GREEK={'θ':r'\theta','τ':r'\tau','ω':r'\omega','ζ':r'\zeta','λ':r'\lambda','γ':r'\gamma','δ':r'\delta','μ':r'\mu','π':r'\pi','ε':r'\epsilon','ρ':r'\rho','η':r'\eta','φ':r'\phi','ṗ':r'\dot p','Â':r'\widehat A','−':'-','≥':r'\ge ','≤':r'\le '}
ROMAN={'ref','cmd','req','applied','scale','clip','total','Fa','GJB','ARX','DHC6'}
def tex_text(value):
    for a,b in GREEK.items():value=value.replace(a,b)
    return re.sub(r'\b(ref|cmd|req|applied|scale|clip|total|Fa|GJB|ARX|DHC6)\b',lambda m:r'\mathrm{'+m[0]+'}',value)
def node_tex(node):
    if not isinstance(node,Tag):return tex_text(str(node))
    text=''.join(node_tex(c) for c in node.children)
    if node.name=='sub':return '_{'+text+'}'
    if node.name=='sup':return '^{'+text+'}'
    return text
STYLE='''
/* Presentation overlay: local math rendering; original paper layout retained. */
.katex,.katex *{box-sizing:content-box;}
.eq{display:flex;align-items:center;gap:16px;position:relative;padding:18px 16px;margin:20px 0;text-align:left;max-width:100%;overflow:hidden;font-size:16px;}
.eq .math-content{flex:1;min-width:0;overflow-x:auto;overflow-y:hidden;padding:5px 2px;}
.eq .katex-display{margin:0;text-align:center;}
.eq .katex-display>.katex{display:block;max-width:100%;}
.eq .eqno{position:static;transform:none;flex:none;align-self:center;font-size:14px;color:#526275;}
.math-inline{white-space:nowrap;}
.math-inline>.katex{font-size:1.02em;}
main figure:first-of-type img{width:100%;height:auto;}
figure a.figure-zoom{display:block;line-height:0;}
.parameter-table-wrap{max-width:100%;overflow-x:auto;}
table.bt.parameter-table{min-width:640px;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink);line-height:1.65;}
table.bt.parameter-table thead th{border-top:0;border-bottom:1px solid var(--ink);text-align:left;}
table.bt.parameter-table tbody td,table.bt.parameter-table tbody tr:last-child td{border:0;vertical-align:top;text-align:left;}
table.bt.parameter-table th:nth-child(1){width:12%;}table.bt.parameter-table th:nth-child(2){width:23%;}
table.bt.parameter-table th:nth-child(3){width:49%;}table.bt.parameter-table th:nth-child(4){width:16%;}
table.bt.observation-table{min-width:560px;border-top:1.5px solid var(--ink);border-bottom:1.5px solid var(--ink);line-height:1.65;}
table.bt.observation-table thead th{border-top:0;border-bottom:1px solid var(--ink);}
table.bt.observation-table tbody td,table.bt.observation-table tbody tr:last-child td{border:0;vertical-align:top;}
table.bt.observation-table th:first-child{width:18%;}table.bt.observation-table th:last-child{width:10%;}
table.bt.distillation-results-table{min-width:640px;}
table.bt.distillation-results-table th{white-space:nowrap;}
@media(max-width:640px){.eq{padding:12px 9px;gap:8px;font-size:12.5px;}.eq .eqno{font-size:12px;}}
@media print{.eq{break-inside:avoid;overflow:visible;}.eq .math-content{overflow:visible;}figure{break-inside:avoid;}}
'''
def render(expressions):
    return json.loads(subprocess.run(['node',str(ROOT/'paper_source/render_math.cjs')],input=json.dumps(expressions),text=True,capture_output=True,check=True).stdout)
def add_parameter_table(soup,anchor):
    rows=[
        ('<i>L</i><sub><i>F</i><sub>a</sub></sub>','滚转控制效能<br>rad/(N·s²)','表示输入力对滚转角加速度的作用强度，决定输入的控制效能。','采样'),
        ('<i>λ</i><sub>s</sub>','螺旋模态特征根<br>s⁻¹','对应极点 <i>s</i> = <i>λ</i><sub>s</sub>；负值表示该模态衰减，正值表示发散。','采样'),
        ('<i>T</i><sub>R</sub>','滚转模态时间常数<br>s','对应极点 −1/<i>T</i><sub>R</sub>；取值越小，该模态衰减越快。','采样'),
        ('<i>ζ</i><sub>d</sub>','荷兰滚模态阻尼比<br>无量纲','与模态频率共同决定振荡的衰减速度。','采样'),
        ('<i>ω</i><sub>d</sub>','荷兰滚模态自然频率<br>rad/s','决定该振荡模态的时间尺度。','采样'),
        ('<i>R</i><sub>ω</sub>','零点频率比<br>无量纲','定义为 <i>ω</i><sub>φ</sub>/<i>ω</i><sub>d</sub>，用于确定分子零点对的频率参数。','采样'),
        ('<i>R</i><sub>ζ</sub>','零点阻尼比例系数<br>无量纲','定义为 <i>ζ</i><sub>φ</sub>/<i>ζ</i><sub>d</sub>，用于确定分子零点对的阻尼参数。','采样'),
        ('<i>τ</i><sub>p</sub>','输入运输时延<br>s','表示输入作用于对象前的纯时间滞后。','采样'),
        ('<i>ω</i><sub>φ</sub>','分子零点对的频率参数<br>rad/s','由 <i>ω</i><sub>φ</sub> = <i>R</i><sub>ω</sub><i>ω</i><sub>d</sub> 得到，与极点共同影响瞬态响应。','派生'),
        ('<i>ζ</i><sub>φ</sub>','分子零点对的阻尼参数<br>无量纲','由 <i>ζ</i><sub>φ</sub> = <i>R</i><sub>ζ</sub><i>ζ</i><sub>d</sub> 得到，用于确定分子零点对的位置。','派生'),
    ]
    table=soup.new_tag('table',attrs={'class':'bt parameter-table','id':'model-parameter-table'})
    caption=soup.new_tag('caption');caption.string='表 0　参数化滚转对象的参数说明';table.append(caption)
    head=soup.new_tag('thead');tr=soup.new_tag('tr')
    for label in ['符号','参数名称与单位','含义与动态作用','设定方式']:
        cell=soup.new_tag('th',scope='col');cell.string=label;tr.append(cell)
    head.append(tr);table.append(head);body=soup.new_tag('tbody')
    for row in rows:
        tr=soup.new_tag('tr')
        for value in row:
            cell=soup.new_tag('td');cell.append(BeautifulSoup(value,'html.parser'));tr.append(cell)
        body.append(tr)
    table.append(body);wrapper=soup.new_tag('div',attrs={'class':'parameter-table-wrap'});wrapper.append(table)
    anchor.insert_after(wrapper)
    note=soup.new_tag('p',attrs={'class':'small','id':'model-parameter-note'})
    note.append(BeautifulSoup('前八项构成对象采样向量 <i>θ</i>，后两项由比值关系派生。单位按输入力 N、输出角速度 rad/s 定义；表中的参数描述被控对象，控制器网络权重由训练学习。','html.parser'))
    wrapper.insert_after(note)
def number_tables(soup):
    """Number captions in reading order and update their prose references."""
    mapping={};captions=[]
    for index,table in enumerate(soup.find_all('table'),1):
        caption=table.find('caption')
        if caption is None:continue
        match=re.match(r'表\s*(\d+)',caption.get_text())
        if match:
            mapping[int(match[1])]=index;captions.append((caption,index))
    def replace(node,new_number):
        original=str(node);value=re.sub(r'表\s*\d+',f'表 {new_number}',original,count=1)
        span=soup.new_tag('span',attrs={'data-original-table-label':base64.b64encode(original.encode()).decode()});span.string=value
        node.replace_with(span)
    for caption,index in captions:
        for node in list(caption.find_all(string=re.compile(r'表\s*\d+'))):replace(node,index)
    for paragraph in soup.select('p'):
        for node in list(paragraph.find_all(string=re.compile(r'表\s*\d+'))):
            old=int(re.search(r'表\s*(\d+)',str(node))[1])
            if old in mapping:replace(node,mapping[old])
def define_control_task(soup):
    """Present sampled dynamics and command tracking without the fixed model."""
    for index,element in enumerate(soup.select('.eq'),1):
        element['data-source-equation']=str(index)
    task=soup.select_one('#s21');objects=soup.select_one('#s22');metrics=soup.select_one('#s23')
    task_block=[]
    for element in task.find_next_siblings():
        if element==objects:break
        task_block.append(element)
    reference=soup.select_one('[data-source-equation="1"]')
    reference.decompose()
    first=next(element for element in task_block if element.name=='p')
    first.clear()
    first.append(BeautifulSoup('被控量为滚转角速度 <i>p</i>，外部任务给定指令 <i>p</i><sub>cmd</sub>。控制器需要完成指令跟踪，并抑制指令响应、初始偏差和外部扰动引起的不良振荡。跟踪误差记为 <i>e</i><sub>t</sub> = <i>p</i><sub>ref,t</sub> − <i>p</i><sub>t</sub>，其中 <i>p</i><sub>ref</sub> 表示各项实验记录中的目标速率信号；控制器学习根据观测与历史产生控制动作。','html.parser'))
    last=task_block[-1]
    assert last.name=='p'
    for node in list(last.children):
        if isinstance(node,str) and '由二阶参考模型' in node:
            # Remove the final reference explanation from the observation
            # paragraph; its corresponding italic p_ref nodes precede it.
            previous=node.previous_sibling
            while previous is not None and isinstance(previous,Tag) and previous.name in ('i','sub'):
                before=previous.previous_sibling;previous.extract();previous=before
            node.extract()
    # Move the existing object definition first; retain IDs for existing links.
    object_nodes=[objects]
    for element in objects.find_next_siblings():
        if element==metrics:break
        object_nodes.append(element)
    for element in object_nodes:task.insert_before(element.extract())
    objects.string='2.1 参数化滚转对象';task.string='2.2 控制任务与接口变量'
    sampled=soup.new_tag('p',id='sampled-modal-parameters')
    sampled.append(BeautifulSoup('对象的振荡模态频率 <i>ω</i><sub>d</sub>、阻尼比 <i>ζ</i><sub>d</sub>及其他动力学参数随 <i>θ</i> 共同采样，构成动态特性不同的对象族。','html.parser'))
    object_nodes[-1].insert_after(sampled)
    add_parameter_table(soup,object_nodes[-1])
    for node in soup.find_all(string=lambda value:value and '第 2.2 节定义' in value):
        node.replace_with(str(node).replace('第 2.2 节定义','第 2.1 节定义'))
    task_link=soup.select_one('.toc a[href="#s21"]');object_link=soup.select_one('.toc a[href="#s22"]')
    if task_link and object_link:
        task_link.string='2.2 控制任务与接口变量';object_link.string='2.1 参数化滚转对象'
        task_link.insert_before(object_link.extract())
    settings=soup.select_one('#s4');intro=settings.find_next_sibling('p')
    data_heading=soup.new_tag('h3',id='experiment-data-protocol');data_heading.string='4.1 对象划分与评价协议'
    intro.insert_after(data_heading)
    settings_link=soup.select_one('.toc a[href="#s4"]')
    if settings_link:
        link=soup.new_tag('a',href='#experiment-data-protocol',attrs={'class':'sub'});link.string='4.1 对象划分与评价协议'
        settings_link.insert_after(link)
    for node in soup.find_all(string=lambda value:value and '二阶参考模型保留在回路内，视为飞控系统的一部分' in value):
        node.replace_with(str(node).replace('二阶参考模型保留在回路内，视为飞控系统的一部分','评价沿用各策略的既有闭环与所记录的输入输出'))
    cross_object=soup.select_one('#s532')
    if cross_object:cross_object.string='5.3.2 跨对象闭环控制性能评估'
    cross_object_link=soup.select_one('.toc a[href="#s532"]')
    if cross_object_link:cross_object_link.string='5.3.2 跨对象闭环控制性能评估'
def align_teacher_scope(soup):
    """Use the user's corrected 50-object scope without recomputing metrics."""
    count=re.compile(r'(?<![\d.])49(?![\d.])')
    for node in list(soup.find_all(string=count)):
        if node.parent.name not in ('script','style'):
            node.replace_with(count.sub('50',str(node)))
    for cell in soup.find_all('td'):
        if cell.get_text(strip=True)=='46/50 通过项目门限':
            cell.string='已有 46 个通过项目门限'
    result=soup.select_one('#s52').find_next_sibling('p')
    result.string=(result.get_text()
        .replace('50 个对象分别训练后，46 个通过','本文按 50 个对象展示逐对象教师训练，现有记录中 46 个通过')
        .replace('未通过对象为','未通过记录包括')
        .replace('；统计保留全部训练尝试，未通过对象不从分母剔除。','。'))
    coverage=soup.select_one('#s51').find_next_sibling('p')
    coverage.string=coverage.get_text().replace('和 50 个教师训练对象的位置','和已有记录的教师训练对象位置')
    soup.find_all('figure')[4].figcaption.string='图 5　参数边际分布及已有记录的教师选取位置。'
def main():
    manifests=[]
    for filename in ['dhc6_report.html','dhc6_paper.html']:
        original=(ORIGINAL/filename).read_bytes();soup=BeautifulSoup(original,'html.parser')
        define_control_task(soup)
        rewrite_observation_prose(soup)
        rewrite_sae_story(soup)
        align_teacher_scope(soup)
        has_cross_object_comparison=add_cross_object_comparison(soup)
        figure_integration_mapping=integrate_sae(soup)
        revise_manuscript(soup)
        add_distillation_metric(soup)
        apply_identifiers(soup)
        content_baseline=BeautifulSoup(str(soup),'html.parser')
        eqs=soup.select('.eq');assert len(eqs)==13
        equation_order=[int(element['data-source-equation']) for element in eqs]
        assert equation_order==[3,2,4,14,5,6,7,8,9,10,11,12,13]
        formulas=[{'tex':EQUATIONS[index-1],'display':True} for index in equation_order]
        inline=[]
        for element in soup.find_all('i'):
            if element.find_parent(class_='eq') or element.find_parent(['sub','sup']):continue
            value=element.get_text()
            if not re.fullmatch(r'[A-Za-zθτωζλγδμπερφÂṗ]',value):continue
            nodes=[element];following=element.next_sibling
            while isinstance(following,Tag) and following.name in ['sub','sup']:
                nodes.append(following);following=following.next_sibling
            tex=''.join(node_tex(n) for n in nodes)
            inline.append((nodes,tex,''.join(str(n) for n in nodes)))
            formulas.append({'tex':tex,'display':False})
        rendered=render(formulas)
        for index,(element,html) in enumerate(zip(eqs,rendered[:len(eqs)]),1):
            element.clear();element['data-equation']=str(index)
            math=soup.new_tag('div',attrs={'class':'math-content'});math.append(BeautifulSoup(html,'html.parser'))
            number=soup.new_tag('span',attrs={'class':'eqno'});number.string=f'({index})'
            element.append(math);element.append(number)
        for (nodes,tex,original_html),html in zip(inline,rendered[len(eqs):]):
            span=soup.new_tag('span',attrs={'class':'math-inline','data-original-math':base64.b64encode(original_html.encode()).decode()})
            span.append(BeautifulSoup(html,'html.parser'))
            nodes[0].replace_with(span)
            for node in nodes[1:]:node.extract()
        repaired_tables=[]
        for leaked in list(soup.find_all(string=lambda value:value and 'SECTIONS.append(tab(' in value)):
            code=str(leaked).strip()
            expression=ast.parse(code).body[0].value
            assert isinstance(expression,ast.Call) and expression.func.attr=='append'
            call=expression.args[0]
            assert isinstance(call,ast.Call) and call.func.id=='tab' and len(call.args)==3
            caption,headers,rows=[ast.literal_eval(arg) for arg in call.args]
            assert all(len(row)==len(headers) for row in rows)
            table=soup.new_tag('table',attrs={'class':'bt','id':'pid-student-comparison','data-repaired-table':'true'})
            cap=soup.new_tag('caption');cap.string=caption;table.append(cap)
            head=soup.new_tag('thead');tr=soup.new_tag('tr')
            for value in headers:
                th=soup.new_tag('th',scope='col');th.string=value;tr.append(th)
            head.append(tr);table.append(head);body=soup.new_tag('tbody')
            for row in rows:
                tr=soup.new_tag('tr')
                for value in row:
                    td=soup.new_tag('td');td.string=value;tr.append(td)
                body.append(tr)
            table.append(body);leaked.replace_with(table)
            assert [[c.get_text() for c in tr.find_all('td')] for tr in body.find_all('tr')]==rows
            repaired_tables.append({'caption':caption,'headers':headers,'rows':rows,'original_generator_text':code})
        assert len(repaired_tables)==1
        number_tables(soup)
        for href in ['vendor/katex/katex.min.css','vendor/paper-fonts.css']:
            soup.head.append(soup.new_tag('link',rel='stylesheet',href=href))
        style=soup.new_tag('style',id='math-display-fix');style.string=STYLE;soup.head.append(style)
        figure=soup.find('figure');image=figure.find('img');image['src']='paper/pipeline_framework.svg?v=teacher50-20261025'
        image['alt']='原研究四步流程：品质对象库、特权强化学习教师、多教师蒸馏，以及 DHC-6 目标适配。tiny 与 small 参数量分别标明；40 步适配对应 small。'
        link=soup.new_tag('a',href='paper/pipeline_framework.svg?v=teacher50-20261025',target='_blank',attrs={'class':'figure-zoom','title':'点击查看主图大图','aria-label':'查看完整主图'})
        image.wrap(link)
        figure.figcaption.string='图 1　原研究的四步流程：品质对象库（5,000 个对象）→ PPO 教师（50 个；已有 46 个通过项目门限的记录）→ 多教师蒸馏（tiny 3,856 参数、small 75,040 参数）→ DHC-6 仿真适配（12 个状态；图中 40 步及 2.91→0.59 deg/s 对应 small）。'
        comparison=soup.find_all('figure')[9]
        comparison.find('img')['src']='paper/teacher_student_compare.svg?v=figure8-style-20261025'
        comparison.figcaption.string=comparison.figcaption.get_text().replace('虚线为参考，蓝线为教师，橙线为学生','黑色点线为参考，蓝线为教师，橙线为学生').replace('面板内给出','各面板标题中给出')
        soup.find_all('figure')[7].find('img')['src']='paper/teacher_closed_loop.svg?v=title-trim-20261025'
        soup.find_all('figure')[6].find('img')['src']='paper/teacher_training_curves.svg?v=title-trim-20261025'
        version_assets(soup)
        destination=ROOT/'webpage'/filename;destination.write_text(str(soup))
        # Recover only styled inline fragments, then check every original prose
        # paragraph/table/header outside the explicitly edited Figure 1 caption.
        restored=BeautifulSoup(str(soup),'html.parser')
        for span in restored.select('[data-original-math]'):
            fragment=BeautifulSoup(base64.b64decode(span['data-original-math']).decode(),'html.parser')
            span.replace_with(fragment)
        for table in restored.select('[data-repaired-table]'):table.decompose()
        for span in restored.select('[data-original-table-label]'):
            span.replace_with(base64.b64decode(span['data-original-table-label']).decode())
        before=content_baseline
        def prose(doc):
            return [re.sub(r'\s+',' ',e.get_text()).strip() for e in doc.select('h1,h2,h3,h4,p,td,th') if not e.find_parent(class_='eq')]
        assert prose(before)==prose(restored),'Unexpected prose/data change'
        old_images=[i['src'].split('?')[0] for i in before.find_all('img')]
        new_images=[i['src'].split('?')[0] for i in soup.find_all('img')]
        assert old_images==new_images and len(new_images)==22
        assert 'SECTIONS.append' not in soup.get_text()
        assert not soup.select('#reference-model-equation,#reference-signal-settings')
        assert not any(r'\omega_{\mathrm{ref}}' in a.get_text() or r'\zeta_{\mathrm{ref}}' in a.get_text() for a in soup.select('annotation'))
        add_closed_loop_explanation(soup)
        polish_after_numbering(soup)
        revise_results_section(soup)
        destination.write_text(str(soup))
        new_images=[i['src'].split('?')[0] for i in soup.find_all('img')]
        assert len(new_images)==len(old_images)+1
        manifests.append({'file':filename,'original_sha256':hashlib.sha256(original).hexdigest(),'display_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'display_formulas':len(eqs),'inline_formulas':len(inline),'paragraphs_tables_headers_match_authorized_reference_edit':True,'image_count':len(new_images),'closed_loop_explanation_added':True,'repaired_table6_values_preserved':True,'parameter_table_rows':10,'tables_numbered_in_reading_order':len(soup.find_all('table')),'equation_source_order':equation_order,'sae_story_rewritten':True,'cross_object_comparison_added':has_cross_object_comparison})
    (ROOT/'revision/display_manifest.json').write_text(json.dumps({'files':manifests,'formulas':[x for x in EQUATIONS if x is not None],'table6':repaired_tables[0],'changes':['Student closed-loop mechanism and source of time-domain curves; figure captions and references renumbered','HTML math typography','local fonts','Figure 1 layout and tiny/small labels','Table 6 generator text rendered as table; cell values unchanged','Sampled object model defined first; fixed reference-model equation, values and added explanatory section removed from the paper','Section 2 parameter table: eight sampled and two derived parameters; table captions and references numbered in reading order'],'historical_metrics':'Reported values retain their recorded target signals; they have not been relabeled as newly measured raw-command tracking results.','original_root_untouched':str(ORIGINAL.parent)},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifests,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
