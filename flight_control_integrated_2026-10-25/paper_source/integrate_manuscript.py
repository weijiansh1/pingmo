"""Integrate the full existing study and its checks without replacing its mainline."""
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'revision/integrated_base_v2.md').read_text()
supp=(ROOT/'revision/supplemental_route_source.md').read_text()

def section(text,start,end):
    return text.split(start,1)[1].split(end,1)[0].strip()

def supplemental(text):
    text=text.replace('(paper/','(supplement/paper/').replace('(evidence/','(supplement/evidence/')
    text=text.replace('(time_domain_gallery.html)','(supplement/time_domain_gallery.html)')
    for n in range(10,1,-1):
        text=re.sub(r'图\s*'+str(n)+r'(?!\d)',f'图 S{n-1}',text)
    text=text.replace('表 2','表 S1').replace('表 3','表 S2')
    text=text.replace('附录 C.1','补充协议的完整配对记录').replace('附录 C.4','补充协议的可达性诊断记录').replace('附录 C.3','补充协议的完整计数')
    return text

base=base.replace('# 基于飞行品质模型的飞控大模型研究', '# 基于飞行品质模型的统一飞控控制器研究',1)
base=re.sub(r'^> 核验修订版[^\n]*', '<p class="meta">整体研究整合版 · 滚转通道验证 · 2026-10-25</p>',base,flags=re.M)
abstract='''面向动态特性不同的对象，统一飞控控制器需要兼顾指令跟踪、振荡抑制、部署信息和模型规模。本文研究“品质相关对象族—特权强化学习教师—多轮蒸馏学生—目标适配”的构建路线，以滚转通道验证其可行性。研究包括 5,000 个参数化对象、49 教师的源域主线、tiny（3,856 参数）和 small（75,040 参数）学生、扩展对象及指令幅值测试，以及 DHC-6 仿真辨识对象上的适配。已有报告中，tiny/small 在 44 个蒸馏对象上分别通过 37/38 个，在四个留出对象上均通过三个；学生相对教师仍有精度损失，大幅值和分布外对象上出现明显退化。

围绕统一控制器依靠有限历史适应不同动力学的问题，进一步研究延迟信息可解码性、SAE 表示及闭环等效时延，并通过原 tiny 的复现、公共参考配对、统一训练范围和多策略种子实验检查结论。原 tiny 在 510 对象上的 4,080 个对象—指令判定得到复现；在另一组 120 对象的公共目标配对检查中，公共参考下通过率为 89.58%。补充的 12 源对象、三个种子协议完成了教师学习和六轮 DAgger，显示迭代训练改善验证误差，但能力保留、未见对象恢复和持续控制仍不稳定。上述实验共同支持统一小型控制器的构建与进一步适配，同时明确其可靠性边界。结果按实际模型、参考接口和评价窗口报告；已有主线中未随包提供完整原始记录的数值标为报告值，具体复核范围见实验设置。

**关键词：** 飞控控制器；飞行品质模型；特权强化学习；多教师蒸馏；DAgger；迁移学习；延迟表征
'''
base=base.split('## 摘要',1)[0]+'## 摘要\n\n'+abstract+'\n## 1 引言'+base.split('## 1 引言',1)[1]
intro='''飞控控制器需要完成指令跟踪，并在初始偏差和外部扰动后抑制不良振荡。对象的响应速度、阻尼、操纵效能和时延各不相同，面向对象族的共享策略需要利用输入输出历史调整动作。本研究中的品模用于组织有控制意义的动态变化，最终产物是统一飞控控制器；滚转是当前验证入口。

研究主线从品质相关参数采样开始，为各对象学习专用策略，再将多对象控制行为压缩到普通观测下的学生，最后检查新对象泛化与目标域适配。已有 49 教师主线及 tiny/small 学生提供了端到端探索，扩展对象、幅值扫描和 DHC-6 适配展示了能力与边界。延迟表征和等效时延分析继续回答学生从历史中获得了什么信息、这些表示是否改善闭环，以及响应是否满足所采用的研究性评价条件。

围绕这些已有结果，本文补充参考接口、时间因果性、样本隔离与训练重复的检查。各实验共同检验同一条控制器构建路线：原始学生和其派生表示实验保持各自的权重身份，补充协议重新训练的学生用于检查实现和控制表现。模型规模相同不代表权重相同；输入条件、指令和评价窗口变化时，结果不能直接作为同一模型的前后成绩。第 4 节给出对应关系，后续证据放在各自回答的控制问题下。
'''
base=base.split('## 1 引言',1)[0]+'## 1 引言\n\n'+intro+'\n### 1.1 相关工作'+base.split('### 1.1 相关工作',1)[1]

# Preserve original method/results; remove stale statements about later checks.
base=base.replace('修订后的 PPO 尚未重跑这一主线预算。','后续补充协议已在 12 个源对象上按同一每教师预算完成三个种子的训练，但没有重训原主线全部 49 个对象。')
base=base.replace('第 3—5 节主线的归档报告值','第 3—5 节的已有主线结果')
base=base.replace('### 5.2 第二步：逐对象教师训练（归档报告值）','### 5.2 第二步：逐对象教师训练')
base=base.replace('### 5.3 第三步：统一学生蒸馏（归档报告值）','### 5.3 第三步：统一学生蒸馏')
base=base.replace('### 5.4 第四步：DHC-6 仿真目标上的小样本微调（归档报告值）','### 5.4 第四步：DHC-6 仿真目标上的小样本微调')
base=base.replace('49 个对象分别训练后，46 个通过','原主线报告 49 个对象分别训练后，46 个通过',1)
base=base.replace('图 7 汇总 21 个具备逐轮固定验证记录的通过教师。','图 7 汇总另一批旧教师库中 21 个具备逐轮固定验证记录的通过教师。该库有 68 个条目、32 个准入，不能将本图直接视为 49 教师主线的逐轮核验。')
base=base.replace('*图 7　21 个通过教师的固定验证 RMSE、操纵总变差及 checkpoint 选择过程。*','*图 7　旧教师库中 21 个通过教师的固定验证 RMSE、操纵总变差及 checkpoint 选择过程；该旧库与 49 教师主线分别记录。*')
base=base.replace('*图 8　教师训练完成后的源域闭环效果：不同品质等级的阶跃与多正弦响应。*','*图 8　旧教师库中代表性教师的源域闭环效果：不同部分品质标签的阶跃与多正弦响应。保留其原始轨迹与来源，不将其作为 49 教师主线的全部通过率证据。*')

protocol_note='''### 4.1 同一研究中各组结果的对应关系

| 研究内容 | 实际策略与对象 | 这组结果回答的问题 |
|---|---|---|
| 源域构建主线 | 原稿报告的 49 教师；44 个蒸馏对象、4 个留出对象；tiny/small | 多对象专用策略能否形成统一控制器，压缩后损失多少能力 |
| 扩展对象与幅值测试 | 原始 tiny；120 对象抽样，以及 300 ID＋500 OOD 扫描 | 已有统一策略随对象和指令幅值如何变化 |
| 原始策略复现与参考检查 | 随包 tiny，权重 SHA-256 前缀 ff5f5b5b1a63；510 对象复现、120 对象配对 | 原有策略是否可复现，参考输入条件是否改变闭环表现 |
| 表征和等效时延 | 原始 tiny 的轨迹、raw_ft/SAE 等派生策略及受控重训 | 历史中的信息能否读出，表征改动是否改善实际控制 |
| 训练与恢复补充协议 | 12 源对象×3 种子；每个新学生对应 12 个教师；ID/OOD 各 24 个 | 完整新日志下的训练重复、蒸馏保留、未见对象与长时恢复 |
| 目标适配 | 原 small 的 40 步适配；补充 tiny 的等预算 64 步对照 | 已有控制经验能否用于新目标，初始化是否有帮助 |

原主线 49/46 教师、tiny/small 配对统计、800 对象扫描与 small 40 步适配保留原报告值及图。其完整逐项原始记录未随交付包提供，尚未完成独立复核；这项材料限制不等于判定原结果错误。原 tiny 权重及随包 SAE/频响记录可以复核，具体范围和结果在对应小节展开。补充协议的教师、学生权重和全部逐案例轨迹齐全，属于同一研究的补充验证，不能替换原学生身份或追认旧成绩。

补充协议采用公共参考、同一时刻采样，训练/验证/测试分别为 32/16/18 条独立波形；指令指标从 1 s 指令起始计算，另测 30 s 恢复。原稿代表性曲线包含 2 s 后的 RMSE，部分原训练监控指令也有重复。因此不同表格的百分比必须结合各自分母和窗口阅读，不能直接相减解释为方法升级或退化。补充协议的 S1–S3 是三个重新训练的学生，不是原始 tiny 的三种绘图方式。

学生以源对象验证集选模，测试对象不参与选择。其 BC 训练 30 个 epoch，随后六轮实际闭环采集、标注和聚合，每轮训练 8 个 epoch，三个种子各累计 1,075,200 个样本。教师/学生网络和 PPO、GAE 定义沿用第 3 节；补充协议的精确配置见[冻结协议](supplement/evidence/controller_route_protocol.json)。

'''
base=base.replace('## 5 结果\n',protocol_note+'## 5 结果\n',1)

s51=section(supp,'### 5.1','### 5.2')
s51=s51.split('\n',1)[1].strip() # drop remainder of heading
teacher_part,distill_part=s51.split('以这些教师的轨迹',1)
teacher_part=supplemental(teacher_part)
teacher_part='#### 5.2.1 统一协议下的重复训练\n\n以下补充实验采用 12 个源对象、三个种子，保留所有对象和训练结果；其教师没有替换前述原主线教师。\n\n'+teacher_part+'\n\n'
base=base.replace('### 5.3 第三步',teacher_part+'### 5.3 第三步',1)

student_part=section(supp,'### 5.2','### 5.3').split('\n',1)[1].strip()
recovery_part=section(supp,'### 5.3','### 5.4').split('\n',1)[1].strip()
new_student='#### 5.3.5 多轮蒸馏和能力保留的补充验证\n\n下述 S1–S3 来自 12 对象补充协议，分别重新训练，不是原始 tiny。\n\n'+supplemental('以这些教师的轨迹'+distill_part)+'\n\n'+supplemental(student_part)+'\n\n#### 5.3.6 振荡恢复与持续控制\n\n'+supplemental(recovery_part)+'\n\n'
base=base.replace('### 5.4 第四步',new_student+'### 5.4 第四步',1)

target_part=section(supp,'### 5.4','## 6 讨论').split('\n',1)[1].strip()
new_target='#### 5.4.1 相同目标预算下的适配对照\n\n补充实验使用前述三次新训练的 3,856 参数学生，另设冻结和同预算从零训练对照；它检验初始化的作用，与原 small 的 40 步适配共同构成目标适配证据。两组网络规模、辨识记录、更新次数和测试指令不同，数值分别报告。\n\n'+supplemental(target_part)+'\n\n'
base=base.replace('## 6 讨论\n',new_target+'## 6 讨论\n',1)
base=base.replace('相同目标数据预算下的随机初始化、冻结策略和常规控制器对照仍需补齐。','补充 tiny 协议已提供同目标预算的随机初始化与冻结策略对照；原 small 配方对应的独立对照仍待补齐，二者不互相替代。')
base=base.replace('### 6.2 修订后新增对照','### 6.2 输入条件与表示的受控比较')
base=base.replace('没有重训原 49 个教师，也没有建立修复前后控制性能对照。','后续已完成第 5 节的 12 对象补充协议，但没有重训原 49 个教师，也没有建立原 49 对象上仅改变实现修复的性能对照。')

conclusion='''本文围绕统一飞控控制器形成了相互关联的研究：以品质相关参数组织对象族，通过专用强化学习教师和多轮蒸馏获得小型学生，再检验对象泛化、历史信息表征、闭环等效时延与目标适配。已有 49 教师主线、tiny/small 学生及扩展测试保留了端到端探索的证据；原 tiny 的轨迹复现、参考配对、表征对照和三种子补充协议进一步限定了结论的适用条件。

现有结果显示，统一学生能够承接多对象控制行为，预训练可成为目标适配的起点；同时，原主线和补充协议都存在蒸馏精度损失，较大指令、分布外对象和长时恢复仍可能失败。延迟可解码性不自动转化为闭环收益，SAE 辅助分支尚未显示跨种子一致的增益。原始 tiny 与补充学生是不同权重和训练协议，后者的差例不能用于直接否定前者，前者的正结果也不能代替后者的恢复验证。

下一步应围绕同一控制需求匹配训练目标、选模指标和完整闭环测试，优先处理可复现的控制退化，并用保持其他条件一致的对照分析信息、容量、采样和优化的作用。本文结论限定于滚转参数化对象及 DHC-6 辨识仿真，尚未建立可靠通用飞控或真实飞机上的品质保证。

'''
base=base.split('## 7 结论',1)[0]+'## 7 结论\n\n'+conclusion+'## 参考文献'+base.split('## 参考文献',1)[1]
base=re.sub(r'^### (5\.3\.[1-4])',r'#### \1',base,flags=re.M)

# Restore real mathematical typography for the original thirteen equations.
equations=[
r'P_{\mathrm{ref}}(s)=\frac{\omega_n^2}{s^2+2\zeta_n\omega_ns+\omega_n^2}e^{-\tau_{\mathrm{ref}}s}P_{\mathrm{cmd}}(s),\quad\omega_n=2\,\mathrm{rad/s},\quad\zeta_n=0.7.\tag{1}',
r'\begin{gathered}o_t=\operatorname{Norm}([p_{\mathrm{cmd}},p_{\mathrm{ref}},p,e,\int e\,dt,\dot p,\dot p_{\mathrm{ref}},F_{\mathrm{req}},F_{\mathrm{applied}},h_{t-31:t},a_{t-7:t-1}])\in\mathbb R^{176},\\o_t^{\mathrm T}=[o_t,z_t]\in\mathbb R^{188},\qquad o_t^{\mathrm S}=o_t.\end{gathered}\tag{2}',
r'G_\theta(s)=\frac{L_{F_a}s(s^2+2\zeta_\phi\omega_\phi s+\omega_\phi^2)}{(s-\lambda_s)(s+1/T_R)(s^2+2\zeta_d\omega_ds+\omega_d^2)}e^{-\tau_ps}.\tag{3}',
r'\begin{gathered}\operatorname{RMSE}_p=\sqrt{\frac1M\sum_{t=1}^M(p_{\mathrm{ref},t}-p_t)^2},\quad\operatorname{Peak}=\max_t|p_{\mathrm{ref},t}-p_t|,\\\operatorname{TV}_F=\frac1T\sum_{t=1}^M|F_{\mathrm{req},t}-F_{\mathrm{req},t-1}|.\end{gathered}\tag{4}',
r'\theta^{(i)}\sim q(\theta\mid c_i),\qquad g_i=\operatorname{Grade}_{\mathrm{partial\ IV-A}}(\theta^{(i)},c_i).\tag{5}',
r'F_{\mathrm{req},t+1}=\operatorname{clip}\!\left(F_{\mathrm{req},t}+(88\,\mathrm{N/s})a_t\Delta t,-22\,\mathrm N,22\,\mathrm N\right),\quad\Delta t=0.02\,\mathrm s.\tag{6}',
r'r_t=-50\left[\left(\frac{e_{t+1\mid t}}{p_{\mathrm{scale}}}\right)^2\Delta t+0.02\left(\frac{F_{\mathrm{applied},t+1}}{22\,\mathrm N}\right)^2\Delta t+0.02a_t^2\right].\tag{7}',
r'\begin{gathered}\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\\\widehat A_t=\delta_t+\gamma\lambda(1-d_t)\widehat A_{t+1},\quad\gamma=0.995,\quad\lambda=0.95.\end{gathered}\tag{8}',
r'\mathcal L_{\mathrm{clip}}(w)=-\mathbb E_t\!\left[\min\!\left(\rho_t\widehat A_t,\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\widehat A_t\right)\right],\quad\epsilon=0.2.\tag{9}',
r'\mathcal L_{\mathrm{total}}=\mathcal L_{\mathrm{clip}}+0.5\,\mathbb E[(V_v(s_t)-R_t)^2]-0.001\,\mathcal H(\pi_w),\quad R_t=\widehat A_t+V_{\mathrm{old}}(s_t).\tag{10}',
r'\begin{gathered}D_0=\{(o_t,\mu_{\mathrm T}(o_t,z_t))\},\\D_{k+1}=D_k\cup\{(o_t^{\mathrm S,k},\mu_{\mathrm T}(o_t^{\mathrm S,k},z_t))\}.\end{gathered}\tag{11}',
r'w_{\mathrm S}^*=\operatorname*{arg\,min}_w\frac1{|D_K|}\sum_{(o,u_{\mathrm T})\in D_K}\|\mu_{\mathrm S}(o;w)-u_{\mathrm T}\|^2.\tag{12}',
r'\begin{gathered}e_t(W)=p_{\mathrm{ref},t}-p_t(W;\widehat G_{\mathrm{ARX}}),\\\mathcal L_{\mathrm{DHC6}}(W)=\frac1{N_{t\ge2\,\mathrm s}}\sum_{t\ge2\,\mathrm s}e_t(W)^2,\\W_{k+1}=\operatorname{AdamUpdate}(W_k,\nabla_W\mathcal L_{\mathrm{DHC6}};\eta=3\times10^{-5}).\end{gathered}\tag{13}',
]
matches=list(re.finditer(r'\$\$(.*?)\$\$',base,re.S))
assert len(matches)==13,len(matches)
counter=iter(equations)
base=re.sub(r'\$\$(.*?)\$\$',lambda m:'$$ '+next(counter)+' $$',base,flags=re.S)

# Inline variable names are rendered only in prose, outside links/code/math.
tokens={'p_cmd':r'p_{\mathrm{cmd}}','p_ref,t':r'p_{\mathrm{ref},t}','p_ref':r'p_{\mathrm{ref}}',
'F_req':r'F_{\mathrm{req}}','F_applied':r'F_{\mathrm{applied}}','L_Fa':r'L_{F_a}',
'τ_ref':r'\tau_{\mathrm{ref}}','τp':r'\tau_p','τ_p':r'\tau_p','τe':r'\tau_e','τ_e':r'\tau_e',
'p_scale':r'p_{\mathrm{scale}}','e_{t+1|t}':r'e_{t+1\mid t}','ṗ_ref':r'\dot p_{\mathrm{ref}}','ṗ':r'\dot p',
'λ_s':r'\lambda_s','T_R':r'T_R','ζ_d':r'\zeta_d','ω_d':r'\omega_d','R_ω':r'R_\omega','R_ζ':r'R_\zeta',
'μ_T':r'\mu_{\mathrm T}','z_t':r'z_t','a_t':r'a_t','u_t':r'u_t','d_t':r'd_t','θ':r'\theta'}
protect=re.compile(r'\$\$.*?\$\$|\\\(.*?\\\)|`[^`]*`|!?\[[^\]]*\]\([^)]*\)|<[^>]*>',re.S)
token_pattern=re.compile('|'.join(re.escape(k) for k in sorted(tokens,key=len,reverse=True)))
segments=[];pos=0
for match in protect.finditer(base):
    segments.append(token_pattern.sub(lambda m:r'\('+tokens[m.group()]+r'\)',base[pos:match.start()]))
    segments.append(match.group());pos=match.end()
segments.append(token_pattern.sub(lambda m:r'\('+tokens[m.group()]+r'\)',base[pos:]))
base=''.join(segments)
(ROOT/'webpage/dhc6_report.md').write_text(base)
print('Integrated manuscript:',len(base),'characters')
