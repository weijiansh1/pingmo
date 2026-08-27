p/F_as 参数物理范围资料包
版本：2026-08-26

先打开：p_Fas参数范围与采样建议.xlsx

结论
1. 目前公开的一手资料足以构造“分机类、分飞行状态的研究包络”，但不足以估计8个参数的真实联合概率分布。
2. 不能把8个参数各自取上下限后独立均匀采样。先选aircraft_class/flight_phase/condition，再联合采样。
3. L_Fa不要独立采样。建议先采准稳态滚转灵敏度K_p/F，再用
   L_Fa = K_pF / (T_R * R_omega^2)
   派生，其中R_omega=omega_phi/omega_d。
4. 螺旋模态主存lambda_s：稳定<0，中性=0，发散>0。发散时T2=ln(2)/lambda_s。
5. omega_phi通过R_omega派生。zeta_phi建议直接条件采样；R_zeta只统计，不作主采样轴。
6. GJB/MIL数值是品质边界，不是实际飞机采样分布。

最简研究范围（未锁定机类前，只能作为临时覆盖）
- T_R：有人固定翼核心0.15-1.8 s；小型UAS 0.06-0.25 s；OOD到3 s；3-10 s只作极端品质压力。
- tau_p：0.01-0.10 s核心，0.10-0.20 s高延迟，0.20-0.25 s极限。
- zeta_d：0.08-0.50核心，0.02-0.80压力。
- omega_d：有人0.5-3.5 rad/s；小型UAS 1.8-6.0 rad/s；专门UAS压力可到8。
- lambda_s：稳定[-0.10,-0.003] 1/s；中性单独分类；发散按T2=4-100 s采样。
- R_omega：0.80-1.15核心，0.65-1.35扩展。
- zeta_phi：0.20-0.35核心，0.10-0.70压力，证据置信度低于前五项。
- L_Fa：同口径力指令战斗机用K_p/F=5-25 deg/s/lbf派生；约0.471 SI以上只作外推。

辅助量
- F_as,max：必须按centerstick/sidestick/wheel分桶。中杆Level 1参考约89-111 N；不能统一套用。
- Fdot_as,max：没有找到跨机型通用公开标准。本包500-2000 N/s仅作力感系统敏感性，默认1000 N/s。
- p_c,max：按任务与可达性推导，p_c,max <= eta*K_pF*Fmax，eta初始取0.6-0.8；战斗机约300 deg/s只是型号例证。

文件
- p_Fas参数范围与采样建议.xlsx：主工作簿。
- parameter_ranges.csv：8参数建议的机器可读版本。
- scenario_ranges.csv：按机类/任务的范围。
- evidence_points.csv：可计算的宽表证据点。
- evidence_facts.csv：来源中的单项数值事实。
- standard_boundaries.csv：GJB/MIL品质边界转录。
- sampling_rules.csv：联合采样顺序和准入闸门。
- sampling_config.json：供后续生成器读取的临时配置。
- evidence/：GJB关键表截图及说明。
- SHA256SUMS.txt：包内文件校验值。

使用限制
- 本包不是适航符合性材料，也不是对某一型飞机的鉴定。
- NASA的SST和部分UAS数据是模拟/模型构型，已在表中与实机飞行辨识分开标记。
- 等效时延不必等于纯运输时延；使用e^{-tau*s}前必须记录定义。
- UAS通常没有物理F_as输入，因此只能贡献模态包络，不能直接贡献L_Fa或F_as,max。
