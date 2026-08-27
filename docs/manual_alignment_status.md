# GJB 手册 v1 对齐状态

权威规格：仓库外层的 `GJB_MoE_SAC_滚转品质实验手册_v1.md`。

权威原始扫描件：仓库根目录 `GJB_2874-1997_电传操纵系统飞机的飞行品质.pdf`。本扫描件 PDF 页码比印刷页码提前 4 页：滚转轴 A5.4 起于 PDF 214／印刷 210；图 A116 位于 PDF 240／印刷 236；PDF 241／印刷 237 给出 `ψ_β` 的定义，PDF 243／印刷 239 为图 A119。全书阅读记录、标准结构和项目术语边界见 [`references/gjb_2874_1997_project_memory.md`](references/gjb_2874_1997_project_memory.md)。

## 延迟术语纠偏（2026-08-27）

- GJB 表 A21 的滚转轴 `τ_e` 是从完整的滚转操纵力到滚转速率响应进行低阶等效系统拟合后得到的总等效时间延迟。它可能吸收前置滤波器、SAS、舵机、传感器/结构滤波、数字采样保持和计算等相位损失，不由某一个采样周期或硬件常数直接决定。
- 当前代码的 `tau_p` 是 `FractionalDelay` 实现的纯运输延迟 `exp(-tau_p*s)`，不是已经辨识的 GJB `τ_e`。当前 3,000 机库的 `tau_p` 范围约为 `0.000116-0.499665 s`，可作为延迟压力分布，但不能直接用表 A21 打等级。
- `0.08 s` 是只作用于 RL/受限 Oracle 修正通道的一阶命令滞后时间常数，不是纯延迟、不是由 `0.02 s` 控制周期推导的值，也没有被 GJB 指定。其物理接入点和证据来源尚待确定。
- `0.005 s` 是 plant 离散推进步长；`0.02 s` 是策略更新与动作保持周期。50 Hz 零阶保持在低频下约贡献 `0.01 s` 相位延迟，但最终是否计入以及计入多少，必须以声明输入点后的端到端 LOES 拟合为准。
- 当前机库记录没有落实采样配置要求的 `delay_definition`，仓库也尚无 `0.1-10 rad/s` 的 `τ_e` 联合拟合器。完成这两项之前，禁止把 `tau_p`、`0.08 s` 或它们的算术和报告成 GJB 等效时间延迟。

## 禁止作为正式结论的历史产物

- `data/aircraft/generated/p_channel_library_20260827*`：以直接 `L_Fa` 采样构造，未按 IV-A 的 `S_1s` 响应标定；仅保留作原型回归数据。
- `checkpoints/exploratory_sac_*`：包含一次 ID-test 训练泄漏，永久废弃。
- `checkpoints/privileged_sac_train_core_0000` 与对应图：证明双状态 SAC 管线可运行，但使用旧 plant bank、旧 reference 和无量纲动作，不能用于 GJB 或 Teacher 结论。

## 已完成的本地对齐

- IV-A Profile 已冻结：`F_pilot,scale=22 N`，动作权限候选为 0.1 / 0.2 / 0.3 / 0.5，归一化限速为 `4 / s`。
- 新库 `data/aircraft/generated/p_channel_library_iv_a_manual_v1/` 已由 16,384 个 Sobol 候选生成，固定选出 3,000 个 IV-A plant；所有候选的数值 `S_1s` 标定相对误差小于 `6e-13`。
- diagnostic reference 保留 raw 的 `S_1s` 和代码级纯运输延迟 `tau_p`，仅消除极零不匹配；constrained oracle 与 RL 共用 50 Hz、力幅与速率限幅。这里的 `tau_p` 尚未验证为 GJB `τ_e`。
- `src.benchmark.time_domain.evaluate_roll_response` 中的绝对峰 `rho_osc` 只保留作造库诊断；它不是 A120/A116 指标，正式审计只使用严格的 `P1 -> P2 -> P3` 提取路径。
- 50 Hz 环境使用 142 维 `[当前量（含命令/实际执行量）, 32×历史, theta]` 输入；命令端受速率限制，RL 等效修正量经人为设定的 `0.08 s` 一阶命令滞后后进入 P-channel。该状态不是已辨识舵机。环境记录 `F_pilot`、命令/实际 `ΔF`、`F_eq`、动作速率、饱和比例和 cancellation 指标；cancellation 仅作诊断，未写入 reward。
- 已生成 G2 基线：`img/手册G2_原始参考Oracle受限响应.png` 与 `results/手册G2_参考与Oracle训练前检查.json`。在 `train_core-0000`、0.3×22 N 下，raw/ref RMSE 为 0.3144，受限 oracle/ref RMSE 为 0.1254；但受限 oracle 饱和比例为 86.4%，因此只能证明可改善，不能声明已完全可实现。

## 当前 Gate

- **G0 已执行，Gate 结论待补充审查**：A18/A19/A21/A31 已转录；A116 的当前 IV-A 所需 A/C、Level 1/2 边界已从 PDF 240／印刷 236 的网格拐点数字化到 `data/gjb_a116_boundary.csv`，并保存了来源。`ψ_p` 已由当前 P-channel 的荷兰滚复极点留数计算，作为图 A119 允许的 `ψ_β` 替代量。A120（PDF 244／印刷 240）已按第一滚转速率峰 `P1`、其后第一谷 `P2`、第二峰 `P3` 实现；响应先去除螺旋模态。每架机均以偏航自由、保持阶跃输入审计，并按其自身需要的输入幅值在 `1.7*T_d` 达到 60 度滚转角；**22 N 只属于训练环境的命令归一化尺度，不是 A116 审计上限。**完整结果在 `results/手册G0_A116_IV-A审计.json`：3,000 架中 2,541 架可严格评估（Level 1：34，Level 2：27，超过 Level 2：2,480）；459 架不赋等级（402 架在当前 20 秒窗口内缺第二峰，54 架单位响应非正，3 架 A120 分母非正）。G0 的审计实现和数据产物已完成；剩余事项是审查 402 架是否需要随荷兰滚周期自适应延长窗口，或其响应确实没有可定义的 A120 峰—谷—峰结构。这个问题不限制控制器的本地/GPU探索训练，但在声称全包线 GJB 结论前必须澄清。
- **G1 未通过**：尚缺完整的 ±step、pulse、doublet、roll reversal、3-2-1-1、随机/扫频 raw benchmark，以及 `T_R^eq`、端到端 `τ_e`、LOES residual 与自动 GJB 标签；还需为每架飞机补齐 `delay_definition`。
- **G2 未通过**：reference 的同 `S_1s` / 同延迟与 oracle 的相同约束已验证；仍需在全部权限（0.1/0.2/0.3/0.5）和规定激励上审查“受限 oracle 是否可实现”。
- **G3 未开始**：禁止启动正式单机 SAC；更禁止全局 SAC、MoE、ID/OOD/Stress 的正式训练或比较。
