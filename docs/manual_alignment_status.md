# GJB 手册 v1 对齐状态

权威规格：仓库外层的 `GJB_MoE_SAC_滚转品质实验手册_v1.md`。

权威原始扫描件：`C:\Users\24307\Desktop\品模\GJB_2874-1997_电传操纵系统飞机的飞行品质.pdf`。本扫描件 PDF 页码比印刷页码提前 4 页：滚转轴 A5.4 起于 PDF 214／印刷 210；图 A116 位于 PDF 240／印刷 236；PDF 241／印刷 237 给出 `ψ_β` 的定义，PDF 243／印刷 239 为图 A119。

## 禁止作为正式结论的历史产物

- `data/aircraft/generated/p_channel_library_20260827*`：以直接 `L_Fa` 采样构造，未按 IV-A 的 `S_1s` 响应标定；仅保留作原型回归数据。
- `checkpoints/exploratory_sac_*`：包含一次 ID-test 训练泄漏，永久废弃。
- `checkpoints/privileged_sac_train_core_0000` 与对应图：证明双状态 SAC 管线可运行，但使用旧 plant bank、旧 reference 和无量纲动作，不能用于 GJB 或 Teacher 结论。

## 已完成的本地对齐

- IV-A Profile 已冻结：`F_pilot,scale=22 N`，动作权限候选为 0.1 / 0.2 / 0.3 / 0.5，归一化限速为 `4 / s`。
- 新库 `data/aircraft/generated/p_channel_library_iv_a_manual_v1/` 已由 16,384 个 Sobol 候选生成，固定选出 3,000 个 IV-A plant；所有候选的数值 `S_1s` 标定相对误差小于 `6e-13`。
- diagnostic reference 保留 raw 的 `S_1s` 和物理延迟，仅消除极零不匹配；constrained oracle 与 RL 共用 50 Hz、力幅与速率限幅。
- `src.benchmark.time_domain.evaluate_roll_response` 中的绝对峰 `rho_osc` 只保留作造库诊断；它不是 A120/A116 指标，正式审计只使用严格的 `P1 -> P2 -> P3` 提取路径。
- 50 Hz 环境使用 142 维 `[当前量（含命令/实际执行量）, 32×历史, theta]` 输入；命令端受速率限制，实际执行量经 `0.08 s` 一阶滞后后进入飞机。环境记录 `F_pilot`、命令/实际 `ΔF`、`F_eq`、动作速率、饱和比例和 cancellation 指标；cancellation 仅作诊断，未写入 reward。
- 已生成 G2 基线：`img/手册G2_原始参考Oracle受限响应.png` 与 `results/手册G2_参考与Oracle训练前检查.json`。在 `train_core-0000`、0.3×22 N 下，raw/ref RMSE 为 0.3144，受限 oracle/ref RMSE 为 0.1254；但受限 oracle 饱和比例为 86.4%，因此只能证明可改善，不能声明已完全可实现。

## 当前 Gate

- **G0 已执行，Gate 结论待补充审查**：A18/A19/A21/A31 已转录；A116 的当前 IV-A 所需 A/C、Level 1/2 边界已从 PDF 240／印刷 236 的网格拐点数字化到 `data/gjb_a116_boundary.csv`，并保存了来源。`ψ_p` 已由当前 P-channel 的荷兰滚复极点留数计算，作为图 A119 允许的 `ψ_β` 替代量。A120（PDF 244／印刷 240）已按第一滚转速率峰 `P1`、其后第一谷 `P2`、第二峰 `P3` 实现；响应先去除螺旋模态。每架机均以偏航自由、保持阶跃输入审计，并按其自身需要的输入幅值在 `1.7*T_d` 达到 60 度滚转角；**22 N 只属于训练环境的命令归一化尺度，不是 A116 审计上限。**完整结果在 `results/手册G0_A116_IV-A审计.json`：3,000 架中 2,541 架可严格评估（Level 1：34，Level 2：27，超过 Level 2：2,480）；459 架不赋等级（402 架在当前 20 秒窗口内缺第二峰，54 架单位响应非正，3 架 A120 分母非正）。G0 的审计实现和数据产物已完成；剩余事项是审查 402 架是否需要随荷兰滚周期自适应延长窗口，或其响应确实没有可定义的 A120 峰—谷—峰结构。这个问题不限制控制器的本地/GPU探索训练，但在声称全包线 GJB 结论前必须澄清。
- **G1 未通过**：尚缺完整的 ±step、pulse、doublet、roll reversal、3-2-1-1、随机/扫频 raw benchmark，以及 `T_R^eq`、LOES residual 与自动 GJB 标签。
- **G2 未通过**：reference 的同 `S_1s` / 同延迟与 oracle 的相同约束已验证；仍需在全部权限（0.1/0.2/0.3/0.5）和规定激励上审查“受限 oracle 是否可实现”。
- **G3 未开始**：禁止启动正式单机 SAC；更禁止全局 SAC、MoE、ID/OOD/Stress 的正式训练或比较。
