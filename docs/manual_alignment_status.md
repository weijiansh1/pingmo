# GJB 手册 v1 对齐状态

权威规格：仓库外层的 `GJB_MoE_SAC_滚转品质实验手册_v1.md`。

权威原始扫描件：仓库根目录 `GJB_2874-1997_电传操纵系统飞机的飞行品质.pdf`。本扫描件 PDF 页码比印刷页码提前 4 页：滚转轴 A5.4 起于 PDF 214／印刷 210；图 A116 位于 PDF 240／印刷 236；PDF 241／印刷 237 给出 `ψ_β` 的定义，PDF 243／印刷 239 为图 A119。全书阅读记录、标准结构和项目术语边界见 [`references/gjb_2874_1997_project_memory.md`](references/gjb_2874_1997_project_memory.md)。

## 延迟术语纠偏（2026-08-27）

- GJB 表 A21 的滚转轴 `τ_e` 是从完整的滚转操纵力到滚转速率响应进行低阶等效系统拟合后得到的总等效时间延迟。它可能吸收前置滤波器、SAS、舵机、传感器/结构滤波、数字采样保持和计算等相位损失，不由某一个采样周期或硬件常数直接决定。
- 当前代码的 `tau_p` 是 `FractionalDelay` 实现的纯运输延迟 `exp(-tau_p*s)`，不是已经辨识的 GJB `τ_e`。当前 3,000 机库的 L1-L3 范围约为 `0.010031-0.199954 s`，OOD 最大为 `0.498005 s`，但均不能直接用表 A21 打等级。
- `0.08 s` 是只作用于 RL/受限 Oracle 修正通道的一阶命令滞后时间常数，不是纯延迟、不是由 `0.02 s` 控制周期推导的值，也没有被 GJB 指定。其物理接入点和证据来源尚待确定。
- `0.005 s` 是 plant 离散推进步长；`0.02 s` 是策略更新与动作保持周期。50 Hz 零阶保持在低频下约贡献 `0.01 s` 相位延迟，但最终是否计入以及计入多少，必须以声明输入点后的端到端 LOES 拟合为准。
- 新机库已保存 `delay_definition=pure_transport_delay_before_p_channel`；仓库仍无 `0.1-10 rad/s` 的 `τ_e` 联合拟合器。完成拟合前，禁止把 `tau_p`、`0.08 s` 或它们的算术和报告成 GJB 等效时间延迟。

## 禁止作为正式结论的历史产物

- `data/aircraft/generated/p_channel_library_20260827*`：以直接 `L_Fa` 采样构造，未按 IV-A 的 `S_1s` 响应标定；仅保留作原型回归数据。
- `checkpoints/exploratory_sac_*`：包含一次 ID-test 训练泄漏，永久废弃。
- `checkpoints/privileged_sac_train_core_0000` 与对应图：证明双状态 SAC 管线可运行，但使用旧 plant bank、旧 reference 和无量纲动作，不能用于 GJB 或 Teacher 结论。

## 已完成的本地对齐

- IV-A Profile 已冻结：`F_pilot,scale=22 N`，动作权限候选为 0.1 / 0.2 / 0.3 / 0.5，归一化限速为 `4 / s`。
- 新库 `data/aircraft/generated/p_channel_library_iv_a_manual_v1/` 已由 16,384 个 Sobol 候选生成，按 A18/A19/A35/A31 最差静态条款固定选出 L1/L2/L3 各 900 和 OOD 300；`plants.jsonl` SHA-256 为 `479c16d7b5ddb363c0e2d069a7a1e2d510f0a77cf8335eb86da6bb359ca4b118`。
- diagnostic reference 保留 raw 的 `S_1s` 和代码级纯运输延迟 `tau_p`，仅消除极零不匹配；constrained oracle 与 RL 共用 50 Hz、力幅与速率限幅。这里的 `tau_p` 尚未验证为 GJB `τ_e`。
- `src.benchmark.time_domain.evaluate_roll_response` 中的绝对峰 `rho_osc` 只保留作造库诊断；它不是 A120/A116 指标，正式审计只使用严格的 `P1 -> P2 -> P3` 提取路径。
- 50 Hz 环境使用 142 维 `[当前量（含命令/实际执行量）, 32×历史, theta]` 输入；命令端受速率限制，RL 等效修正量经人为设定的 `0.08 s` 一阶命令滞后后进入 P-channel。该状态不是已辨识舵机。环境记录 `F_pilot`、命令/实际 `ΔF`、`F_eq`、动作速率、饱和比例和 cancellation 指标；cancellation 仅作诊断，未写入 reward。
- 历史 G2 图和 JSON 属于上一版机库；`train_core-0000` 已随新机库改变，旧数字不得作为当前 G2 证据。

## 当前 Gate

- **G0 对新机库待重跑**：A18/A19/A21/A30/A31/A35 边界和 A116/A120 实现仍保留，但历史 `results/手册G0_A116_IV-A审计.json` 对应旧机库，不能与新哈希混用。新机库目前只完成静态等级和数值有限性检查；22 N 仍只是研究命令尺度，不是 A116 审计上限。
- **G1 未通过**：新机库已有 `delay_definition`，但尚缺完整的 ±step、pulse、doublet、roll reversal、3-2-1-1、随机/扫频 raw benchmark，以及 `T_R^eq`、端到端 `τ_e`、LOES residual 与响应级 GJB 标签。
- **G2 未通过**：reference 的同 `S_1s` / 同延迟与 oracle 的相同约束已验证；仍需在全部权限（0.1/0.2/0.3/0.5）和规定激励上审查“受限 oracle 是否可实现”。
- **G3 未开始**：禁止启动正式单机 SAC；更禁止全局 SAC、MoE、ID/OOD/Stress 的正式训练或比较。
