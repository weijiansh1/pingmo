# 当前 GJB 对齐状态

当前第一阶段设计见 [`第一阶段_SAC控制设计.md`](第一阶段_SAC控制设计.md)。该文档取代历史
G0-G4 路线以及以固定 reference tracking 为主目标的环境设计。

## 当前已冻结

- 控制对象为单通道 `F_eq -> p`，SAC 输出同一输入点上的等效修正力 `delta_F`。
- 对象与策略均为 `0.001 s`；每个对象样本决策一次，不再使用 `0.02 s` 动作保持。
- `tau_p` 是抽样后固定的纯运输延迟，对象内部用分数 FIFO 实现；最小抽样值为 `0.001 s`。
- Actor 明确读取当前滚转、振荡、释放恢复和延迟响应代价，再输出下一步力修正。
- Actor/特权 Critic 均不读取物理延迟 FIFO；reward 不人工搬移到旧动作。
- 默认取消无证据的 `0.08 s` 修正通道滞后；幅值权限为 `+/-6.6 N`，限速为 `26.4 N/s`。
- 环境不再以固定 `p_ref` 为主要目标，训练使用响应品质区间和动作代价。

## GJB 边界

权威扫描件为仓库根目录 `GJB_2874-1997_电传操纵系统飞机的飞行品质.pdf`。全书项目记忆见
[`references/gjb_2874_1997_project_memory.md`](references/gjb_2874_1997_project_memory.md)，IV-A
参数边界审计见 [`references/gjb_2874_1997_iv_a_parameter_audit.md`](references/gjb_2874_1997_iv_a_parameter_audit.md)。

- A18 的 `T_R`、A19 的螺旋倍幅时间、A35 的荷兰滚参数和 A31 的 `S1s` 用于机库分层及响应解释。
- 当前 SISO 模型没有 `beta` 和 `r`，因此振荡与释放恢复结果必须标为代理。
- 表 A21 的 `tau_e` 是完整闭环响应的 LOES 拟合参数，不等于抽样的 `tau_p`。仓库尚未实现正式
  `tau_e` 联合拟合，禁止直接用 `tau_p` 给最终系统延迟分级。

## 当前实验状态

- 代码接口、多指令、response-cost、MLP/MoE 中等规模网络和统一评测正在 CPU 测试。
- MLP-SAC 与 4-expert MoE-SAC 均计划 5 个种子；正式长 GPU 训练尚未启动。
- 历史 `checkpoints/exploratory_sac_*`、reference/oracle 报告和 G0-G4 报告只作回归材料，不能作为
  当前设计结论。
