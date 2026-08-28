# 当前 GJB 对齐状态

当前第一阶段设计见 [`第一阶段_SAC控制设计.md`](第一阶段_SAC控制设计.md)。该路线是“固定飞机
specialist SAC Teacher -> 条件 Dense Student”，取代历史全机群 Global SAC、G0-G4 和首阶段
MoE 路线。

## 当前代码契约

- 控制对象为单通道 `F_as -> p`；命令为滚转角速度 `p_c`。
- 二阶研究参考模型产生 `p_ref`，默认 `omega_n=2 rad/s, zeta=0.7`。
- 每个 Teacher 只负责一架固定飞机，Actor 不读取 `theta`。
- Teacher 动作是完整 `F_as`，不是修正量 `delta_F`。
- plant 和策略均为 `0.001 s`；默认 `+/-22 N`、`88 N/s`，无额外 0.08 s 一阶滞后。
- Reward 仅由参考跟踪、力能量和逐毫秒动作变化三项组成。
- Student 才读取八维归一化 `theta`，并通过 specialist 动作监督蒸馏训练。
- Teacher 数量按 1、10、50/100 逐级扩展；MoE 暂不进入默认路线。

## GJB 边界

权威扫描件为仓库根目录 `GJB_2874-1997_电传操纵系统飞机的飞行品质.pdf`。全书项目记忆见
[`references/gjb_2874_1997_project_memory.md`](references/gjb_2874_1997_project_memory.md)，IV-A
参数边界审计见 [`references/gjb_2874_1997_iv_a_parameter_audit.md`](references/gjb_2874_1997_iv_a_parameter_audit.md)。

- GJB 没有唯一规定当前二阶 `p_ref` 的 `omega_n=2, zeta=0.7`；这是待验证的研究选择。
- 当前 SISO 模型没有独立 `beta` 和 `r`，不能正式分离荷兰滚、螺旋或完整滚偏航耦合品质。
- 表 A21 的 `tau_e` 是总闭环 LOES 拟合结果，不等于对象抽样的纯运输延迟 `tau_p`。
- 当前传递函数在原点有零点，恒定滚转率参考的有界力可达性必须由响应与权限审计确认。
- 本阶段输出是 GJB 启发式控制研究，不是型号合格结论。

## 当前实验状态

历史 68M 参数全机群 MLP-SAC 在训练飞机上也出现 45/45 harm，只保留为失败基线。新 specialist
与蒸馏代码已建立本地端到端烟测；尚未据此宣称单机 Teacher 收敛、跨飞机 Student 泛化或正式
GPU 长训练成功。
