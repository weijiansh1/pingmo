# Flight RL Control

本仓库实现单通道、分飞机的控制学习路线。当前实验先为每个固定 P-channel 训练无 PID
示范的纯奖励 TD3 Teacher，再用 Student-driven / DAgger 把 32 个 Teacher 蒸馏为一个读取飞机
参数的 Dense Student。v4 在蒸馏中加入同轨迹动作增量匹配、困难样本加权和稳定性优先的闭环
选型。PID-guided TD3、theta-routed linear MoE、历史 Global SAC 和 G0-G4 仍保留作工程对照与
诊断，不与当前结果混写。

v5 先对冻结 v4 做了 requested-force 硬限速诊断，没有重新训练网络。训练飞机选出的 `11 N/s`
把 holdout 平均动作 TV 从 `249.48 N` 降到 `67.30 N`，但在 `boundary-1605` doublet 上把最大
峰值误差从 `4.76 deg/s` 推高到 `7.98 deg/s`，因此不能作为最终控制器。完整选择合同、时域
失败曲线和未见飞机结果见
[`docs/student_slew_limit_diagnostic_v5_20260830.md`](docs/student_slew_limit_diagnostic_v5_20260830.md)。

当前控制契约见 [`docs/第一阶段_SAC控制设计.md`](docs/第一阶段_SAC控制设计.md)。GJB 阅读边界见
[`docs/references/gjb_2874_1997_project_memory.md`](docs/references/gjb_2874_1997_project_memory.md)。
实验目录与 manifest 规则见 [`docs/experiment_artifacts.md`](docs/experiment_artifacts.md)。
v4 训练配置、时域曲线、未见飞机结果和失败门禁的完整快照见
[`docs/stability_aware_v4_results_20260830.md`](docs/stability_aware_v4_results_20260830.md)。v3 历史
快照保留在 [`docs/current_code_and_results_20260830.md`](docs/current_code_and_results_20260830.md)。

## PID-guided / MoE 工程流水线

```text
固定飞机 G(theta_i) -> 独立 RL Teacher pi_i(o)
                                  |
                                  v
                    Teacher-driven 初始数据
                                  |
                                  v
              theta-routed linear MoE pi(o, theta)
                                  |
                                  v
                    Student-driven / DAgger 聚合
```

- 命令 `p_c` 是滚转角速度指令。
- 二阶参考模型叠加该飞机同一个纯运输延迟 `tau_p` 后产生 `p_ref`。
- Teacher Actor 读取 4 个瞬时信号 `p_c, p_ref, p, error` 和 3 个控制器状态
  `integrated error, p_dot, previous requested F_as`；默认不读取原始序列窗口，也不读取 `theta`。
- 训练期 Critic 才读取对象连续状态、纯时延 FIFO、执行机构状态和固定维命令上下文；这些
  privileged 字段不进入 Teacher Actor、蒸馏 observation 或部署 Student。
- Teacher Actor 是约 995 万可训练参数的有界残差策略；控制先验以归一化线性层嵌入 Actor，在线
  TD3 更新学习小残差。部署 checkpoint 不实例化 PID 控制器对象，并完整记录该结构事实。
- Teacher 和 Student 都保证零状态零动作及正负滚转镜像。
- Teacher 动作是完整 `F_as`，不是 `F + delta_F`。
- Student 接收同一 7 维 observation 和归一化的八维 `theta`，输出同一定义的 `F_as`。路由器
  只读取静态 `theta`；线性专家只使用其中的 `error, integrated error, p_dot` 三个控制特征，
  不利用 `p_c, p_ref, p` 的相关性做离线拟合捷径，路由也不会随时域响应逐步跳变。
- `integrated error`、`p_dot`、`previous requested F_as` 是固定维控制状态；系统不输入
  `[p_{t-k:t}, u_{t-k:t}]` 一类原始时间序列窗口，也不使用 GRU/TCN。
- `basic/moe_td3.py` 与 `basic/results_tcn/` 是早期“无 Reference 残余阻尼”诊断实验，其
  `TemporalObservation` 会人工缓存 256 个过去采样。它们不被当前 Teacher/Student 流水线导入，
  也不是当前结果的历史信号来源。
- plant 以 `0.001 s` 更新，策略每 `0.020 s` 更新一次；每个策略动作内执行 20 个对象子步。
- 蒸馏首轮由 Teacher 驱动，后续轮次必须由当前 Student 驱动并由对应 Teacher 标注。
- 当前正式验证让 Teacher Bank 中的全部飞机参与训练，并为每架飞机保留独立的未见命令作为
  validation。整架飞机 holdout 作为更严格的泛化诊断单独报告；当前阶段不声称已经解决对完全
  未见 `theta` 的零样本控制。

## 执行顺序

完整的带门禁入口：

```bash
python scripts/35_run_teacher_student_pipeline.py \
  --teacher-algorithm pid-guided-td3 \
  --plant-id <plant-1> --plant-id <plant-2> \
  --teacher-network-width 704 --teacher-residual-blocks 10 \
  --student-architecture theta_routed_linear_moe \
  --distillation-split-strategy all_aircraft_command_holdout \
  --dagger-rounds 3 --device cuda
```

该命令只有在所有 Teacher 通过跟踪、峰值误差、请求动作总变差和饱和率门禁后才进入蒸馏。
每轮保存独立 dataset、Student checkpoint、闭环评测和 Teacher/Student 同图对比；最终目录按验证
闭环误差选择最佳轮次。单步入口仍保留用于诊断和消融。

### 纯奖励 TD3 修订实验

`scripts/41_train_pure_reward_td3.py` 是不使用 PID 示范、行为克隆或控制先验的独立实验入口。
它与上面的 7 维 PID-guided 正式流水线使用不同的 Actor 输入合同：在相同 7 个当前控制量后增加
`p_ref` 导数、`commanded/applied F_as` 和固定 26 步 requested-action 队列，共 35 维，在 50 Hz 下覆盖 0.52 s，
超过当前 3000 架库中的最大 `tau_p=0.498005 s`。所有待蒸馏 Teacher 必须使用同一个队列宽度。

该入口默认从连续参数分布采样 4--8 s 的 step、doublet、sine 和 multisine 指令，使用
`gamma=0.9995`；episode time limit 只触发环境 reset，不会清除 Bellman bootstrap，且 Critic 不读取
人工 episode progress。固定的六条 evaluation 指令仍只用于跨版本和 PID 同条件比较。

```bash
python scripts/41_train_pure_reward_td3.py \
  --plant-id <plant-id> --output <run-dir> \
  --steps 30000 --requested-action-history-steps 26 --device cuda
```

这是组合修订实验，不是单因素 delay ablation；若要归因，需要分别关闭动作记忆、截断修正、长
discount 和随机指令采样。

## 验证

```bash
pytest -q
```

单元与烟测只证明接口、时序、反向传播、checkpoint、蒸馏 split 和闭环评测可以运行，不证明
Teacher 已收敛，也不构成正式 GJB 一致性结论。
