# 第一阶段：单机 RL Teacher 与 theta 路由 Student

本文冻结 2026-08-29 起采用的第一阶段代码契约。它取代历史的全机群统一 SAC、G0-G4、
response-cost Actor 和 Dense Student 默认路线。历史代码和报告只作诊断材料。

## 1. 研究目标与边界

第一阶段只研究一个 SISO 通道：

```math
G_{\theta_i}(s)=\frac{p(s)}{F_{as}(s)}.
```

对第 `i` 架固定飞机先训练一个专用控制器 `pi_i`，随后把多个专用控制器蒸馏成统一
Student `pi_student(o, theta)`。当前模型只有滚转角速度 `p`，没有独立的 `beta`、`r`、真实
舵面或六自由度状态。因此本阶段叫“P-channel 滚转率整形”，不能声称已经单独控制或正式评定
荷兰滚、螺旋、滚偏航耦合或整机 GJB 飞行品质。

## 2. 信号定义

`p_c` 是滚转角速度指令，单位为 `rad/s`，配置和图上同时给出 `deg/s`。第 `i` 架飞机的理想
响应由单位直流增益二阶模型叠加该飞机同一个纯运输延迟给出：

```math
M_{ref,i}(s)=
\frac{\omega_n^2}{s^2+2\zeta_{ref}\omega_n s+\omega_n^2}e^{-\tau_{p,i}s},
\qquad p_{ref,i}=M_{ref,i}p_c.
```

第一版取 `omega_n=2 rad/s`、`zeta_ref=0.7`。这是可修改、待实验验证的研究目标，不是 GJB
直接给出的唯一最佳参数。复制 `tau_p` 是为了不处罚控制器无法消除的对象纯延迟；它不表示
`tau_p` 等于标准中的闭环等效延迟 `tau_e`。

控制链为：

```text
p_c -> same tau_p -> ideal second-order model -> p_ref

[p_c, p_ref, p, error, integral error, p_dot, u_previous] -> RL Teacher_i
                                                                    |
                                                                    v
                 F_as -> same tau_p -> G0(theta_i) -> p
```

Teacher 动作是传递函数的完整输入 `F_as`。不存在额外驾驶员力，也不使用 `F + delta_F`。

## 3. 必须保留的物理警告

当前 `p/F_as` 分子含一个 `s`，所以原点处有零点，直流增益为零。由此不能先验假定：有界的
恒定 `F_as` 一定能无限期维持非零恒定 `p`。二阶 step `p_ref` 先作为明确、可复现实验目标；若
控制器长期饱和仍无法跟踪，应首先检查目标可达性和输入定义，不能直接归因于 RL 没学会。

图中的 `p_raw` 也需要准确解释。当前没有“原机 `p_c -> F_as` 控制器”，因此代码采用显式的
归一化开环基线：`p_c / 30 deg/s` 映射到 `F_as / 22 N`，再施加相同限幅和限速。它只用于
固定画图与回归比较，不是原型飞机已有控制律，也不是逆直流增益。

## 4. 单机 Teacher 契约

每个 Teacher 只控制一组训练期固定的 `theta_i`。所有 Teacher 网络结构相同，权重独立。
正式 Teacher 使用 PID-guided TD3：将调好的归一化线性 PID 律嵌入 Actor 作为控制先验，
在线 TD3 更新只学习幅值受限的小残差。部署 checkpoint 内是一个完整 Actor，推理时不实例化
PID 控制器对象。线性先验与无偏置奇对称残差使 `pi(-o)=-pi(o)` 结构上成立，因而零指令、
零状态时请求动作严格为零。

### Actor 输入

默认 Actor observation 为 7 维：

| 区间 | 类别 | 内容 |
| --- | --- | --- |
| `0:4` | 瞬时信号 | 归一化 `p_c, p_ref, p, p_ref-p` |
| `4:7` | 控制器状态 | 归一化 `integral error, p_dot, previous requested F_as` |

`theta=[L_Fa, lambda_s, T_R, zeta_d, omega_d, R_omega, R_zeta, tau_p]` 不进入 Teacher Actor。
每个 Teacher 的对象在训练期固定，因此飞机身份隐含在该 Teacher 的权重中，不需要 TCN 做在线
辨识。后三项依赖累计值或上一拍，属于历史派生的有限维控制器状态；它们不是 TCN/GRU 所需的
原始序列窗口。`history_steps` 只表示额外的原始 observation 窗口，默认是 0。

训练期使用 asymmetric / privileged Critic。它在上述 Actor observation 之外读取四维对象连续
状态、对象纯时延 FIFO、commanded/applied force、episode 进度和命令 profile one-hot。这样 Q
函数对延迟对象保持 Markov：即使当前 `p` 相同，Critic 仍能区分管道中尚未到达对象的旧控制力。
这些字段只存在于训练期 Critic，不进入 Teacher Actor、蒸馏数据 observation 或最终 Student；
每个 checkpoint 都保存独立的 `actor_observation_contract` 和
`critic_observation_contract`，不能将两者混用。固定飞机 Critic 仍无需 `theta`。

### 动作与时序

- plant 步长：`0.001 s`；
- 策略周期：`0.020 s`，每次决策内执行 20 个 `0.001 s` 对象子步；
- Actor 输出：归一化 `a in [-1,1]`；
- 实际动作：`F_as=22a N`；
- 默认限速：`88 N/s`，即每毫秒最多变化 `0.088 N`；
- 默认不额外加入执行机构一阶滞后；对象输入与独立 Reference 分支使用同一个抽样纯运输延迟
  `tau_p`。

延迟造成的后果通过闭环观测返回 Actor。TD3 transition 保持
`(o_t,a_t,r_t,o_{t+1})`，不把 reward 人工挪给旧动作。

### Reward

每个 20 ms 策略步使用三项归一化代价。跟踪与力能量按时间积分，请求动作跳变按每次决策计：

```math
r_t=-s_r\left\{\Delta t\left[
w_e\left(\frac{p_t-p_{ref,t}}{p_{scale}}\right)^2+
w_u\left(\frac{F_{as,t}}{F_{max}}\right)^2
\right]+w_{du}(a_t-a_{t-1})^2\right\}.
```

当前基线取 `s_r=1/0.02=50`、`w_e=1.0`、`w_u=0.02`、`w_du=0.02`；该值由同一飞机、
同一 seed 的受控权重筛选冻结。`a_t-a_{t-1}` 直接作用于网络请求动作且不再额外乘 `dt`，否则在 20 ms 周期下会
被错误削弱 50 倍。执行机构限速器不能替策略掩盖抖动。GJB 的轨迹级指标不硬塞入逐步 reward。

### 指令与评测

默认 Teacher 先用 `+/-10, +/-20, +/-30 deg/s` step。扩展套件提供 doublet、sine 和
multisine。评测留出 `+/-15, +25 deg/s` step、不同 doublet、`0.75 Hz` sine 和另一组
multisine。训练 episode 按带种子的随机排列无放回轮转，每完成一轮命令库再洗牌，避免有限训练
预算下正负指令抽样失衡。

每个训练完成的 Teacher 写出：

- `teacher_actor.pt`：只含部署 Actor、固定飞机、配置和来源；
- `training_checkpoint.pt`：Actor、twin Critic、target 网络和优化器；
- `evaluation.json`：逐指令 tracking、动作 RMS、总变差、饱和率和相对 raw 变化；
- `evaluation.json` 同时把 episode cost 拆成 tracking、force energy 和 requested-force delta 三项，
  用于检查 Reward 权衡；
- `response_comparison.png`：`p_c, p_ref, p_raw, p_RL Teacher` 与对应 `F_as`；
- `all_evaluation_commands.png`：全部留出命令及 requested/applied `F_as`，避免只展示单条曲线；
- `report.json` 和 `progress.json`。

正式大 Teacher 使用宽 704、10 个残差块：部署 Actor 为 `9,948,225` 个可训练参数，
含 twin Critic 的训练期总参数为 `29,915,075`。约 10M 指单个部署 Actor，不包括只在训练期使用的
Critic 和 target 网络。

## 5. Teacher Bank

`scripts/16_train_pid_guided_teacher_bank.py` 从训练 split 中按品质区间轮转选择飞机。一个飞机和一个 seed 对应
一个独立目录；`run_id` 同时包含飞机、seed 和完整配置 hash。`teacher_bank.json` 记录数据源
hash、配置、飞机、checkpoint 和完成状态，已完成 run 可跳过。`--workers` 可并行训练多个独立
Teacher。只有 tracking improvement、平均 RMSE、峰值误差、requested action 总变差和饱和率
全部通过门禁的 Teacher 才能进入完整 Bank；否则 Bank 状态为 `quality_gate_failed`，蒸馏拒绝启动。
正式门禁要求所有留出命令均优于开环基线、平均 RMSE 不超过 `1 deg/s`、最大峰值误差不超过
`3 deg/s`、平均 requested-force 总变差不超过 `50 N`、平均饱和比例不超过 `1%`。

扩展顺序固定为：

```text
1 架完整链路 -> 10 架 -> 50/100 架 -> 再决定是否扩大
```

“3000 架参数库”不等于第一轮必须启动 3000 个 RL Teacher。

## 6. 蒸馏数据

第 0 轮对每个 Teacher 做确定性闭环 rollout，保存：

```text
(Teacher observation o_t, normalized theta_i, Teacher action a_i,t)
```

`theta` 只在收集标签时附加，不会反向进入 specialist Teacher。随后第 1 轮起使用当前 Student
在每架飞机上闭环飞行，并在 Student 实际访问的每个 observation 上查询对应 Teacher：

```text
Student(o, theta) 驱动对象 -> visited o -> matching Teacher(o) 给标签
```

每轮数据与此前所有轮次聚合后继续训练 Student。这是纯 Student-driven DAgger，用来修正离线
Teacher 轨迹造成的 covariate shift。manifest 的每个 shard 明确记录 `collection_round`、
`driver` 和 Teacher checkpoint hash。

正式第一阶段使用 `all_aircraft_command_holdout`：Teacher Bank 中的所有飞机都用各自
`train-*` 指令参与 Student 拟合，每架飞机各自的 `eval-*` 指令只用于闭环 validation。
`aircraft_holdout` 保留为更严格的零样本诊断；当前 11 架 Teacher 的覆盖度还不支持对完全未见
`theta` 声称泛化成功。指令 holdout 与整机 holdout 的结果必须分开报告。

## 7. theta 路由线性 MoE Student

Student 学习：

```math
\pi_{student}(o_t,\bar\theta_i)\approx\pi_i(o_t),
```

其中八维 `theta` 在固定物理范围内归一化。Student 输出仍是归一化完整 `F_as`。
路由器只读静态 `theta`，因而同一架飞机在整段时域响应中的路由权重不会跳变；每个无偏置线性专家
只读 observation 的 `[3,4,5]`，即 `error, integrated error, p_dot`。这样不会利用
`p_c, p_ref, p` 之间的线性相关性走离线拟合捷径。当前 11 专家模型只有 121 个可训练参数；
容量由闭环效果决定，不为了模型规模人为扩大。

蒸馏首先报告 action MSE/MAE；每轮随后重新闭环比较 raw、specialist Teacher 和 Student，报告
tracking、harm rate、动作 RMS、总变差与饱和率。每架飞机生成同一坐标系下的
`p_c / p_ref / Teacher / Student` 与控制力对比图。最终 checkpoint 按留出指令上的闭环
Student-Teacher RMSE 差和质量门禁选择，而不是按离线 MSE 选择。Dense Student 与整机 holdout
MoE 结果保留在 `diagnostics/`，不进入正式结论。

## 8. 代码入口

```bash
# 完整门禁流水线
python scripts/35_run_teacher_student_pipeline.py \
  --teacher-algorithm pid-guided-td3 \
  --plant-id <plant-1> --plant-id <plant-2> \
  --teacher-network-width 704 --teacher-residual-blocks 10 \
  --student-architecture theta_routed_linear_moe \
  --distillation-split-strategy all_aircraft_command_holdout \
  --dagger-rounds 2 --device cuda

# 已有 Teacher Bank 时，只运行 student-driven 蒸馏
python scripts/34_distill_student_driven.py \
  --split-strategy all_aircraft_command_holdout \
  --student-architecture theta_routed_linear_moe \
  --dagger-rounds 2 --device cuda
```

对应实现：

- `src/envs/reference_model.py`：二阶参考模型；
- `src/envs/roll_rate_commands.py`：训练/留出滚转率指令；
- `src/envs/specialist_tracking_env.py`：固定飞机、直接力动作环境；
- `src/teacher/specialist/`：单机训练和 Teacher Bank；
- `src/distillation/`：分片收集、监督训练与闭环验证；
- `src/student/moe/`：`theta` 路由线性 MoE Student；
- `src/student/dense/policy.py`：Dense/MoE checkpoint 的统一部署包装；
- `scripts/15_compare_teacher_pid.py`：在同一六命令时域图和 JSON 中比较 raw、PID 与 RL Teacher。

## 9. 当前证据边界

2026-08-29 的正式运行包含 11 架飞机（Level 1/2/3 分别为 5/3/3）和每架 6 个
留出指令，共 66 个闭环组合。平均 tracking RMSE 为：Raw `16.667157 deg/s`、PID
`0.297852 deg/s`、RL Teacher `0.297807 deg/s`、Student `0.297813 deg/s`。Student 的最大
峰值误差为 `2.176415 deg/s`，无控制力饱和，请求力总变差与 Teacher 之比为 `1.000176`。

这组证据支持“一个无原始历史窗口的 `theta` 路由 Student 可以在已知 Teacher-Bank 飞机上
复现独立 RL Teacher”。它不支持两个更强结论：第一，当前 RL Teacher 与 PID 几乎等效，
未显示出有意义的性能超越；第二，当前是已知飞机上的指令 holdout，不是对完全未见
`theta` 的零样本泛化证据。本阶段仍只是 P-channel 时域跟踪研究，不等于完整 GJB
横航向评定。
