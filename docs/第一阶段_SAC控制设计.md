# 第一阶段：单机 SAC Teacher 与条件 Student

本文冻结 2026-08-28 起采用的第一阶段代码契约。它取代历史的全机群统一 SAC、G0-G4、
response-cost Actor 和首阶段 MoE 路线。历史代码和报告只作回归材料。

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

`p_c` 是滚转角速度指令，单位为 `rad/s`，配置和图上同时给出 `deg/s`。理想响应由单位直流
增益二阶参考模型给出：

```math
M_{ref}(s)=\frac{\omega_n^2}{s^2+2\zeta_{ref}\omega_n s+\omega_n^2},
\qquad p_{ref}=M_{ref}p_c.
```

第一版取 `omega_n=2 rad/s`、`zeta_ref=0.7`。这是可修改、待实验验证的研究目标，不是 GJB
直接给出的唯一最佳参数。

控制链为：

```text
p_c -> reference model -> p_ref

[p_c, p_ref, p, error, history] -> SAC_i -> F_as -> G(theta_i) -> p
```

SAC 动作是传递函数的完整输入 `F_as`。不存在额外驾驶员力，也不使用 `F + delta_F`。

## 3. 必须保留的物理警告

当前 `p/F_as` 分子含一个 `s`，所以原点处有零点，直流增益为零。由此不能先验假定：有界的
恒定 `F_as` 一定能无限期维持非零恒定 `p`。二阶 step `p_ref` 先作为明确、可复现实验目标；若
控制器长期饱和仍无法跟踪，应首先检查目标可达性和输入定义，不能直接归因于 SAC 没学会。

图中的 `p_raw` 也需要准确解释。当前没有“原机 `p_c -> F_as` 控制器”，因此代码采用显式的
归一化开环基线：`p_c / 30 deg/s` 映射到 `F_as / 22 N`，再施加相同限幅和限速。它只用于
固定画图与回归比较，不是原型飞机已有控制律，也不是逆直流增益。

## 4. 单机 Teacher 契约

每个 Teacher 只控制一组训练期固定的 `theta_i`。所有 Teacher 网络结构相同，权重独立。

### Actor 输入

默认 Actor observation 为 1256 维：

| 区间 | 内容 |
| --- | --- |
| `0:6` | 当前归一化 `p_c, p_ref, p, p_ref-p, p_dot, previous F_as` |
| `6:1256` | 最近 250 ms 的 `250 x 5` 历史：`p_c, p_ref, p, error, F_as` |

`theta=[L_Fa, lambda_s, T_R, zeta_d, omega_d, R_omega, R_zeta, tau_p]` 不进入 Teacher Actor。
250 ms 覆盖训练机库约 0.20 s 的最大运输延迟并留出余量。固定飞机的动力学通过训练环境和响应
历史隐式体现。Critic 额外读取四维连续对象状态及 episode
进度，仍无需 `theta`。

### 动作与时序

- plant 步长：`0.001 s`；
- 策略周期：`0.001 s`，每个对象样本决策一次；
- Actor 输出：归一化 `a in [-1,1]`；
- 实际动作：`F_as=22a N`；
- 默认限速：`88 N/s`，即每毫秒最多变化 `0.088 N`；
- 默认不额外加入执行机构一阶滞后；对象自己的纯运输延迟仍由抽样 `tau_p` 决定。

延迟造成的后果通过 `p`、误差和历史返回 Actor。普通 SAC transition 保持
`(o_t,a_t,r_t,o_{t+1})`，不把 reward 人工挪给旧动作。

### Reward

每个 1 ms 样本使用三项归一化积分代价：

```math
r_t=-dt\left[
w_e\left(\frac{p_t-p_{ref,t}}{p_{scale}}\right)^2+
w_u\left(\frac{F_{as,t}}{F_{max}}\right)^2+
w_{du}\left(\frac{F_{as,t}-F_{as,t-1}}{\Delta F_{max}}\right)^2
\right].
```

默认 `w_e=1.0`、`w_u=0.02`、`w_du=0.05`。第一项同时处罚慢、快、超调和偏离参考的振荡；
后两项限制大力和逐毫秒抖动。GJB 的轨迹级指标不硬塞入逐步 reward。

### 指令与评测

默认 Teacher 先用 `+/-10, +/-20, +/-30 deg/s` step。扩展套件提供 doublet、sine 和
multisine。评测留出 `+/-15, +25 deg/s` step、不同 doublet、`0.75 Hz` sine 和另一组
multisine。

每个训练完成的 Teacher 写出：

- `teacher_actor.pt`：只含部署 Actor、固定飞机、配置和来源；
- `training_checkpoint.pt`：Actor、Critic、target Critic、优化器和温度；
- `evaluation.json`：逐指令 tracking、动作 RMS、总变差、饱和率和相对 raw 变化；
- `response_comparison.png`：`p_c, p_ref, p_raw, p_SAC` 与对应 `F_as`；
- `report.json` 和 `progress.json`。

默认 specialist Actor 宽 128、2 个残差块，约 23 万参数；含 twin Critic 的可训练量约 69 万。
该默认值用于先验证单机问题，不代表最终 Student 容量上限。

## 5. Teacher Bank

`scripts/12_train_specialists.py` 从训练 split 中按品质区间轮转选择飞机。一个飞机和一个 seed 对应
一个独立目录；`run_id` 同时包含飞机、seed 和完整配置 hash。`teacher_bank.json` 记录数据源
hash、配置、飞机、checkpoint 和完成状态，已完成
run 可跳过。

扩展顺序固定为：

```text
1 架完整链路 -> 10 架 -> 50/100 架 -> 再决定是否扩大
```

“3000 架参数库”不等于第一轮必须启动 3000 个 SAC。

## 6. 蒸馏数据

对每个 Teacher 在它实际训练过的指令套件上做确定性闭环 rollout，保存：

```text
(Teacher observation o_t, normalized theta_i, Teacher action a_i,t)
```

`theta` 只在收集标签时附加，不会反向进入 specialist Teacher。数据按 Teacher 分片保存；默认每
10 个 1 ms 样本保存一个监督样本，原闭环仍以 1 kHz 执行。

这样不会把未训练指令上的偶然 Teacher 行为当作默认监督标签；更丰富的 Student 数据应先把
Teacher 的 `command_mode` 切到 `extended` 并重新训练。Teacher 数量不少于 2 时，
train/validation 按整架飞机划分，同一飞机的相邻样本不会跨 split。只有 1 架飞机的管线烟测
使用“训练命令/留出命令”划分，并在 manifest 中明确标记
`single_aircraft_command_holdout`，不能当作跨飞机泛化证据。

## 7. Dense Student

Student 学习：

```math
\pi_{student}(o_t,\bar\theta_i)\approx\pi_i(o_t),
```

其中八维 `theta` 在固定物理范围内归一化，训练范围外的值默认不裁剪，以保留 OOD 信号。Student
输出仍是归一化完整 `F_as`。默认网络宽 512、8 个残差块，约 486 万参数；它是可配置的首个
Dense 基线，不是最终“大模型”规模结论。

蒸馏首先报告 action MSE/MAE；随后必须重新闭环比较 raw、specialist Teacher 和 Student，报告
tracking、harm rate、动作 RMS、总变差与饱和率。离线动作误差低不等于闭环稳定。

MoE 只在 Dense Student 出现明确容量不足或跨动力学负迁移后作为 Student 对照，不属于本阶段
默认训练链。

## 8. 代码入口

```bash
# 独立 specialist Teachers
python scripts/12_train_specialists.py --count 1 --device cuda

# Teacher Bank -> 分片监督数据
python scripts/30_collect_distillation_data.py --device cuda

# Dense conditional Student
python scripts/31_distill_dense_student.py --device cuda

# raw / Teacher / Student 闭环评测
python scripts/33_evaluate_student.py --device cuda
```

对应实现：

- `src/envs/reference_model.py`：二阶参考模型；
- `src/envs/roll_rate_commands.py`：训练/留出滚转率指令；
- `src/envs/specialist_tracking_env.py`：固定飞机、直接力动作环境；
- `src/teacher/specialist/`：单机训练和 Teacher Bank；
- `src/distillation/`：分片收集、监督训练与闭环验证；
- `src/student/dense/`：条件 Student 网络与部署包装。

## 9. 当前证据边界

代码烟测通过只表示架构接通。进入多机 Teacher 前，第一架飞机至少需要满足：曲线定义正确、奖励
分解有限、动作未靠持续饱和或高总变差取巧，并且 Teacher 相对明确的 raw 基线有可解释改善。
若参考本身在当前输入权限下不可达，应修改控制问题或参考，而不是盲目扩大网络和训练步数。
