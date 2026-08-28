# 第一阶段 SAC 单通道滚转品质控制设计

本文冻结当前第一阶段的控制对象、1 ms 时序、Actor 输入、reward、网络规模和评测口径。
它取代历史 G0-G4 路线以及以固定参考模型跟踪误差为主目标的设计。

## 1. 控制对象

当前只研究单输入单输出通道：

```text
F_pilot(t) + delta_F_RL(t) = F_eq(t)
                            |
                            v
               P-channel: F_eq -> p
                            |
                            v
                   roll rate p(t)
```

SAC 输出归一化动作 `a in [-1, 1]`，映射为 `delta_F_RL in [-6.6, 6.6] N`。
命令限速为归一化 `4/s`，对应 `26.4 N/s`。当前默认不加入没有物理证据的
`0.08 s` 修正通道一阶滞后，即 `augmentation_lag_time_constant_s=0`。

这不是副翼/方向舵的完整物理闭环。模型只有 `p`，没有独立的 `beta`、`r` 和真实舵面状态，
因此不能把结果写成完整荷兰滚、螺旋或正式 GJB 一致性结论。

## 2. 1 ms 控制时序

对象与策略均固定为 `dt=0.001 s`，每个对象样本都允许 SAC 决策一次：

```text
时刻 t 已有响应 p_t、趋势和当前品质代价 c_t
                    |
                    v
Actor(s_t, c_t, theta) -> a_t -> 限幅/限速 -> delta_F_t
                    |
                    v
F_pilot,t + delta_F_t -> 对象内部纯运输延迟 tau_p -> P-channel 推进 1 ms
                    |
                    v
得到 p_(t+1)，计算 c_(t+1) 和 r_(t+1)，构造下一观测
```

`c_t` 主要来自更早动作经过运输延迟后刚刚显现的响应。它必须进入 Actor，Actor 才能据此
修正下一步力。实现不把 reward 人工搬到旧 transition，不构造额外延迟 replay，也不把对象
内部 FIFO 暴露给 Actor 或 Critic；经验池保存普通的一步 transition。

## 3. 延迟的三个不同量

| 量 | 含义 | 当前处理 |
| --- | --- | --- |
| `tau_p` | 每架飞机抽样后固定的输入前纯运输延迟 | 对象内部 `FractionalDelay`；最小抽样值 `0.001 s`；同时作为 `theta` 输入 |
| `delay_response_cost_t` | 当前指令已经等待响应的时长所形成的 Actor 反馈 | 响应出现前逐毫秒增长，`log(1 + wait/0.1s)` 后输入 Actor |
| `added_onset_delay` | 控制器相对同机 raw 额外造成的响应迟滞 | 只作为 reward；不处罚不可改变的 `tau_p` 常数 |

GJB 表 A21 的 `tau_e` 是完整系统响应经 LOES 拟合得到的等效延迟。当前没有拟合 `tau_e`，
也不把 `tau_p` 直接按表 A21 分级。

## 4. Actor 输入

默认 Actor 输入为 268 维，排列固定如下：

| 索引 | 内容 |
| --- | --- |
| `0:6` | 当前 `F_pilot/22N, p, p_dot/5, phi, a_cmd_prev, a_applied` |
| `6:10` | 当前 `roll_cost, oscillation_cost, spiral_recovery_cost, delay_response_cost` |
| `10:260` | 最近 50 ms 的 50 x 5 历史：`F, p, p_dot, phi, a_applied` |
| `260:268` | 归一化 `theta=[L_Fa, lambda_s, T_R, zeta_d, omega_d, R_omega, R_zeta, tau_p]` |

四个品质反馈使用 `log1p` 压缩非负代价，防止少数 OOD 响应使网络输入失控。
Critic 额外读取四维连续 P-channel 状态和 episode 进度，总输入为 273 维；它同样不读取延迟 FIFO。

## 5. 当前 reward

不使用固定 `p_ref(t)`。每一步 reward 为当前响应代价与动作代价的负和：

```math
r_t = -(0.25 c_{wrong}
      + 1.00 c_{added-delay}
      + 1.00 c_{S1s}
      + 1.00 c_{osc}
      + 0.50 c_{spiral-recovery}
      + 0.02 a_t^2 dt
      + 0.05 (Delta a_t / Delta a_max)^2 dt).
```

持续性响应和动作代价按秒积分，因此从 50 Hz 改到 1 kHz 不会把它们机械放大 20 倍。

- `wrong-way`：超过运输延迟宽限后，滚转速率仍与当前力方向相反。
- `S1s`：从零开始并保持的阶跃，在 1 s 时超过 IV-A 杆式 Level 1 建议上限 `3.38 deg/N`。
- `oscillation`：离散保持指令下检测 `P1 -> P2 -> P3`，使用 A120 形式的响应比作为训练代理。
- `spiral-recovery`：pulse/doublet 等释放后仍存在的归一化滚转速率能量。
- `added-delay`：raw 已有可测响应但受控响应仍未出现时的额外迟滞。

直接处罚 `T_R`、`lambda_s`、`zeta_d`、`omega_d` 或 `tau_p` 这些固定抽样参数，只会给同一架
飞机增加动作无法改变的常数 reward。它们用于上下文和分层；真正驱动控制的是可被动作改变的响应代价。

## 6. 指令分布

训练与统一评测使用同一个 `CommandProfile` 实现，但评测参数单独留出：

- 正负、多幅值 step；
- pulse；
- 正负 doublet；
- 多频率 square；
- 多频率 sine；
- chirp；
- 正负 staircase；
- 固定随机种子的 piecewise 指令。

每条指令在 1 ms 对象网格上预先生成，默认长度 10 s。训练与评测清单不共享幅值、频率、
时序或随机种子。

## 7. 每次统一比较的响应

| 指令 | 主要比较量 | 结论边界 |
| --- | --- | --- |
| held step | 响应起始迟滞、`S1s`、峰值/RMS、A120 振荡比代理 | A116 仍需适用性和相角边界检查 |
| pulse/doublet | 释放后 `p` RMS、后续滚转角漂移 | 只能称螺旋/恢复代理，不能代替闭环 `lambda_s` 辨识 |
| sine | 幅值增益、相位滞后 | 单频响应诊断 |
| chirp | 保留完整扫频轨迹 | 后续 LOES/频响辨识输入；当前不输出正式 `tau_e` |
| 所有指令 | reward 分解、动作 RMS、总变差、饱和率、cancellation | 防止靠大力、抖动或抵消驾驶员输入取巧 |

统一控制器顺序为 `Raw -> Linear -> MLP-SAC Teacher -> 4-expert MoE-SAC Teacher`。

## 8. Teacher 数量与网络规模

MLP 与 MoE 各训练 5 个独立随机种子，共 10 个正式 run。每个 run 有独立目录、日志、
checkpoint 和完整报告；先跑 50,000 step 吞吐试验，再决定是否执行 2,000,000 step 训练。

| 项目 | MLP-SAC | 4-expert MoE-SAC |
| --- | ---: | ---: |
| Actor 主干 | 宽 896，14 个全宽残差块 | 宽 896，10 个共享残差块；4 个 expert 各 2 个 448 瓶颈残差块 |
| Actor 参数量 | 22,773,634 | 23,046,790 |
| MoE/MLP Actor 比 | - | 101.20% |
| Twin Critic 参数量 | 45,556,226 | 45,556,226 |
| 可训练总量（Actor + Twin Critic + alpha） | 68,329,861 | 68,603,017 |
| 含冻结 target Critic 的网络存储量 | 113,886,086 | 114,159,242 |
| Router 输入 | - | 仅最后 8 维 `theta` |

每个全宽残差块采用 `Pre-LayerNorm -> Linear -> SiLU -> Linear -> 0.1 residual -> add`；
MoE expert 使用同样结构但把中间宽度降到 448。残差结构用于稳定深网络的梯度，不能替代梯度裁剪、
初始化检查和训练吞吐验证。两种 Teacher 的可训练参数均处于 6,000-7,000 万区间，差异约 `0.40%`；
公平比较以总可训练参数为准，而不是只比较 Actor。MoE router 使用 `512-256` 的 `theta` 编码器和
宽 512 的路由层，仍严格不读取响应状态。

共同 SAC 参数：`gamma=0.9999`、target Polyak `0.005`、学习率 `3e-4`、自动温度初值
`alpha=0.1`、目标熵 `-1`、batch `256`、replay `200,000`、warmup `20,000`、梯度范数上限
`10`。MoE 额外使用 `0.01` 专家均衡系数，并报告每个专家平均使用率和 router entropy。

长训练尚未启动。当前代码与短烟测只能证明接口和优化步骤可运行，不能证明控制效果。
