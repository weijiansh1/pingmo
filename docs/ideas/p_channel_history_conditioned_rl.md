# P 通道历史条件强化学习控制研究思路

更新时间：2026-09-03

状态：研究设计修正版，尚未冻结为最终实验协议。

## 1. 一句话目标

针对一组动力学参数和纯时延不同的飞机，训练一个不读取传递函数参数的最终 Student。Student 仅根据公共指令、可测量状态和近期输入输出历史，在线适应当前飞机，并把不同飞机的 `p` 通道整形成统一的目标飞行品质响应。

```text
Can one history-conditioned RL controller adapt to hidden aircraft
dynamics and delay, and approach per-aircraft specialist performance?
```

这项工作的核心不是“做一个很大的 SAC”，也不是“证明 PID 不能泛化”，而是研究：

```text
hidden plant family
        +
one deployable controller
        +
online adaptation from I/O history
        |
        v
uniform desired p-channel quality
```

## 2. 被控对象

每架飞机的单输入单输出模型为：

```text
G_i(s) = p(s) / F_as(s)
       = G_0,i(s) * exp(-tau_i * s)
```

其中：

```text
G_0,i(s)
  |
  +-- static gain
  +-- zeros
  +-- roll pole
  +-- spiral pole
  +-- Dutch-roll poles
  +-- other modeled dynamics

tau_i
  |
  +-- physical pure transport delay
```

同一个控制力输入作用于不同飞机，会得到不同的时域响应：

```text
same 3 N step
      |
      +----> Aircraft A ----> fast and well damped p(t)
      |
      +----> Aircraft B ----> oscillatory p(t)
      |
      +----> Aircraft C ----> slow and delayed p(t)
```

开环 `3 N` 阶跃实验只用于验证飞机模型和展示飞机族差异，不等同于闭环控制实验。

## 3. 最终闭环结构

```text
                         common desired response
                                  p_ref_0
                                     |
                                     v
     p_c --------------------> +-------------+
                               |             |
     current p --------------> |   Student   | ----> requested F_as
                               |             |              |
     recent p/u history -----> +-------------+              v
                                                        +---------+
                                                        | Aircraft|
                                                        |   G_i   |
                                                        +---------+
                                                             |
                                                             +----> p
```

最终 Actor 不读取以下参数：

```text
theta_i = [gain, poles, zeros, zeta_d, omega_d, tau_i, ...]
```

它必须根据近期的输入输出历史形成隐式飞机表征。

## 4. Observation 边界

首选的可部署 Actor 输入为：

```text
current signals:
    p_c
    p_ref_0
    p
    e_0 = p_ref_0 - p

history signals:
    p[t-H : t]
    requested_F_as[t-H : t]
    p_c[t-H : t]
```

如果真实系统能够测量执行机构实际输出，才增加：

```text
applied_F_as[t-H : t]
```

不能为了仿真方便，把真实部署时不可获得的延迟队列直接输入 Actor。

仅看当前时刻通常不是 Markov 状态：

```text
Aircraft A: tau = 0.03 s
Aircraft B: tau = 0.30 s

At time t:
    same p_c
    same p
    same error
    same previous action

but:
    different actions are still pending inside the two plants
```

历史编码器的作用是从“过去发出了什么”和“随后飞机怎样响应”中估计增益、延迟、频率和阻尼。

## 5. Reference 与延迟的修正

### 5.1 不能向 Student 泄露 tau

曾考虑给每架飞机生成：

```text
p_ref_i(t) = p_ref_0(t - tau_i)
```

这可以用于固定飞机的专用 Teacher，但如果把 `p_ref_i` 输入 Student，就会形成隐式参数泄露：

```text
tau_i --> shifted p_ref_i --> Student
```

而且真实部署时，如果 `tau_i` 未知，也无法生成这个信号。

### 5.2 修正后的信息边界

Actor 只接收与飞机无关的公共参考：

```text
p_ref_0 = M_0(s) * p_c
```

训练环境内部可以利用已知仿真参数构造奖励目标：

```text
p_ref_reward_i(t) = p_ref_0(t - tau_i)
```

于是：

```text
Actor input:
    common p_ref_0

Reward evaluator only:
    delayed(p_ref_0, tau_i)
```

这属于训练环境的 privileged reward information，不属于部署时的 Actor 输入。是否让 Critic 看见 `tau_i` 必须作为单独 ablation；第一版首选 Actor 和 Critic 都不直接读取 `theta_i`。

### 5.3 物理边界

因果控制器不能消除纯时延。目标应该是：

```text
do not claim:
    RL removes tau_i

claim:
    after the unavoidable tau_i,
    RL improves damping, overshoot and response shape
```

时延本身在评价中单独报告。如果某架飞机的 `tau_i` 已经超过飞行品质允许上限，应标记为物理不可行或超出研究包线，不能要求 RL 凭空修复。

## 6. PID 的正确角色

不能从“一组 PID 增益跨飞机失败”推出“PID 不能泛化”。固定 PID、增益调度 PID、自适应控制是不同基线。

```text
1. Local PID_i
   tuned separately for Aircraft i
   role: per-aircraft classical oracle

2. Frozen nominal PID
   one fixed gain set for every aircraft
   role: non-adaptive robustness baseline

3. Gain-scheduled PID(theta_i)
   explicit parameters select/interpolate gains
   role: privileged model-conditioned baseline
```

正确的比较关系为：

```text
                              expected role
Local PID_i                  classical upper reference
Specialist RL Teacher_i      learned specialist reference
Frozen nominal PID           fixed-controller baseline
Gain-scheduled PID(theta)    privileged adaptive baseline
One history Student          proposed deployable controller
```

跨飞机 PID 实验只能支持有限结论：

```text
PID_0803 applied to Aircraft_0306 becomes unstable
        |
        v
this frozen gain set is not robust for this plant pair
```

它不能单独证明 RL 更好。最终需要证明 Student 在相同条件下比 frozen PID 更稳，并尽量接近 `Local PID_i`。

## 7. Teacher 的角色

Teacher 是产生高质量专用控制知识的训练工具，不是最终部署控制器。

```text
Aircraft 1 <----> Teacher 1
Aircraft 2 <----> Teacher 2
Aircraft 3 <----> Teacher 3
    ...               ...
Aircraft N <----> Teacher N
```

每个 Teacher 的最低要求：

```text
- stable on every required command
- tracking close to its local PID oracle
- no persistent tail oscillation
- no sustained force chattering
- obey force and rate limits
- pass every-command gates, not only average gates
```

Teacher 的数量不应预先固定为 32、100 或全部飞机。首选方式是先覆盖代表性区域，再由 Student 的失败区域驱动增补。

```text
initial representative Teachers
              |
              v
         train Student
              |
              v
 locate aircraft regions where Student fails
              |
              v
 add or improve Teachers in those regions
              |
              +----------> repeat
```

## 8. Student-driven 蒸馏

Student-driven 的关键是训练状态来自 Student 自己的 rollout，而不是只模仿 Teacher 轨迹。

```text
sample Aircraft i
        |
        v
Student rolls out on Aircraft i
        |
        v
collect states visited by Student
        |
        v
query Teacher_i on those states
        |
        v
update one Student
        |
        +----------> repeat
```

最终推理时：

```text
unseen Aircraft
      |
      v
one Student + history encoder
      |
      v
F_as
```

未见飞机没有对应 Teacher。Teacher 只在训练阶段存在。

## 9. TCN、GRU、MoE 的位置

TCN 或 GRU 是历史编码器，用于构造隐式飞机特征：

```text
[p history, command history, action history]
                     |
                     v
                 TCN / GRU
                     |
                     v
               latent feature z_t
```

MoE 是可选的策略容量和条件计算结构：

```text
latent z_t --> router --> expert 1 --+
                         expert 2 --+--> F_as
                         expert K --+
```

它们不应成为论文主线本身：

```text
research question > observation contract > reward/evaluation > algorithm size
```

首个可信基线应先验证普通 history-conditioned policy。只有容量或多模态动力学成为明确瓶颈后，再用 MoE，并加入负载均衡和路由熵约束解决专家集中问题。

## 10. Reward 与评价

第一版逐步奖励可以保持可解释：

```text
e_i(t) = p_ref_reward_i(t) - p_i(t)

r_t = -w_e  * normalized(e_i)^2
      -w_u  * normalized(F_as)^2
      -w_du * normalized(delta_F_as)^2
```

这三项分别约束：

```text
tracking quality
control effort
action smoothness
```

当前已发现的风险是：平方动作增量可能容忍长期的小幅反复动作，而评价使用绝对总变化量时会判定为持续抖动。

```text
training term:
    sum(delta_u^2)

evaluation metric:
    sum(abs(delta_u)) / duration

small repeated delta_u may look cheap during training
but produce a large TV rate during evaluation
```

因此 Teacher/Student 的正式录用门禁必须逐条指令检查：

```text
for every required command:
    RMSE              < threshold
    peak error        < threshold
    force TV rate     < threshold
    saturation ratio < threshold
    tail oscillation  < threshold
```

不能仅使用六条指令平均值。若简单 reward 反复产生尾部振荡，再通过 ablation 增加窗口尾部误差或振荡能量项，而不是一开始堆叠大量难解释的指标。

## 11. 训练与评价时长

训练可以使用较短、随机化的窗口提高效率，但最终评价必须覆盖慢速 Dutch-roll 模态。

```text
training:
    short randomized command windows
    many command types and amplitudes

evaluation:
    long deterministic traces
    at least several periods of the slowest relevant mode
```

若 Dutch-roll 周期约为 `14.4 s`，`10 s` 评价不能证明振荡已经衰减。正式评价时长应根据飞机参数自动确定，例如取固定下限与若干个模态周期的较大值。

## 12. 公平实验矩阵

```text
Controller                 Seen plants       Unseen plants
----------------------------------------------------------------
Local PID_i                oracle             oracle for analysis
Frozen nominal PID         baseline           baseline
Gain-scheduled PID(theta)  privileged         privileged
RL Teacher_i               specialist         unavailable
One RL Student             evaluated          key result
```

所有可比实验必须使用：

```text
same p_c samples
same initial conditions
same plant integration step
same policy update period
same force magnitude limit
same force rate limit
same actuator dynamics
same evaluation horizon
same per-command metrics
```

需要区分两种 PID 图：

```text
Experiment A:
    Local PID_0306 vs transferred PID_0803
    both control Aircraft_0306
    question: does the transferred controller work on the target?

Experiment B:
    the same frozen PID_0803 controls Aircraft_0803 and Aircraft_0306
    question: how does one controller respond on two plants?
```

两种实验都需要，但不能混用结论。

## 13. 推荐实验顺序

```text
Stage 0
Plant validation
3 N step -> delay, gain, period, damping

Stage 1
Classical baselines
Local PID_i + frozen PID + optional gain scheduling

Stage 2
Single-aircraft RL validity
Raw vs local PID vs one RL Teacher

Stage 3
Representative Teacher set
Every Teacher passes per-command gates

Stage 4
History-conditioned single Student
Student-driven aggregation/distillation

Stage 5
Unseen-aircraft evaluation
Interpolation + boundary + delay stress tests

Stage 6
Optional MoE and RL fine-tuning
Only after the basic Student is credible
```

## 14. 当前结果应如何定性

当前已有结果仍有研究价值，但暂时只能称为探索性结果：

```text
68 attempted Teachers
        |
        +-- 32 passed the old averaged bank gates
        +-- 36 rejected

post-hoc audit:
        7 / 32 accepted Teachers exceeded the old TV threshold
        on at least one individual command
```

这说明旧门禁会被跨指令平均值稀释。现有 Teacher Bank 和 Student 结果不能直接作为最终结论，需要在 Reference 信息边界、延迟处理和逐指令门禁冻结后重新评价。

## 15. 当前可冻结的主张

```text
Use specialist RL controllers to provide per-aircraft control knowledge.
Use student-driven distillation to obtain one history-conditioned Student.
The Student does not observe transfer-function parameters and must adapt
to hidden aircraft dynamics and delay from recent input-output history.
```

中文表述：

```text
利用专用 RL Teacher 学习不同飞机的控制规律，
再通过 student-driven 蒸馏得到一个最终 Student。
Student 不读取传递函数参数，而是根据近期输入输出历史，
在线适应未知飞机动力学与延迟，并逼近专用控制器的性能。
```

## 16. 下一步冻结项

在继续训练前，应依次冻结：

```text
1. Actor 实际可获得的 observation
2. common p_ref_0 的动态参数和依据
3. reward 中 tau_i 的使用范围
4. PID 三类基线及其公平协议
5. Teacher 的逐指令录用门禁
6. 自动评价时长规则
7. Student-driven 数据聚合流程
```

这些项目冻结后，再重新判断应该训练多少 Teacher、是否需要 MoE，以及当前结果中哪些可以保留。
