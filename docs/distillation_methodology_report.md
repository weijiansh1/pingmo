# pingmo 蒸馏方法论文献综述与迭代路线

> 目标：把 pingmo 的"多专家 Teacher → 单一 Dense Student"蒸馏从"试错调参"收敛为一条有文献依据、
> 可归因、可复现的方法论，并给出按预期收益排序的落地路线图。
>
> 本文档是对 `stability_aware_v4_results_20260830.md`、`student_slew_limit_diagnostic_v5_20260830.md`
> 和 `v6_implementation_spec.md` 的上游综述：它回答"为什么当前做法不够、文献里哪些方法能补、按什么顺序做"。

---

## 1. 摘要

pingmo 当前的蒸馏是 **策略蒸馏（Policy Distillation）+ DAgger + 行为克隆（Behavioral Cloning）** 的混合体：
用 SAC/TD3 逐机训练的专家策略作 Teacher，按 DAgger 的"Student 闭环访问、Teacher 标注"循环聚合数据，再用
**加权动作 MSE + 增量 MSE** 监督回归一个确定性 Dense Student。

把它放进文献谱系后，三个失败模式的机理与对应补法都很清楚：

| 失败模式 | 观察值 | 文献根因 | 对应补法（本文推荐） |
| --- | ---: | --- | --- |
| requested-force TV 比超标 | `2.133 > 1.25` | own-delta 约定允许 Student 绝对基线漂移；监督信号只有动作均值 | 残差动作头（已实现 V6）+ 价值/优势加权 |
| doublet 快速反向处峰值 | `7.98 deg/s`（v5 限速） | 单一固定 slew 上限抹平了必要的瞬态动作带宽 | 增量头 + 只惩罚"超出 Teacher 局部变化率"的 excess（V6 已实现） |
| 未见飞机 RMSE 落后逐机 PID | `2.38 vs 1.41 deg/s` | 只做单 Teacher 动作匹配，跨 `theta` 局部插值差；分布漂移（covariate shift） | 集成/共识监督 + DART 噪声注入 + 价值蒸馏 |

核心论断一句话：**当前方法在"模仿信号梯度"里只用到最弱的一档（动作均值回归），
文献给的可加杠杆依次是 动作增量 → 动作分布（KL）→ 专家价值/优势（AggreVaTe）→ 集成共识。
每上一档，都在往"让 Student 学习'什么是好动作'而不只是'Teacher 做了什么动作'"走。**

---

## 2. pingmo 蒸馏在文献谱系中的定位

### 2.1 谱系与 pingmo 的对应

| 范式 | 代表工作 | 核心监督信号 | pingmo 现状 |
| --- | --- | --- | --- |
| 行为克隆（BC） | Pomerleau 1989；Bain & Sammut 1995 | 状态→动作 回归 | ✅ round 0 Teacher 驱动 |
| DAgger | Ross, Gordon, Bagnell 2011 (AISTATS) | 学习者访问状态上的专家标注 | ✅ round 1..N Student 驱动 |
| DART | Laskey et al. 2017 (CoRL) | 专家动作注入噪声 → 覆盖误差分布 | ❌ 未实现 |
| 策略蒸馏 | Rusu et al. 2016 (arXiv:1511.06295) | Q 值/动作的 KL 或回归 | ⚠️ 只做了动作回归，未做 Q/KL |
| 价值模仿 | Ross & Bagnell 2014；Sun et al. 2017 ICML（AggreVaTe/AggreVaTeD） | 专家 cost-to-go / Q 值 | ❌ 未实现（critic 已存但未用） |
| 集成蒸馏 | EPD/DPD（AAAI 2020）、BRED（NeurIPS offRL） | 多 Teacher 共识/方差 | ⚠️ 有 32 个专家但逐机单对单标注 |
| 置信/不确定性加权 | 2IWIL（Wu et al. 2019）、CAIL（2021）、CIQL | 专家置信度 / 优势加权 | ⚠️ 用 3 个手调 hard-scale 近似 |

### 2.2 结论

pingmo 已经站上了谱系里最关键的"闭环 DAgger"档（这比纯 BC 强得多，是 v4 比 v3 稳定的原因），
但**在监督信号的"强度"和"来源"上停在最保守档**：

- 信号**强度**：只用动作均值 MSE，没有用到 Teacher 的动作分布（SAC 有 `μ,σ`）、也没有用到 TwinQ 值。
- 信号**来源**：只用"匹配的逐机 Teacher 的动作"，没有用到 Teacher 的价值函数、也没有跨 `theta` 的集成共识。

这正是后文迭代路线的两条主线：**信号升级**（§4）和**来源升级**（§5，集成/共识）。

---

## 3. 三个失败模式的机理诊断

### 3.1 TV 比超标（v4：2.133）

v4 的增量损失是 own-delta 约定 `(u^S_t−u^S_{t−1}) ≈ (u^T_t−u^T_{t−1})`。它只约束"两边的增量一样"，
不约束"绝对动作一样"。于是一个在绝对动作上缓慢漂移的 Student 也能拿到很低的 own-delta 损失，
却产生远高于 Teacher 的 requested-force 总变差（TV）。

文献里这是**分布匹配的动机**：KL 目标是"在**每个状态**匹配 Teacher 的动作分布"，天然锚定绝对动作，
不会出现"增量对、绝对漂"的解。Rusu et al. 2016 在离散动作上直接比较了 NLL/MSE/KL，结论是
低温度 KL 更稳。连续控制下 MSE 仅在"相同协方差 + 单峰"时与 KL 等价（见 §4.2），而 pingmo 的
确定性 Student 匹配 SAC 的均值，恰好丢掉了 Teacher 的 `σ`。

**V6 的处理**（已实现）：把 Student 改成残差头 `u_t = clip(u_{t−1} + Δu_t)`，蒸馏标签改为
`label = u^T_t − u_{t−1}`（残差约定），再加"只罚超出 Teacher 局部变化率 + margin 的 excess"。
这从**参数化层面**消掉了"绝对漂移"自由度，是对 own-delta 的直接修正，不依赖额外调参。

### 3.2 doublet 峰值（v5 限速：7.98）

v5 用单一固定 slew limiter（11 N/s）把 TV 压到 0.574 倍，却在 doublet 快速反向处引入 7.98 deg/s 峰值。
机理：固定斜率上限在"稳态小误差需平滑"和"指令反转需大带宽"之间是全局一刀切，无法按状态分配带宽。

文献立场（DART / AggreVaTe）：正确的做法不是事后 clamp 输出，而是**在数据收集或目标里让 Teacher
示范"如何从误差恢复"**。V6 的 excess 惩罚正是"按 Teacher 的局部变化率给带宽"，而不是给一个常数上限。

### 3.3 未见飞机落后 PID（2.38 vs 1.41）

两个正交原因：
1. **跨 `theta` 局部插值差**：Student 对未见 `theta` 只能靠网络对 `θ` 的连续泛化。单一 Teacher 逐机标注
   没有给"邻近 `theta` 的共识"，插值处容易过拟合某架飞机的特异动作。文献给的是**集成蒸馏**
   （"Double Down on Distillation" 2025：集成 + 多样化数据能让学生泛化优于单 Teacher）。
2. **covariate shift**：Student 闭环访问的状态与 round 0 的 Teacher 分布不一致。DAgger 已缓解一部分，
   但 DART 的"专家动作注入优化噪声"是更省算力的 off-policy 补法，能把 round 0 的数据分布直接对齐到
   Student 的误差分布。

---

## 4. 方法论主线一：监督信号升级

把"Student 应该学什么"按强度排序（从弱到强），每一档对应具体可落地的改动：

### 4.0 动作均值（当前 baseline，v4）

```
L = weighted_teacher_action_mse(u^S, u^T)
```

最弱：把每个状态的动作当独立回归目标，且只匹配均值。

### 4.1 动作 + 增量（v4 已做、v6 已升级为残差）

```
L = action_mse + λ·delta_mse        （v4 own-delta → v6 残差 delta）
```

V6 的残差约定把"增量匹配"升级为"从 Student 自己的上一个动作出发，向 Teacher 目标移动"，
等价于在动作空间加了一个平滑/连续性先验。文献无 1:1 对应，但属于"reparameterization 让目标与归纳偏置对齐"。

### 4.2 动作分布（KL）—— 中等杠杆，需 Teacher 是随机策略

SAC Teacher 天然有 `μ(s), σ(s)`。连续控制下：

```
D_KL(π^S ‖ π^T) = 0.5·[ log(σ^S²/σ^T²) + (σ^S² + (μ^S−μ^T)²)/σ^T² − 1 ]   （对角高斯）
```

- 当 Teacher `σ→0`（确定性）时，KL 退化为 MSE；pingmo 当前"确定性 Student 匹配 SAC 均值"正是丢掉 `σ` 的退化情形。
- 文献一致结论（Rusu 2016；Sensors 2024 "Policy Compression"）：**若 Teacher 是随机策略，应同时蒸馏 `μ` 和 `σ`**，
  用 KL 或 `Huber(μ) + λ·Huber(σ)`，否则 Student 无法复现 Teacher 的探索/恢复行为，泛化和鲁棒性下降。

**落地**：给数据集增加 `teacher_action_std` 列（`SquashedGaussianActor.sample` 已能返回 `(action, log_prob/σ)`），
`IncrementalDenseStudent`/`DenseConditionalStudent` 增加一个 σ 输出头，损失加 KL 或 Huber(σ) 项。
优先级**中**：它主要影响随机/探索情形，而部署控制器是确定性执行的，对 TV 和 doublet 的直接收益小于 4.3。

### 4.3 价值 / 优势蒸馏（AggreVaTe 路线）—— 最高杠杆

这是本次综述最值得做的一项。AggreVaTe/AggreVaTeD（Ross & Bagnell 2014；Sun et al. 2017）把监督目标
从"模仿动作"换成"最小化专家 cost-to-go"：

```
π ← argmin_π  E_{s ~ d_π} [ Σ_a π(a|s) Q^*(s,a) ]      （动作维度：用 Q 值加权/引导）
```

pingmo **已经具备全部原料**，只是没接线：
- 每个 SAC Teacher checkpoint 里存了 `critic` / `target_critic`（`TwinQCritic`）的 state_dict、
  `critic_observation_dim`、`critic_observation_contract`（见 `trainer.py:724-725`）。
- 环境在 reset/step 时返回 `critic_state`（`trainer.py:578,628`），`build_specialist_env` 已按
  `critic_include_episode_progress` / `critic_include_command_context` 构造。
- 缺口只有一处：`load_specialist_actor` 返回的 `SpecialistActorPolicy` 只包 actor，不包 critic，
  无法查询 `Q(s_c, a)`。

**最小接线**（约一个文件量级）：
1. 新增 `SpecialistCriticPolicy`（或扩展 `load_specialist_actor`），加载 twin critic，暴露
   `q_values(critic_state, action)`。
2. 收集时（`collect_data.py` / `student_driven.py` 的 `_collect_student_driven_profile`）在记录
   `observation` 的同时记录 `critic_state`，并计算 `q^T = Q(s_c, u^T)`、`q^S = Q(s_c, u^S)`。
3. 新增数据集列 `teacher_advantage = q^T − q^S`（或直接存 `q^T`、`q^S`），在损失里作为
   **样本权重**（`weight × exp(advantage/τ)` 或 `weight × clip(1 + κ·advantage, ...)`）或
   额外正则项（`− λ·Q(s_c, u^S)`）。

**为什么它直接命中 pingmo 的三个失败模式**：
- **TV**：价值函数对"小幅高频修正"不敏感（Q 值只关心长期回报），但对"跟踪误差"敏感。用 Q 加权后，
  Student 会自动学"该平滑就平滑、该用力就用力"，而不是机械复制 Teacher 的逐步动作。
- **doublet**：Q 值在指令反转处对"动作来得太慢"给出明确梯度（AggreVaTe 用 Q 对动作的梯度引导），
  不会像固定 slew 一样一刀切。
- **未见飞机 / 超越 PID**：AggreVaTe 系列的核心理论结论是**当专家次优时，价值模仿能学出比专家更好的策略**。
  当前 Student 必须靠"复制 Teacher"来逼近，价值信号给了它"在 Teacher 之上优化"的自由度——这是赶超逐机 PID
  （当前 2.38 vs 1.41）最现实的一条路。

> 注意一个坑：AggreVaTeD 原文用 `Q^*(s,a)`（最优价值），而 pingmo 只有**逐机 SAC 的 `Q_π`（on-policy 价值）**，
> 不是全局最优 Q。这会使价值信号偏向 Teacher 自身的次优性。缓解办法：用 twin critic 的最小值（保守）估计、
> 或 `q^T − q^S` 相对优势而非绝对 Q（见 §6 的 P1.2）。

### 4.4 动作平滑蒸馏（ZAPS-DA 路线）—— 直接命中 TV/doublet 的新轴

本轮补充检索发现一条与 pingmo TV/doublet 问题**几乎同构**的文献线：连续控制 RL 里 SAC/TD3 的
feedforward critic 对连续动作做独立评估、缺乏时间一致性约束，导致策略输出高频抖动；直接给 actor 加
平滑惩罚会把平滑梯度与 RL reward 梯度耦合、破坏训练。解法是把"平滑"做成**蒸馏目标**：

- **ZAPS-DA（Zero-Phase Action Policy Smoothing with Decoupled Actor）**：保留原 actor 正常训 RL；
  另训一个**解耦 actor**，监督目标是 replay buffer 里经 **Savitzky–Golay 零相位滤波**后的动作
  `ã_t`（`L_imit = E‖tanh(μ_ψ(s)) − ã‖²`，再做幅度匹配）。部署的是这个解耦的前馈 actor。
  报告称转向抖动降 **14–21×**、任务成功率几乎不损（仅 6.3% reward 代价）。这是"把非因果滤波器
  因果蒸馏进一个前馈网络"——pingmo 的 incremental Student 本质上是同类思想的时间域近似（用
  `u_{t-1} + Δu` 的自回归结构内建平滑），ZAPS-DA 则把平滑目标显式化。
- **CAPS**：在 actor loss 直接加**时间项**（相邻输出差）和**空间项**（近邻状态敏感性），λ_T/λ_S 可调。
  与 pingmo 的 excess 惩罚同源，但 ZAPS-DA 批评它与 RL 目标共享参数、易失稳。
- **ASAP**：惩罚二阶差分压制高频振荡；**PAVE**：稳定 critic 的 action-gradient 场。

**对 pingmo 的直接启示**：V6 的增量参数化是"结构内建平滑"，但它的残差标签 `u^T_t − u_{t-1}` 仍直接
继承 Teacher 的逐步动作，若 Teacher 本身抖动，Student 仍会学进去（见本文实验 regime 2）。一个
更彻底的 P0.5 是把 **ZAPS-DA 的滤波目标**接到现有残差标签上：对 Teacher 动作序列做零相位/低通滤波后
再算残差标签，让 Student 直接学"平滑后的 Teacher"而不是"抖动的 Teacher"。这与 P0 完全正交、成本低，
且对 doublet（高频峰值）尤其对症。

---

## 5. 方法论主线二：监督来源升级（集成 / 共识 / 分布对齐）

### 5.1 集成共识监督（针对未见飞机插值）

当前：每个状态只由"匹配的逐机 Teacher"标注 → 单点监督，`theta` 插值处易过拟合。

文献：BRED（NeurIPS offRL）、EPD/DPD（AAAI 2020）、"Double Down on Distillation"（2025）都显示
**集成共识（多 Teacher 平均/加权）降低方差、提升泛化**。pingmo 的 32 个逐机 Teacher 天然构成一个
"覆盖 `theta` 的专家集"。

**落地（中等成本）**：收集时对每个状态，用"`θ` 距离最近的 K 个 Teacher"做软加权平均作为标签：

```
u_label(s) = Σ_k w_k(θ, θ_k) · u_k(s) / Σ_k w_k     ,   w_k = softmax(−‖θ−θ_k‖²/τ_θ)
```

等价于一个非参数的"跨 `theta` 集成蒸馏"，直接作用在插值风险上。可选地，用集成的**方差**作为
不确定度权重（方差大 = 插值不确定 → 降权或额外采样）。

### 5.2 DART 噪声注入（针对 covariate shift，off-policy 省钱）

DART（Laskey et al. 2017）的核心：在**专家收集阶段**给 Teacher 动作注入优化过的噪声，让演示覆盖
"从误差恢复"的轨迹，从而把 round 0 的数据分布对齐到 Student 的误差分布——**无需额外的 Student 闭环轮次**。

落地：round 0 收集时 `u_demo = clip(u^T + ε, −1, 1)`，`ε ~ N(0, σ_opt)`，`σ_opt` 用
"Student 当前动作误差的估计"去匹配（DART 的关键：噪声尺度要对齐学习器误差，不能乱加）。用
`driver_actions` 记录加了噪声的动作（语义上仍是"Teacher 带扰动的演示"）。文献明确警告**朴素各向同性大噪声会失败**，
必须调 `σ_opt`。

### 5.3 DAgger 数据聚合：从"均匀累计"到"轮次加权 / balanced replay"

当前 `all_shards = prior_shards + new_shards`（`student_driven.py:524`）是标准 DAgger 均匀累计，
但对"早期 Teacher 强制数据"与"近期 Student 驱动数据"一视同仁。文献（BRED balanced replay）建议：
- **近期轮次上采样**：`weight_round(r) = decay^(R−r)`，让最新 on-policy 数据主导；
- 或按 `driver` 平衡 Teacher/Student 数据比例。

**落地（低成本）**：在 `DistillationDataset` 加一个 `round_weight` 列，`sample_weight` 乘上它。
这是最便宜、最先值得做的数据层实验之一。

---

## 6. 优先级迭代路线图

按"预期 KPI 收益 / 实现成本"排序。V6（P0）已实现，P1 是下一个最高杠杆。

| 优先级 | 名称 | 文献依据 | 落地文件 | 预期效果 | 成本 |
| --- | --- | --- | --- | --- | --- |
| **P0** | 残差增量动作头 | 本文 §4.1（参数化修正 own-delta） | `network.py` `losses.py` `dataset.py` `distill.py` | TV 比 ≤1.25 且不牺牲 doublet | 已实现，待 GPU 验证 |
| **P0.5** | 滤波动作目标蒸馏 | ZAPS-DA、CAPS、ASAP（§4.4） | `dataset.py`（残差标签源） | doublet/TV 进一步下压，尤其抖动 Teacher | 低（对标签序列滤波，正交于 P0） |
| **P1** | 价值/优势加权蒸馏 | AggreVaTe/AggreVaTeD、CIQL、DOR-PDAWAC | `trainer.py`（暴露 critic）、`collect_data.py`、`dataset.py`、`losses.py` | 赶超 PID；TV/doublet 双改善 | 中（接线 critic + 数据列 + 损失项） |
| **P2** | DART 噪声注入 round 0 | Laskey et al. 2017 | `collect_data.py` | 未见飞机 covariate shift 缓解 | 低（改收集逻辑 + 调 σ_opt） |
| **P3** | 轮次加权 / balanced replay | BRED | `dataset.py` `student_driven.py` | 数据分布对齐、加速收敛 | 低 |
| **P4** | `theta` 邻居集成共识监督 | 集成蒸馏（2025） | `student_driven.py` 收集段 | 未见飞机插值风险降低 | 中（需多 Teacher 前向） |
| **P5** | KL / μ+σ 分布蒸馏 | Rusu 2016、Sensors 2024 | `network.py`（σ 头）、`losses.py` | 泛化/鲁棒性 | 中（收益偏随机情形） |
| **P6** | 用 hard-scale 换置信度/优势权重 | 2IWIL、CAIL | `dataset.py`（权重构造） | 去掉 3 个手调尺度 | 低（P1 的子集，可合并） |

### 6.1 推荐执行顺序（一个可归因的增量序列）

1. **先跑 P0（V6）**：`--student-architecture incremental --initial-sample-stride 1 --student-sample-stride 1`。
   只验证一个假设：残差约定能否把 TV 比压到 ≤1.25 而不引入 doublet 峰值退化。
2. **P0.5（滤波目标）**：若 P0 跑通后 doublet/TV 仍超标（尤其 Teacher 抖动明显），对残差标签源做
   零相位/低通滤波，与 P0 正交、成本低。
3. **P3（轮次加权）**：最便宜的数据层改动，独立可测，和 P0 正交。
4. **P1（价值加权）**：最高杠杆，但改动最大，单独一轮，和 P0 消融对比。
5. **P2（DART）**、**P4（集成共识）**：按 P1 结果决定顺序——若未见飞机仍是主要短板先 P4，
   若 covariate shift 明显先 P2。P4 可参考多 Teacher 梯度匹配（Yang & Zhang 2024）。
6. **P5（分布蒸馏）**：留到确定性基线稳定后，作为鲁棒性增强，不阻塞主线。

每步都必须在**同一 6 架 holdout + 同一 10 架零样本集合**上复测（见 §7），并与 v4 基线逐项对比，
禁止"多改一轮只看最终值"。

### 6.2 P1 的两种落地强度

- **P1.1 样本权重版**（已实现）：`weight ← weight × exp(κ·clip(advantage, −c, c))`，其中
  `advantage = Q_min(s, u^T) − Q_min(s, u^S)`。只改权重，风险最低。实现采用**指数折入**而非早先
  草稿的 `1 + κ·clip(...)` 线性式，因为指数式对任意 advantage 都保持正权重，不需要对负 advantage
  做逐行 clamp 到正数，且与 CIQL 的 `exp(advantage/τ)` 温度形式一致。
- **P1.2 Q-正则版（AggreVaTe 更彻底）**：损失加 `−λ·Q(s_c, u^S)`（鼓励 Student 动作抬升 Teacher Q 值），
  配合 twin-critic 最小值做保守估计。收益更大，但需要仔细调 `λ` 防止 Student 钻 Q 函数空子
  （off-policy Q 的外推区不可信，务必用 twin-min 和 clip）。P1.2 尚未实现，留作 P1.1 验证后的升级。

---

## 7. 评估协议与严谨性

文献与 v5 报告共同指向三个方法论层面的硬约束：

1. **预注册测试集**：现有 10 架"未见飞机"已被多轮对比使用，不再是盲测集（v5 报告已明确指出）。
   下一次声称"零样本泛化"前，须在方法与阈值冻结后，从飞机库**预注册一组全新的 untouched test aircraft**。
2. **固定 holdout 与公平对照**：6 架整机 holdout 与逐机 PID 的评测窗口（PID 用 5 s 调参、30 s 测试）
   必须保持一致并写明 scope。任何新方法不得根据 holdout 结果回头改选阈值。
3. **单变量归因**：每轮只改一个维度（P0→P3→P1→…），消融对比；用 `round_metrics.csv` 里的
   `action RMSE / delta RMSE / TV 比 / 峰值 / 改善率 / 伤害率` 六项做归因，不只看一个标量。

---

## 8. 参考文献

1. Rusu A. A., Colmenarejo S. G., Gulcehre C., et al. *Policy Distillation*. arXiv:1511.06295 (ICLR 2016). https://arxiv.org/abs/1511.06295
2. Ross S., Gordon G. J., Bagnell J. A. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS 2011.
3. Laskey M., Lee J., Fox R., Dragan A., Goldberg K. *DART: Noise Injection for Robust Imitation Learning*. CoRL 2017, PMLR 78. https://mlanthology.org/corl/2017/laskey2017corl-dart/
4. Ross S., Bagnell J. A. *Reinforcement and Imitation Learning via Interactive No-Regret Learning*. arXiv:1406.5979 (2014).
5. Sun W., Venkatraman A., Gordon G. J., Boots B., Bagnell J. A. *Deeply AggreVaTeD: Differentiable Imitation Learning for Sequential Prediction*. ICML 2017. https://mlanthology.org/icml/2017/sun2017icml-deeply/
6. *Ensemble Policy Distillation in Deep Reinforcement Learning*. AAAI-20 Workshop on RL in Games. https://rlg.mlanctot.info/2020/papers/AAAI20-RLG_paper_34.pdf
7. *BRED: Ensembled offline RL policies distilled for fine-tuning / balanced replay*. NeurIPS Offline RL Workshop. https://offline-rl-neurips.github.io/pdf/13.pdf
8. Wu Y.-H., Charoenphakdee N., Bao H., Tangkaratt V., Sugiyama M. *Imitation Learning from Imperfect Demonstration*. ICML 2019 (2IWIL). arXiv:1901.09387
9. Zhang S., Cao Z., Sadigh D., Sui Y. *Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality*. NeurIPS 2021 (CAIL).
10. Bu X., et al. *Learning from Imperfect Demonstrations through Dynamics Evaluation* (CIQL). arXiv:2312.11194
11. *Policy Compression for Intelligent Continuous Control on Low-Power Edge Devices* (KL / μ+σ distillation). Sensors 2024, 24(15):4876. https://www.mdpi.com/1424-8220/24/15/4876
12. *How Ensembles of Distilled Policies Improve Generalisation in RL* ("Double Down on Distillation"). arXiv:2505.16581. https://arxiv.org/abs/2505.16581
13. Rajeswaran A., et al. *Proximal Policy Distillation*. arXiv:2407.15134 (2024).
14. *ZAPS-DA: Zero-Phase Action Policy Smoothing with Decoupled Actor for Continuous Control in RL*. arXiv:2605.30612.
15. Mysore S., Mabsout B., Mancuso R., Saenko K. *CAPS: Robust Continuous Control with Action and Sensitivity Policies*. arXiv:2103.08214.
16. Yang J., Zhang J. *A Multi-Teacher Policy Distillation Framework for Enhancing Zero-Shot Generalization of Autonomous Driving Policies*. IEEE TVT 73(7):9734–9746, 2024. DOI:10.1109/TVT.2024.3379972.
17. Hoque R., Balakrishna A., et al. *ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning*. arXiv:2109.08273.
18. *Distill Knowledge in Multi-task Reinforcement Learning with Optimal-Transport Regularization*. arXiv:2309.15603.
19. Yang H., Liu Q. *Advantage Weighted Double Actors-Critics Algorithm Based on Key-Minor Architecture for Policy Distillation (DOR-PDAWAC)*. 计算机科学, 2024(11). https://www.jsjkx.com/CN/abstract/abstract22759.shtml
20. Peng X. B., Kumar A., Zhang G., Levine S. *Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning*. arXiv:1910.00177.
21. Nair A., Gupta A., Dalal M., Levine S. *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*. arXiv:2006.09359.

---

## 9. 本地消融实验（gymnasium-free，CPU 可复现）

为把 §4–§6 的四个核心假设真正测到，在 `experiments/distillation_ablation.py` 里 1:1 复刻生产损失装配，
用合成 smooth-odd Teacher + 生产 `DistillationDataset`/`losses`/student 网络跑三个 regime。
结果 JSON 在 `results/ablation/distillation_ablation_{smooth,jitter,capacity}.json`。

### 9.1 Regime 1：平滑 Teacher（易拟合，检验表征 + 对称性）

| 配置 | forced MSE | deployed MSE | TV 比 | max|Δu| |
| --- | --- | --- | --- | --- |
| v4 dense | 2e-6 | 2e-6 | **0.998** | 0.033 |
| v6 incremental | 1.1e-5 | 1.5e-5 | **0.876** | 0.028 |
| v6 odd 关 | 5e-5 | 8.2e-5 | 0.886 | 0.030 |
| v6 hardcase | 1e-5 | 1.5e-5 | 0.880 | 0.029 |
| v6 advantage 0.5/1.0 | 1.1e-5 | 1.5e-5 | 0.882/0.886 | 0.028 |

- **H3（奇对称）确认且强**：关掉 odd-policy 后 MSE 恶化 **~5×**（1.1e-5→5e-5），在 odd 真值上对称约束是免费的精度来源。
- **H1 部分**：incremental 是**平滑器**——TV 比 0.876 < 1.0（低于 Teacher 自身），dense 则 0.998 ≈ 1.0（忠实复制）。
- **H2/H4 在此 regime 无信号**：因为 Student 已近乎完美拟合（MSE~1e-5），权重没有发挥空间。

### 9.2 Regime 2：抖动 Teacher（复现 TV 爆炸）

对 Teacher 动作注入高频抖动 `0.15·sin(6t)`，制造 SAC 式抖动：

| 配置 | forced MSE | deployed MSE | TV 比 | max|Δu| |
| --- | --- | --- | --- | --- |
| dense | 0.026 | 0.026 | **1.691** | 0.266 |
| inc dw=5/1/0.1/0 | 0.0016 | 0.061–0.067 | **0.670–0.675** | 0.083–0.091 |

- **H1 确认**：dense 忠实复制并**放大**抖动（TV 比 1.69，与生产 2.133 同构）；incremental 把 TV 比压到 **0.67**、
  doublet 代理 max|Δu| 从 0.266 降到 0.083（**3.2×**）。incremental 的增量结构天然低通，无法跟踪高频抖动。
- **新发现（教师强制 vs 自回归鸿沟）**：incremental forced MSE 0.0016 vs deployed MSE 0.065（**40× 鸿沟**）。
  这正是 round-0 教师强制 + 残差标签 `u^T_t−u^T_{t-1}` 的固有分布偏移——训练时喂的是 Teacher 上一步动作，
  部署时喂的是 Student 自己上一步动作，误差逐歩复利。**学生驱动 DAgger 回合 + 残差重标（P1/§6）不是可选项，是必选项**。

### 9.3 Regime 3：欠参数 Student（给加权留 headroom）

width=16、1 个残差块，Teacher 轻抖动：

| 配置 | deployed MSE | TV 比 |
| --- | --- | --- |
| tiny dense | 0.00185 | 0.157 |
| tiny inc | 0.00250 | 0.152 |
| tiny inc adv 0.5 | 0.00255 | 0.164 |
| tiny inc adv 2.0 | 0.00265 | **0.190** |
| tiny inc hardcase | 0.00249 | 0.156 |

- **H2 的警示性负结果**：用 `|Δu^T|`（变化率）作 advantage 代理时，κ 越大 deployed MSE 越差、TV 比越高
  （0.152→0.190）。原因：变化率代理把"动得快"当作"难"，而这里"动得快"正是要压制的抖动 → 加权=放大噪声。
  **结论：加权信号必须是"价值/重要性"（critic 的 Q(u^T)−Q(u^S)），不能是"变化率"**——这正是 P1 坚持接 critic
  而非复用 hard-scale 的理由。该负结果**只证伪"变化率=难度"这类代理**（对应 `hard_action_rate_scale`），
  与 P6 用 2IWIL/CAIL 的置信度加权（基于 demo 最优性）是两回事；P6 仍值得做，但信号源必须是"最优性/价值"
  而非"动作幅度或变化率"。
- **H4 中性**：hard-case 权重（|tracking error|）在此合成 setup 里与"真正难"不重合，无显著作用。

### 9.4 实验结论（一句话版）

> 增量表征是 TV/doublet 的结构性平滑器（1.69→0.67）；奇对称是免费精度；优势加权只有接**价值信号**才有意义，
> 变化率/误差幅度这类"难度代理"会放大抖动；教师强制→自回归的鸿沟（40×）是必须靠学生驱动回合关闭的核心缺口。

---

## 附：一句话方法论

> **用 DAgger 保证数据"在 Student 自己的分布上"，用残差参数化保证动作"平滑"，
> 用 Teacher 价值函数保证监督信号是"什么是好"而不是"Teacher 做了什么"，
> 用跨 `theta` 集成共识保证泛化不再依赖单点插值。**

把信号从"动作"升级到"价值"（P1）是当前投入产出比最高的下一步；V6（P0）是它的前提，必须先跑通验证残差约定本身不退化。
