# pingmo 蒸馏方法论 · 汇总与逻辑图

> 一份把「诊断 → 文献 → 实现 → 实验 → 结论」串成思维链的浓缩索引。
> 详细论证见 `distillation_methodology_report.md`，实验代码 `experiments/distillation_ablation.py`，
> 结果 `results/ablation/*.json`。

---

## 0. 一句话定位

> **用 DAgger 保证数据在 Student 自己的分布上，用残差增量保证动作平滑，用 Teacher 价值函数保证监督信号是
> 「什么是好」而非「Teacher 做了什么」，用跨 θ 集成共识保证泛化不依赖单点插值。**

---

## 1. 思维链总览（顶层逻辑图）

```mermaid
flowchart TD
    O["观测：3 个失败模式<br/>TV 比 2.133 · doublet 7.98 · unseen 2.38 vs PID 1.41"]
    O --> R["根因诊断（逐模式归因）"]
    R --> R1["根因① 信号约定错<br/>v4 own-delta 损失"]
    R --> R2["根因② 信号维度低<br/>只有动作均值，无价值"]
    R --> R3["根因③ 信号来源窄<br/>单一 Teacher + 插值泛化差"]
    R1 --> M1["P0 残差增量<br/>P0.5 滤波目标"]
    R2 --> M2["P1 价值/优势加权"]
    R3 --> M3["P4 集成共识<br/>P2 DART"]
    M1 --> V["实验验证（3 regime / 16 配置）"]
    M2 --> V
    M3 --> V
    V --> C["结论：增量是结构性平滑器<br/>奇对称是免费精度<br/>加权必须接价值信号<br/>40× 强制→自回归鸿沟是核心缺口"]
```

---

## 2. 失败模式 → 根因 → 对策（映射逻辑图）

```mermaid
flowchart LR
    subgraph 失败["三个失败模式"]
        F1["TV 比 2.133（抖）"]
        F2["doublet 峰值 7.98"]
        F3["unseen 2.38 &gt; PID 1.41"]
    end
    subgraph 根因["根因"]
        G1["own-delta 约定<br/>(uS−uS₋₁)≈(uT−uT₋₁)<br/>反馈复利放大"]
        G2["监督只有动作均值<br/>无「好坏」信号"]
        G3["Teacher 抖动 + 单点插值"]
    end
    subgraph 对策["对策（P 编号）"]
        H1["P0 残差增量<br/>P0.5 滤波目标"]
        H2["P1 价值加权<br/>（接 critic）"]
        H3["P4 集成共识<br/>P2 DART"]
    end
    F1 --> G1 --> H1
    F2 --> G2 --> H2
    F2 -.-> H1
    F3 --> G3 --> H3
    F3 -.-> H2
```

---

## 3. 方法论阶梯（ladder）与文献锚点

```mermaid
flowchart TD
    L0["基线：动作均值 BC"] --> L1["P0 动作 + 增量（残差参数化）<br/>修正 own-delta 约定"]
    L1 --> L05["P0.5 滤波动作目标蒸馏<br/>ZAPS-DA · CAPS · ASAP"]
    L1 --> L2["P1 价值 / 优势蒸馏<br/>AggreVaTe · CIQL · DOR-PDAWAC"]
    L2 --> L3["P2 DART 噪声注入<br/>抗 covariate shift"]
    L2 --> L4["P4 集成 / 共识监督<br/>多 Teacher 梯度匹配"]
    L3 --> L5["P5 分布蒸馏（KL / μ+σ）<br/>鲁棒性增强"]
    L4 --> L5
    L05 -.-> L5
```

> 阶梯的递进逻辑：**信号维度**（动作→增量→分布→价值）与**信号来源**（单 Teacher→集成）两条主线正交，
> 可分别消融、组合推进。

---

## 4. 关键设计决策链（为什么这么做）

```mermaid
flowchart TD
    D0["决策点：动作表征用全量还是增量？"] -->|全量| D1["dense：忠实复制 Teacher，含抖动"]
    D0 -->|增量| D2["incremental：u_t=clip(u₋₁+Δu)<br/>结构性低通，压 TV/doublet"]
    D2 --> D3["决策点：残差标签用什么？"]
    D3 -->|"own-delta uT−uT₋₁"| D4["round 0 教师强制<br/>→ 40× 自回归鸿沟"]
    D3 -->|"residual uT−uS₋₁"| D5["学生驱动回合重标<br/>→ 关闭分布偏移"]
    D4 --> D6["结论：学生驱动 DAgger 是必选项"]
    D5 --> D6
    D2 --> D7["决策点：加权信号用什么？"]
    D7 -->|"变化率 |ΔuT|"| D8["放大抖动 ✗ 证伪"]
    D7 -->|"价值 Q(uT)−Q(uS)"| D9["聚焦「真正难」✓ P1"]
```

---

## 5. 实验逻辑与结论

```mermaid
flowchart TD
    E["3 个 regime"] --> R1["Regime1 平滑 Teacher<br/>（易拟合）"]
    E --> R2["Regime2 抖动 Teacher<br/>（复现 TV 爆炸）"]
    E --> R3["Regime3 欠参数 Student<br/>（给加权留 headroom）"]
    R1 --> K1["H3 奇对称 ✓ ~5×<br/>H1 增量更平滑"]
    R2 --> K2["H1 ✓ TV 1.69→0.67<br/>doublet 3.2×<br/>新发现 40× 鸿沟"]
    R3 --> K3["H2 有条件：<br/>价值信号 ✓ / 变化率 ✗"]
```

| 假设 | 判定 | 关键数字 |
| --- | --- | --- |
| H1 增量表征降 TV | ✅ 确认 | TV 比 1.69 → 0.67；max\|Δu\| 0.266 → 0.083（3.2×） |
| H3 奇对称 | ✅ 确认且强 | 关掉后 MSE 恶化 ~5× |
| H2 优势加权 | ⚠️ 有条件 | 价值信号有效；变化率代理 TV 0.152→0.190（有害） |
| H4 hard-case | ⚪ 中性 | 合成 setup 里无效 |

---

## 6. 路线图（依赖关系）

```mermaid
flowchart TD
    P0["P0 残差增量（已实现，待 GPU 验证）"] --> P05["P0.5 滤波目标<br/>（TV/doublet 仍超标时）"]
    P0 --> P3["P3 轮次加权（最便宜，正交）"]
    P0 --> P1["P1 价值加权（已接线，最高杠杆）"]
    P1 --> P24["P2 DART / P4 集成共识<br/>按 P1 结果定序"]
    P24 --> P5["P5 分布蒸馏（不阻塞主线）"]
```

**推荐顺序**：P0 先跑通 → 视结果插 P0.5 → P3 → P1（单变量消融对比 P0）→ P2/P4 → P5。

---

## 7. 文件与记忆索引

| 产物 | 位置 |
| --- | --- |
| 详细报告（§1–§9） | `docs/distillation_methodology_report.md` |
| 本汇总 | `docs/distillation_methodology_summary.md` |
| 实验脚本 | `experiments/distillation_ablation.py` |
| 实验结果 | `results/ablation/distillation_ablation_{smooth,jitter,capacity}.json` |
| P1 实现 | `src/distillation/critic.py`（新增）+ `dataset.py`/`distill.py`/`student_driven.py`/`scripts/34_*` |
| 冒烟测试 | `tests/test_v6_incremental_smoke.py` · `tests/test_v7_advantage_weighting_smoke.py` |
