# 一个控制器，控制一类飞机

## Overview

- **Topic**: 将 32 个逐机纯奖励 TD3 Teacher 蒸馏为一个读取飞机参数的条件 Student
- **Hook**: 同一个滚转角速度指令，为什么不同飞机会给出完全不同的响应？
- **Misconception**: 通用控制不是让所有飞机使用同一个动作，而是让它们使用各自合适的动作，得到同一种目标响应
- **Aha moment**: Student 必须在自己闭环访问到的状态上向 Teacher 学习；离线 action loss 最小，不等于闭环控制最好
- **Target audience**: 了解基本反馈控制或机器学习，但不需要知道本仓库代码细节的工程人员
- **Length**: 约 2 分 10 秒
- **Resolution**: 854x480 草稿，1920x1080 成片，16:9
- **Audio**: 无外部 TTS；画面内中文字幕，并生成 Manim subcaption 轨道

## Color Palette

- Background: `#111318` - 深色教学背景
- Reference: `#58C4DD` - 目标模型、目标响应、逐机 PID 基准
- Student: `#83C167` - 通用 Student 与其闭环响应
- Teacher: `#FFB347` - 逐机 Teacher 与专家标签
- Command: `#FFFF00` - 指令与关键洞察
- Danger: `#FF6B6B` - 原始失配、发散、失败门禁
- Context: `#A7B0BE` - 历史版本、坐标轴和辅助信息

颜色绑定概念而不是图形。全片保持同一语义。

## Arc: Problem-Solution

1. Problem - 同一条指令作用于不同固有品质的飞机，响应差异巨大
2. Goal - 用参考模型定义唯一的目标响应，而不是规定唯一动作
3. Solution - 逐机 Teacher Bank 蒸馏为条件 Student
4. Key insight - 用 Student-driven / DAgger 收集闭环状态并重新标注
5. Result - 未见飞机上显著优于 Raw，v3 明显优于 v2
6. Boundary - 仍落后逐机 PID，峰值与动作平滑性门禁失败

## Scene 1: 同一个指令，三种答案 (~12s)

**Purpose**: 让观众先看到对象差异，而不是先看网络结构。

**Layout**: TOP_BOTTOM，顶部问题，中央三架飞机，底部三个小型响应图。

### Visual elements

- 黄色阶跃指令 `p_c = 20 deg/s`
- 三架参数不同的飞机轮廓
- 三条原始响应：慢、振荡、强烈超调
- 低透明度坐标轴

### Animation sequence

1. 写出问题，停顿 1.5 秒
2. 同一个黄色指令分叉到三架飞机
3. 飞机变形成三个响应图，三条红色曲线依次长出
4. 强调三条曲线终点与形状不同

### Visible subtitle

“控制器面对的不是一架飞机，而是一族 G(theta)。”

## Scene 2: 统一响应，不是统一动作 (~15s)

**Purpose**: 建立仓库真正解决的控制问题。

**Layout**: LEFT_RIGHT，左侧目标响应，右侧三架飞机与不同大小的控制动作。

### Visual elements

- 二阶参考模型，`omega_n = 2.0 rad/s`, `zeta = 0.7`
- 蓝色目标响应 `p_ref`
- 三个不同长度的绿色 `F_as` 箭头
- 三条逐渐贴合目标的绿色响应
- 最后出现公式 `pi(o, theta) -> F_as`, `p ~= p_ref`

### Animation sequence

1. 指令进入参考模型，先画出蓝色目标曲线
2. 三架飞机分别获得不同动作
3. 三条绿色响应变换到同一蓝色目标附近
4. 公式最后出现，作为几何直觉的总结

### Visible subtitle

“通用控制的目标：动作可以不同，飞行品质要一致。”

## Scene 3: 32 个专家，压缩成一个 Student (~15s)

**Purpose**: 解释 Teacher Bank 和条件 Student 的角色分工。

**Layout**: BUILD_UP / ANNOTATED_DIAGRAM，从参数空间扩展到完整蒸馏图。

### Visual elements

- 32 个橙色 Teacher 点，分布在低透明度参数平面
- 每个 Teacher 只对应固定 `theta_i`
- 中央绿色 Student 模块
- 输入：35 维 observation + 8 维 `theta`
- 输出：完整 requested `F_as`
- 参数计数：540,417

### Animation sequence

1. 参数平面内逐批出现 Teacher 点
2. 32 个专家输出收束到 Student
3. Student 输入从 `o` 扩展为 `[o, theta]`
4. 以 `32 -> 1` 和参数计数完成压缩揭示

### Visible subtitle

“每个 Teacher 精通一架飞机；Student 学会根据 theta 切换控制规律。”

## Scene 4: 为什么要让 Student 自己飞 (~17s)

**Purpose**: 用状态分布漂移解释 Student-driven / DAgger，而不是只展示流程框。

**Layout**: LEFT_RIGHT，左侧状态平面，右侧逐轮数据与闭环指标。

### Visual elements

- 橙色 Teacher 轨迹及其训练样本
- 绿色 Student 轨迹从 Teacher 轨迹附近漂出
- 漂出区域内新增橙色 Teacher 标签
- 数据规模 `264k -> 528k -> 792k`
- round 1 与 round 2 的闭环对比

### Animation sequence

1. 只在 Teacher 轨迹上采样，Student 随后漂出已见区域
2. Student 驱动访问真实闭环状态，Teacher 在这些状态重新标注
3. 新数据把 Student 轨迹拉回目标走廊
4. 比较 round 1 和 round 2：round 2 离线差距更小，但闭环 RMSE 和动作 TV 更差
5. 高亮最终选择 round 1

### Visible subtitle

“闭环最好，不等于离线 action loss 最小。”

## Scene 5: 完全未见飞机上的结果 (~14s)

**Purpose**: 用仓库正式结果回答“它到底有没有用”。

**Layout**: DATA STORY，先看 Raw 到 v3 的量级变化，再放大受控方法。

### Visual elements

- 冻结 Student，10 架未见飞机，60 个飞机-命令对
- 平均 RMSE：Raw 48.377，v2 4.442，v3 2.411，逐机 PID 1.409 deg/s
- `58/60 = 96.67%` 相对 Raw 改善
- `51/60 = 85%` 相对 v2 改善

### Animation sequence

1. 红色 Raw 高柱出现
2. 绿色 v3 柱出现，标注约 20 倍更低的平均 RMSE
3. 放大受控方法，依次比较 v2、v3 和逐机 PID
4. 两个比例计数器从 0 动画到 96.67% 和 85%

### Visible subtitle

“一个冻结的 Student，已经能在绝大多数未见测试对上改善原始响应。”

## Scene 6: 有效，但还没有过线 (~10s)

**Purpose**: 保留项目当前结论的工程边界，不把实验包装成已解决问题。

**Layout**: TOP_BOTTOM，顶部总结，中部两道失败门禁，底部下一步方向。

### Visual elements

- 绿色结论：一个控制器开始覆盖一类飞机
- 红色门禁 1：最大峰值误差 `22.906 > 5 deg/s`
- 红色门禁 2：Student / Teacher TV `1.816 > 1.25`
- `quality_gate_failed`
- 下一步：高峰值状态 + 动作平滑性

### Animation sequence

1. 先给出绿色能力总结
2. 两道门禁依次出现并越过阈值线
3. 将最终结论收束为“有效，不等于合格”
4. 以项目名和研究问题淡出

### Visible subtitle

“它证明了路线有效，也精确指出了下一步该修什么。”

## Data Sources

- `docs/current_code_and_results_20260830.md`
- `results/pure_reward_teacher_bank_coverage_v3/student_driven_dense_balanced_holdout/round_metrics.csv`
- `results/pure_reward_teacher_bank_coverage_v3/student_driven_dense_balanced_holdout/pipeline_report.json`
- `results/pure_reward_teacher_bank_coverage_v3/unseen_aircraft_v3/report.json`
- `results/pure_reward_teacher_bank_coverage_v3/unseen_comparison_v3/comparison.json`

所有指标直接固化自上述仓库快照。视频不声称 Student 已优于逐机 PID，也不声称满足完整 GJB 横航向飞行品质要求。
