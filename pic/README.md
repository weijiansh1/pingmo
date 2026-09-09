# `pic.zip` 中文图表说明书

生成日期：2026-09-02

本压缩包收录最近一轮 P 通道强化学习飞控实验的 25 张核心图表，覆盖 Teacher Bank、
Student-driven 蒸馏、整机 holdout、未见飞机泛化，以及固定动作限速诊断。所有图片都在
`images/` 中，并在本文件第 10 节逐张显示。

查看方法：先完整解压 `pic.zip`，保持 `README.md` 和 `images/` 的相对位置不变，再使用 VS Code
Markdown Preview、Typora、Obsidian 或其他 Markdown 预览器打开 `README.md`。普通文本编辑器只会
显示 Markdown 源码；直接在 ZIP 预览窗口中打开也可能无法加载关联图片。压缩包同时提供
`README.html`，不具备 Markdown 预览器时可直接用浏览器打开。

## 1. 先看结论

1. v3 Student 已能在大多数未见飞机上明显改善 Raw 响应，但 `train_boundary-1448` 的
   `+25 deg/s` 长时阶跃会进入持续振荡，峰值与控制动作平滑性门禁失败。
2. v4 增加时序合法的动作增量匹配、困难样本加权和稳定性优先选型后，修复了 v3 最严重的
   长时振荡。六架 holdout 上的跟踪、峰值和绝对 TV 门禁通过，只剩 Student/Teacher TV 比失败。
3. v5 没有训练新网络，只给冻结的 v4 Student 外包 `11 N/s` 固定动作限速器。TV 显著下降，
   但 doublet 快速反向时控制力来不及换向，最大峰值重新超限。因此 v5 是诊断反例，不是新的
   最优控制器。
4. 在十架未见飞机上，v3/v4/v5 都显著优于 Raw，但总体仍落后于逐机专用 PID。当前可以证明
   “统一 Student 具有零样本控制能力”，还不能声称“统一 Student 已全面超过 PID”。

当前推荐基线是 **v4 Dense Student**。整个 v3、v4、v5 流水线的正式状态仍是
`quality_gate_failed`。

## 2. 图例和术语

| 图中名称 | 中文含义 | 判断方法 |
| --- | --- | --- |
| `p_c` | 滚转角速度指令 | 表示最终要求，不要求飞机瞬时跳到该值 |
| `p_ref` | 参考模型输出 | 表示考虑期望动态和时延后的理想响应 |
| `Raw` | 无学习控制器的飞机响应 | 用于判断控制器是否真正改善原始飞机 |
| `PID` | 逐机调参的专用 PID | 是当前传统控制基线，不是统一控制器 |
| `Teacher` | 一架飞机对应一个纯奖励 TD3 Teacher | 给 Student 提供该飞机上的专用控制规律 |
| `Student` | 读取 observation 和八维 `theta` 的统一策略 | 目标是蒸馏多个 Teacher 并泛化到未见飞机 |
| `requested force` | 策略请求的 `F_as` | 用于分析策略本身是否抖动 |
| `applied force` | 经过速率限制和时延后进入飞机的力 | 与 requested 的差异反映执行环节影响 |
| `core` | 参数空间内部的常规飞机 | 通常更接近训练分布中心 |
| `boundary` | 参数空间边界飞机 | 更容易暴露稳定性和外插问题 |
| `holdout` | 蒸馏训练时整架排除的验证飞机 | 用于每轮闭环选型和质量门禁 |
| `unseen aircraft` | 未参与 Teacher Bank、蒸馏或适配的飞机 | 用于统一 Student 的零样本泛化评测 |

不同图的具体颜色可能不同，应以图内 legend 为准。常见时域图中，横轴是时间 `s`，上半图是
滚转角速度 `p (deg/s)`，下半图或右侧图是控制力 `F_as (N)`。

## 3. 核心指标怎么读

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| Tracking RMSE | `p` 与 `p_ref` 的均方根误差 | 越低越好 |
| Maximum peak error | 整段响应中的最大绝对跟踪误差 | 越低越好；当前 holdout 门限为 `5 deg/s` |
| Requested-force TV | `sum(abs(u[t] - u[t-1]))` | 越低越平滑，但不能以牺牲必要瞬态动作为代价 |
| Student / Teacher TV | Student TV 与对应 Teacher TV 的比值 | 越接近 1 越像 Teacher；当前门限为 `<= 1.25` |
| Nearest-aircraft distance | 归一化八维 `theta` 空间中的最近训练对象距离 | 越小通常表示插值更容易，但它不是性能保证 |
| Student / PID RMSE | Student RMSE 除以专用 PID RMSE | `< 1` 表示 Student 胜过 PID，`> 1` 表示落后 |

TV 与仿真时长、策略频率和指令集合有关，只能在相同飞机、相同命令、相同窗口和相同执行约束下
公平比较。RMSE 小也不代表控制器一定可用，还必须同时检查峰值、长时稳定性和控制动作。

## 4. 推荐阅读顺序

### 4.1 Teacher 参数覆盖

![Teacher 参数覆盖](images/v3_01_teacher_coverage.png)

左图把归一化八维飞机参数投影到 PCA 平面。小点是候选飞机，黑色叉号是既有 Teacher，蓝色星号
是新候选，空心圆是冻结的零样本测试飞机。右图比较加入候选前后，每架零样本飞机到最近 Teacher
的归一化距离。柱子下降表示 Teacher Bank 覆盖变密，但 PCA 只是二维审计视图，不代表完整八维
距离。

### 4.2 v3 的典型长时失败

![v3 boundary-1448 全命令响应](images/v3_05_boundary_1448_all_commands.png)

这张图包含六种评测指令。重点看 `+25 deg/s`：v3 在约 9 s 后进入持续振荡，同时 requested 和
applied force 出现大幅周期动作。它解释了 v3 的 `22.9062 deg/s` 最大峰值和 TV 比门禁失败。

### 4.3 v4 对同一失败样本的修复

![v3-v4 boundary-1448 对比](images/v4_06_boundary_1448_response.png)

上图比较 `p_c`、`p_ref`、Teacher、v3 Student 和 v4 Student；下图比较三种控制器请求的
`F_as`。橙色 v3 曲线不断放大并持续振荡，红色 v4 曲线保持稳定且接近 Teacher。这是目前最直接
的闭环改进证据。

### 4.4 v4 的未见飞机结果

![v3-v4 未见飞机公平对比](images/v4_11_v3_v4_comparison.png)

上两幅柱状图分别比较逐机 PID、v3 baseline Student 和 v4 candidate Student 的 RMSE 与 TV。
底部散点图中，横轴是 v3 RMSE，纵轴是 v4 RMSE；点在对角线下方表示 v4 改善。v4 的总体变化
是小幅降低平均 RMSE，并把平均 TV 降低约 11.5%，但不同飞机上的改善不均匀。

### 4.5 v5 固定限速的反例

![固定限速 doublet 失败](images/v5_04_worst_limited_response.png)

左图是 `train_boundary-1605` 的 doublet 响应，右图是 requested/applied force。紫色
`v4 + 11 N/s` 在指令快速反向时不能及时从正力穿过零变成负力，导致 `p` 下冲到约
`-12 deg/s`。这一图说明统一固定限速虽然降低 TV，却会损伤必要的快速瞬态动作。

## 5. v3 图表索引，共 8 张

| 相对路径 | 内容和用途 |
| --- | --- |
| `images/v3_01_teacher_coverage.png` | Teacher 候选、冻结测试机和参数覆盖变化；用于说明 Teacher Bank 为什么要扩充 |
| `images/v3_02_holdout_core.png` | 三架 core holdout 到最近训练飞机的距离；说明整机验证对象确实与训练对象分离 |
| `images/v3_03_holdout_boundary.png` | 三架 boundary holdout 的最近训练飞机距离；用于覆盖边界稳定性风险 |
| `images/v3_04_distillation_progress.png` | v3 各蒸馏轮的离线动作误差、闭环 Student-Teacher 差距和 TV 比；说明最后一轮不一定最好 |
| `images/v3_05_boundary_1448_all_commands.png` | v3 最关键的长时失败样本；六种命令的 `p` 与控制力时域曲线 |
| `images/v3_06_unseen_summary.png` | 十架未见飞机上 Raw、逐机 PID、v3 Student 的 RMSE 和 TV 汇总 |
| `images/v3_07_distance_vs_generalization.png` | 最近训练飞机距离与 Student/PID RMSE 比；水平虚线 1 以下表示 Student 胜过 PID |
| `images/v3_08_v2_v3_comparison.png` | 相同十架飞机、相同六种命令下 v2 与 v3 的公平对比 |

v3 的主要数字：六架 holdout 平均 RMSE `1.0788 deg/s`，最大峰值 `22.9062 deg/s`，平均
requested-force TV `212.84 N`，TV 比 `1.816`。十架未见飞机上，Student 平均 RMSE
`2.4112 deg/s`，逐机 PID 为 `1.4094 deg/s`。

## 6. v4 图表索引，共 11 张

| 相对路径 | 内容和用途 |
| --- | --- |
| `images/v4_01_distillation_progress.png` | 三轮稳定性感知蒸馏的 validation action RMSE、闭环差距和 TV 比；最终按质量门选择 round 0 |
| `images/v4_02_core_0054_response.png` | core-0054 上 Teacher、v3 和 v4 的 matched `+25 deg/s` 时域对比 |
| `images/v4_03_core_0334_response.png` | core-0334 的同条件时域对比 |
| `images/v4_04_core_0515_response.png` | core-0515 的同条件时域对比 |
| `images/v4_05_boundary_1351_response.png` | boundary-1351 的同条件时域对比 |
| `images/v4_06_boundary_1448_response.png` | v4 修复 v3 长时振荡的代表图，优先查看 |
| `images/v4_07_boundary_1605_response.png` | boundary-1605 的同条件时域对比；它也是 v5 doublet 失败涉及的飞机 |
| `images/v4_08_core_0515_all_commands.png` | v4 在六种命令下的完整时域细节；doublet 和后半段 sine 仍可看到 requested/applied force 高频修正 |
| `images/v4_09_unseen_summary.png` | 十架未见飞机上 Raw、PID、v4 Student 的逐机 RMSE 和 TV 汇总 |
| `images/v4_10_distance_vs_generalization.png` | 参数距离与 Student/PID RMSE 比；不能据此认为距离单调决定性能 |
| `images/v4_11_v3_v4_comparison.png` | 相同对象和命令下 v3 与 v4 的公平对比 |

六张 `holdout_stability_comparison/.../response_comparison.png` 的读法相同：上半图看滚转角速度是否
贴近 `p_ref`、是否出现持续振荡；下半图看 requested force 是否高频抖动、饱和或振幅扩大。

v4 的主要数字：六架 holdout 平均 RMSE `0.9457 deg/s`，最大峰值 `4.7569 deg/s`，平均 TV
`249.99 N`，TV 比 `2.133`。峰值门通过，但 TV 比仍失败。十架未见飞机上平均 RMSE
`2.3784 deg/s`，平均 TV `421.65 N`，仍明显高于逐机 PID 的 `72.20 N`。

## 7. v5 图表索引，共 6 张

| 相对路径 | 内容和用途 |
| --- | --- |
| `images/v5_01_slew_tradeoff.png` | 仅在训练飞机上扫描 `7/11/16/29/88 N/s` 和不限速；展示 RMSE 与 TV 的权衡 |
| `images/v5_02_worst_unlimited_response.png` | 训练扫描中未限速 Student 的最坏时域样本，用于定位原始高 TV 行为 |
| `images/v5_03_holdout_summary.png` | 36 个 holdout 组合上 Teacher、未限速 v4 和限速 v4 的逐项 RMSE/TV；不能只看均值 |
| `images/v5_04_worst_limited_response.png` | `11 N/s` 在 boundary-1605 doublet 上的反例，是 v5 最重要的图 |
| `images/v5_05_unseen_summary.png` | 未见飞机上 Raw、PID、未限速 v4 和限速 v4 的 RMSE/TV 汇总 |
| `images/v5_06_distance_vs_generalization.png` | 加限速后，参数距离与 Student/PID RMSE 比的关系 |

`tradeoff.png` 中红点不合格，绿点通过训练飞机选参门。`11 N/s` 在训练飞机上被选中后保持冻结，
没有根据 holdout 结果重新选值。这保证了 holdout 的失败是有效证伪，不是调参泄漏。

v5 holdout 的主要数字：平均 TV 从未限速 v4 的 `249.48 N` 降至 `67.30 N`，TV 比从 `2.128`
降至 `0.574`；但平均 RMSE 从 `0.9446` 升至 `1.0357 deg/s`，最大峰值从 `4.7569` 升至
`7.9847 deg/s`。因此固定 `11 N/s` 限速器不能作为最终方案。

## 8. 结果边界和引用注意事项

1. v4 是当前正式对照基线，v5 是冻结模型上的部署限速诊断。不要把 v5 写成“新训练的 Student”。
2. 逐机 PID 的参数来自 5 s 调参窗口，而最终测试为 30 s。当前公平测试使用相同对象、命令和
   仿真窗口，但论文中仍应注明 PID 调参窗口不匹配。
3. 十架“未见飞机”已用于多次版本比较，现在属于固定回归集，不再是最终意义上的 untouched
   blind test。最终零样本结论应在方法和门限冻结后使用新预注册飞机。
4. 当前模型是 `p/F_as` 的 SISO P 通道，只能验证滚转角速度中的 Dutch-roll 污染，不能代替需要
   `beta`、`r` 或横向加速度的完整 GJB 横航向评测。
5. 所有汇总图都应与典型时域图一起解释。均值可能掩盖单架飞机、单条命令上的长时发散或峰值失败。

## 9. 最适合报告或汇报使用的四张图

| 顺序 | 图 | 要表达的观点 |
| ---: | --- | --- |
| 1 | `images/v3_01_teacher_coverage.png` | Teacher Bank 覆盖如何扩充 |
| 2 | `images/v4_06_boundary_1448_response.png` | 稳定性感知蒸馏确实修复了一个明确的闭环长时失败 |
| 3 | `images/v4_11_v3_v4_comparison.png` | v4 在固定未见飞机回归集上的收益与剩余差距 |
| 4 | `images/v5_04_worst_limited_response.png` | 为什么不能靠统一固定限速直接解决 Student 动作抖动 |

## 10. 全部图表预览

以下内容确保本说明书直接引用包内全部 25 张图片。

### v3：Teacher Bank 与第一版 Student-driven 蒸馏

#### v3-01 Teacher 参数覆盖

![v3-01 Teacher 参数覆盖](images/v3_01_teacher_coverage.png)

#### v3-02 Core holdout 选择

![v3-02 Core holdout](images/v3_02_holdout_core.png)

#### v3-03 Boundary holdout 选择

![v3-03 Boundary holdout](images/v3_03_holdout_boundary.png)

#### v3-04 Student-driven 蒸馏进度

![v3-04 蒸馏进度](images/v3_04_distillation_progress.png)

#### v3-05 Boundary-1448 全命令响应

![v3-05 Boundary-1448 全命令响应](images/v3_05_boundary_1448_all_commands.png)

#### v3-06 未见飞机汇总

![v3-06 未见飞机汇总](images/v3_06_unseen_summary.png)

#### v3-07 参数距离与泛化

![v3-07 参数距离与泛化](images/v3_07_distance_vs_generalization.png)

#### v3-08 v2 与 v3 公平对比

![v3-08 v2 与 v3 对比](images/v3_08_v2_v3_comparison.png)

### v4：稳定性感知蒸馏

#### v4-01 蒸馏进度

![v4-01 蒸馏进度](images/v4_01_distillation_progress.png)

#### v4-02 Core-0054 时域对比

![v4-02 Core-0054](images/v4_02_core_0054_response.png)

#### v4-03 Core-0334 时域对比

![v4-03 Core-0334](images/v4_03_core_0334_response.png)

#### v4-04 Core-0515 时域对比

![v4-04 Core-0515](images/v4_04_core_0515_response.png)

#### v4-05 Boundary-1351 时域对比

![v4-05 Boundary-1351](images/v4_05_boundary_1351_response.png)

#### v4-06 Boundary-1448 时域对比

![v4-06 Boundary-1448](images/v4_06_boundary_1448_response.png)

#### v4-07 Boundary-1605 时域对比

![v4-07 Boundary-1605](images/v4_07_boundary_1605_response.png)

#### v4-08 Core-0515 全命令响应

![v4-08 Core-0515 全命令响应](images/v4_08_core_0515_all_commands.png)

#### v4-09 未见飞机汇总

![v4-09 未见飞机汇总](images/v4_09_unseen_summary.png)

#### v4-10 参数距离与泛化

![v4-10 参数距离与泛化](images/v4_10_distance_vs_generalization.png)

#### v4-11 v3 与 v4 公平对比

![v4-11 v3 与 v4 对比](images/v4_11_v3_v4_comparison.png)

### v5：固定动作限速诊断

#### v5-01 限速率与 RMSE/TV 权衡

![v5-01 限速权衡](images/v5_01_slew_tradeoff.png)

#### v5-02 未限速最坏训练样本

![v5-02 未限速最坏训练样本](images/v5_02_worst_unlimited_response.png)

#### v5-03 Holdout 汇总

![v5-03 Holdout 汇总](images/v5_03_holdout_summary.png)

#### v5-04 固定限速最坏 Holdout 样本

![v5-04 固定限速失败](images/v5_04_worst_limited_response.png)

#### v5-05 未见飞机限速结果

![v5-05 未见飞机限速结果](images/v5_05_unseen_summary.png)

#### v5-06 限速后参数距离与泛化

![v5-06 参数距离与泛化](images/v5_06_distance_vs_generalization.png)

完整方法、数值和下一步计划见项目仓库中的 `docs/recent_work_overview_20260901.md`。
