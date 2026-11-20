# 专用教师跨对象控制与统一学生对照

本实验按用户授权，比较教师在非自身训练对象上的控制效果，以及统一学生在相同对象上的能力保留。使用 `controller-route-v1` 已冻结的配对权重：12 个不同源对象、三次训练重复、每次 12 个教师和一个 tiny 学生。它是原主试验之外的补充对照。所有 12 位教师的权重哈希均与同组学生保存的蒸馏来源一致。本实验没有训练或更新模型。

每个教师控制全部 12 个对象，统一学生也控制这 12 个对象，共 468 个控制器—对象组合。每组合包含 18 条 8 s 指令测试、4 条 8 s 初始偏差/扰动恢复测试，以及 6 条 30 s 长时测试，共 13,104 条轨迹。教师迁移时权重冻结，特权输入来自当前被控对象。学生仅使用普通观测和历史。参考信号、采样对齐、指令和门限沿用该冻结组的归档协议。

主要矩阵使用每组合的 18 条 8 s 指令测试。三次重复汇总如下：

| 控制方式 | 通过/测试组合 | 通过率 | 有限轨迹 RMSE 中位 deg/s | 超限/非有限终止 |
| --- | ---: | ---: | ---: | ---: |
| 教师控制自身对象 | 442/648 | 68.2% | 0.75 | 0 |
| 教师控制其他对象 | 1970/7128 | 27.6% | 3.28 | 12 |
| 同组统一学生 | 430/648 | 66.4% | 0.96 | 0 |

12 条失败轨迹仍计入通过率分母，RMSE 中位数使用有限轨迹。按对象和测试指令等权汇总；没有把确定性测试轨迹当作独立总体重复构造置信区间。学生三次通过率为 87.0%、63.9%、48.1%，自身教师为 82.4%、61.1%、61.1%。该组比较反映多对象能力保留，12 个对象都参与过相应学生的蒸馏。

自身教师及学生共 72 个控制器—对象组合与原归档复算对照；通过/失败逐项一致，RMSE 最大绝对差为 0。`replication_audit.json` 保存全部比较。

文件与复现：

- `protocol.json`：结果产生前确定的对象、权重、测试条件、展示对象和外对象教师。
- `frozen_source/`：本次使用的冻结仿真和评估实现；`route_common.py` 仅调整本地源码导入位置。
- `results/`：每组合的指标和完整 8 s / 30 s 轨迹。
- `run_evaluation.py`：CPU 多进程评估，支持按已完成组合恢复，不执行优化。
- `summary.json` 与 `controller_plant_results.csv`：完整聚合和逐对象数值。
- `summarize_and_plot.py`、`figures/`、`figure_manifest.json`：制图代码、SVG/PDF/PNG 及来源哈希。均未平滑曲线，失败后保留断点。原来预先指定的时域案例保存在 `cross_object_diagnostic`；主时域图在核查后按原独立验证门限筛选，每级取原对象顺序中的首个达标教师。修订依据及全部验证指标见 `illustration_protocol_v2.json`，筛选不使用测试结果。真实外部指令与记录目标分别绘制。
- `palette_audit.json`：背景对比度检查。曲线同时用实线、虚线、点划线和点线区分；矩阵用数字补充颜色。

运行 `python run_evaluation.py`，完成后运行 `python summarize_and_plot.py`。网页静态文件和可下载数值复制到 `paper_web_display_2026-10-25/webpage`，论文构建脚本为 `paper_web_display_2026-10-25/paper_source/build_display.py`。

制图方法参考 scientific-visualization；来源：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv / DOI](https://doi.org/10.48550/arXiv.2609.00065)。来源记录沿用本轮已核对的当前版本。
