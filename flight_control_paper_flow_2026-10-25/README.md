# 飞控论文全文统稿与时域验证 · 2026-10-25

编辑源为 [webpage/dhc6_report.md](webpage/dhc6_report.md)，网页为 [dhc6_report.html](webpage/dhc6_report.html)，PDF 为 [论文_全文统稿_2026-10-25.pdf](论文_全文统稿_2026-10-25.pdf)。[全部 60 个对象的时域图册](webpage/time_domain_gallery.html)同时提供逐对象 SVG 和 [60 页 PDF](webpage/paper/time_domain_atlas.pdf)。

本文围绕“品质相关对象族 → 专用特权教师 → 多轮蒸馏统一飞控控制器 → 未见对象闭环验证”展开。滚转是路线验证入口，目标适配是后续使用环节。正文先解释设计动机，再按教师学习、学生能力保留、未见对象控制、恢复与持续运行的顺序给出证据。

本次保留并恢复教师学习曲线，新增从原始冻结轨迹重绘的教师时域、教师—学生配对、未见对象、恢复和目标适配五组时域图。所有主图都紧邻对应的结果解释。完整图册按对象编号展示全部 60 个对象及三个种子，选用同一正阶跃和多正弦；完整统计继续使用全部 18 条测试指令。

当前论文的数据仍全部来自 controller-route-v1。12 个源对象分别进行三次独立训练，共 36 个教师实例；每个学生使用对应种子的 12 个教师。旧稿的 49 教师报告属于另一批实验，190 对象样例库尚未进行本文的策略训练。新增 GPU 4 计时实验已在独立目录完成，不合并为本文的控制性能结果；[卡 4 实测和 3,000 教师排期](3000教师训练时间估算.md)记录单卡并发吞吐与外推条件。

## 构建与数据

```bash
python paper_source/prepare_fonts.py
python paper_source/dhc6_paper.py
```

仅修改数学和已有字形时可只运行第二条。普通 HTML 构建使用本地 KaTeX 和本地字体；字体子集准备需要网络。PDF 使用 Chromium 的 A4 打印布局。图注与图构成同一分页单元，手机目录可折叠。

```bash
python paper_source/make_time_series_figures.py
```

时域制图只读取原有 NPZ，不运行仿真或训练。冻结数据目录为 `/home/swj/pingmo/flight_control_corrected_2026-10-24/experiments/controller_route_20261024`。选点规则在 `revision/time_series_selection.json`；每条主图轨迹与源文件哈希在 `webpage/evidence/time_series_manifest.json`，可下载数组在 `time_series_selected.npz`。角速度从 rad/s 转为 deg/s，不平滑、不抽稀，不将缺失或发散后的截尾填零。指令 RMSE 与正式测试一致，使用从 1 s 指令开始到终态的窗口。

[全文统稿说明](全文统稿说明.md)记录叙事变化、图表安排与实验范围；`revision/` 保存前稿、差异、数值和渲染检查。此前所有论文、原交付和冻结实验保留。

## 软件与制图来源

制图审查采用 scientific-visualization skill 的数据完整性、原始轨迹、线型冗余和最终尺寸检查流程，未指定期刊，不宣称符合某一期刊投稿要求。相关来源已于 2026-10-25 核对：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner. *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. 2026. [arXiv / DOI](https://doi.org/10.48550/arXiv.2609.00065)，当前记录为 2026-09-02 修订。该来源仅说明辅助制图流程，不作为飞控效果证据。
