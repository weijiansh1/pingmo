# 原版网页显示修复 · 2026-10-25

原始研究基准仍是 `flight_control_original_restored_2026-10-25`，其 2,076 个文件和原模型均未修改。当前服务仅为两份原始 HTML、第一张主图和所需字体/数学资源提供显示覆盖。

- 将原 13 个独立公式及 34 处行内变量转为本地 KaTeX，保留原公式含义与编号，处理分式、上下标、导数及长公式布局。
- 将页面误显示的 `SECTIONS.append(tab(...))` 转回表 6。六行五列的单元格值逐项与原生成代码一致。
- 重绘图 1，保留原四步流程及数值，放大文字并允许点击查看矢量图。按原文明确 tiny/small 的参数量及 small 适配成绩的对应关系。
- 补齐本地中文和数学字体。其余 21 幅原图直接从原始目录提供。

`paper_source/build_display.py` 检查原有全部段落、标题和表格单元格保持一致；修复表 6 的新增单元格另行逐项核对。`revision/display_manifest.json` 保存原公式、表 6 数据、原始与显示 HTML 哈希。`revision/verification.json` 保存真实 Jupyter proxy 验证：86 个资源均返回 200；桌面和 390 px 手机页面无整页横向溢出，数学解析错误为零。长公式在窄屏内允许局部横向查看。

构建：`python paper_source/make_main_figure.py`，然后 `python paper_source/build_display.py`。本地字体已准备；仅更换字形时需要运行 `prepare_display_fonts.py`。这些脚本不训练模型、不修改科研结果。

制图检查参考 scientific-visualization，浏览器检查使用 Playwright。辅助制图流程来源：Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, [arXiv / DOI](https://doi.org/10.48550/arXiv.2609.00065)。2026-10-25 核对当前记录为 2026-09-02 修订。此来源记录在显示修复说明中，未加入原论文科研参考文献。

## 参考模型位置与符号的指定修改

用户随后要求区分对象采样与参考模型。现第 2.1 节先定义对象模型及采样频率、阻尼，第 2.2 节定义控制任务；参考模型移至第 4.1 节，改记为 omega_ref、zeta_ref，并明确 2 rad/s 与 0.7 是原有实验配置，现有材料未给出其选值依据和敏感性论证。公式按新出现顺序编号，相关章节引用同步调整。该项是用户明确要求的文字结构修改，区别于此前只保留全部原段落的显示修复。实验数值、结果和模型不变；差异及验证见 `revision/reference_clarification.diff` 和 `revision/reference_clarification.json`。

## 用户指定删除固定参考模型

已从当前网页删除固定二阶参考模型公式、2 rad/s 与 0.7 的取值说明、先前添加的参考信号小节及目录入口。正文按对象参数采样、指令跟踪和振荡抑制展开。剩余 12 个公式顺序编号。原实验指标中的目标信号仍按实验记录定义，未将历史数值改称重新测量的无滤波指令跟踪结果；本次未修改模型、训练环境或重算实验。验证和差异见 `revision/reference_removal.json` 与 `revision/reference_removal.diff`。

## 第二节参数三线表

按用户要求，在第 2.1 节加入“参数化滚转对象的参数说明”，列出八个采样参数与两个派生参数的符号、名称、单位、动态作用及来源。三线样式仅有表顶、表头下和表底横线，无内部行线或竖线；手机上表格可局部横向滚动。全部表格和正文引用按阅读顺序重新编号为表 1–14，原实验表格单元格数值保持一致。校验见 `revision/parameter_table_verification.json`。

## 图 10 与章节标题

图 10 按图 8 的两列排版、黑色完整轴框、浅灰网格、左对齐标题、顶部图例和底部说明重排为四行两列。8 个指令面板、24 条原 SVG 曲线路径、原有坐标范围及误差标注全部保留；只进行坐标系之间的仿射布局映射和样式修改，没有重新仿真、拟合或计算误差。原图 8 保持原样。过程见 `paper_source/style_figure10.py`，来源哈希与逐曲线校验见 `revision/figure10_style_manifest.json`，实际网页验证见 `revision/figure10_style_verification.json`。

按用户要求，原“5.3.2 扩展对象池抽样评估”改为“5.3.2 跨对象闭环控制性能评估”，目录同步；该节实验内容和结果数据未改。
