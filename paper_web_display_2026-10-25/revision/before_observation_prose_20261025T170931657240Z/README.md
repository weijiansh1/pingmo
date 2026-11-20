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

## 用户指定的 50 对象口径

按用户说明，实际训练对象超过 50 个，本稿统一按 50 个对象、50 个逐对象教师展示。摘要、方法、实验设置、结果、讨论、结论和图 1 同步更新。已有的 46 个通过记录保留，不据此重新计算通过率；学生评估的 48 对象划分和各组原实验数据保留。图 5 仍展示已有逐对象记录的位置。修改前页面、主图和生成脚本保存在 `revision/teacher_scope50_snapshot.json` 指向的目录。

## SAE 叙述、教师跨对象对照与图 8 标题

按用户要求，图 8 标题 B 后的括号说明已删除，保留“B · 训练完成后的闭环控制效果”。只删除该说明的字形并居中标题，全部曲线路径不变；见 `paper_source/trim_figure8_title.py` 与 `revision/figure8_title_edit.json`。

第 6.1 节围绕统一学生的历史输入设计展开，依次讨论时延信息读出、表征替换后的闭环性能、长时响应与 LOES 分析。原六张实验表的全部数据单元格和七幅相关图保留。叙述修正了原始实验口径：SAE 数据使用六条正向阶跃与双脉冲、多正弦；DAgger 追加数据后训练 8 个周期；raw 与 SAE-aux 继承原 tiny 权重，短/中时延限制作用于再训练样本。表征读出不直接解释为控制收益。实现见 `paper_source/sae_story.py`。

第 5.3.2 节新增 12 对象配对组的教师—对象矩阵、三次训练重复结果、同一目标对象上的自身教师/外对象教师/学生时域对照。完整说明和数据位于 `../cross_object_control_2026-10-25`。新增评估不修改原教师、学生或其他冻结实验。

用户随后要求将图 13 的评估内容融入前面的分析，减少单独展示。120 对象的通过率、各等级统计、RMSE、操纵效能相关和慢响应区间已合并到跨对象分析正文，撤下独立 `student_library_eval.svg` 展示。用户进一步指出重排后仍存在学生单独时域图及“图 13 补充展示……”段落；现将 `student_grid.svg` 和该段落一起撤下，保留前面的教师—学生时域对照。原图文件和数据归档保留，没有拼接成更大的图。当前共 22 幅图、15 张表，后续图号和正文引用同步。原图 8、图 10 的编号不变。

网页已通过实际 Jupyter proxy、桌面和手机检查；数学解析错误为零。复核文件位于 `revision/sae_cross_object_verification.json`、`revision/sae_cross_object_proxy.json`。制图来源沿用上文 software 说明。

## 顺序编号、A 标题与图 12 核查

按用户要求，A 标题的括号说明也已删除；A、B 标题均不再附带该类说明。`trim_figure8_title.py` 现同时处理图 7、图 8，仅删除说明字形并居中标题。

正文和图中的对象、教师及 DHC-6 状态改为各实验组内从 1 开始的顺序编号，主试验、配对实验、时延实验分别保持对应关系。DHC-6 使用状态 1–12；配对矩阵使用对象 1–12、教师 1–12。源编号映射在 `webpage/evidence/display_identifier_mapping.json`，对应源图位于 `paper_source/identifier_sources`。`display_identifiers.py` 仅改文字字形和标签锚点，逐文件核对全部非文字几何不变；记录见 `revision/readable_identifier_assets.json`。网页下载的配对结果 CSV 同步使用展示编号，其所有数值字段与原始 468 行记录一致。

新增“通过率”定义，明确配对组按对象—指令组合统计，120 对象结果按每个对象的归档汇总指标统计。三项门限的达标比例不等同于振荡衰减或渐近稳定性。

用户质疑图 12 后核查发现：最初按每级最小源编号选择的三个对象中，前两位自身教师并未达到训练时的独立验证门限。原图还仅画滤波目标，未画真正外部指令。此次按原独立验证门限筛选，第一组 12 个教师中 8 个达标，每级依原对象顺序取首个，得到对象 1、6、9；迁移教师为教师 6、9、1。选择只用训练阶段保存的验证指标，不用测试结果，完整矩阵与全部统计保持不变。新图明确画出外部指令和实验记录目标，保留 8 s 阶跃及 30 s 恢复测试；三类控制器的输入数组逐项一致。修订规则和全体资格记录在 `../cross_object_control_2026-10-25/illustration_protocol_v2.json`。

原来的失败例保存在 `paper/cross_object_diagnostic.svg` 并在正文提供链接。它显示，即使整段 RMSE、峰值、操纵量变化均满足当前门限，后段振荡仍可能增长，评估抑振需要另看长时恢复和末段残余振荡。图 12 属于 12 对象补充组，不替代 50 教师主试验。

## SAE 在全文中的结构调整

用户再次指出 SAE 放在讨论末尾像工作汇编，并明确本轮只安排位置和行文，不运行实验。本轮没有启动训练或重新仿真。

第 3.3.1 节先说明学生撤去特权信息后需要从历史中获取对象动态线索，再提出对照问题：冻结的 PCA、AE、SAE 前端能否替代随控制任务学习的时序编码。基线本来就有 16 维历史特征，各替换前端也输出 16 维，正文因此按表征学习方式解释该实验。

第 5.3.5 节把关键消融接在学生结果后，主文保留四段分析和一幅闭环对照图，结论限于现有 SAE 方案未带来收益、当前学生保留基线历史编码。完整数据设置、探针表、训练对照、LOES 与稳健性材料集中至参考文献后的附录 A。删除原讨论中的独立 SAE 大节，并同步相关工作、结论、目录和章节引用。

原 22 幅图和 15 张表的数据全部保留，其中正文 16 幅图、附录 6 幅图；关键闭环消融图现为图 14，后续图号及引用同步。修复了 PID 与 DHC-6 小节共用 `s54` 的锚点。图 11 下补充了行列、对角线、末行和三组重复的读图说明及数值示例。

实现为 `paper_source/integrate_sae.py`；修改前版本位于 `revision/before_sae_structural_integration`。逐表数据、全部图像来源、编号、目录链接和网页显示复核见 `revision/sae_structural_integration_verification.json` 与 `revision/sae_structural_integration_proxy.json`。

## 公式 2 排版

按用户要求，公式 2 的观测向量改为完整一行，教师和学生的输入定义另起一行，取消向量中途的强制换行。符号、维数、观测内容和编号保留。桌面显示完整，窄屏在公式内部横向滚动；网页截图和 PDF 排版预览保存在 proxy 的 `output/playwright/equation2-*`，验证记录为 `revision/equation2_layout_verification.json`。

## 闭环仿真与时域曲线来源图

按用户要求，在第 2.2 节新增图 1 及一段说明，明确采样参数如何确定对象、学生动作如何经力接口与执行机构作用于对象、数值仿真如何逐步得到响应并记录成时域曲线。原四步研究流程图顺延为图 2，后续图号与正文引用同步。现有 22 幅图保留，全文现为 23 幅图；表格数据、公式与原段落经逐项检查，段落只更新图号引用。新增图为机制示意，不引入新的实验数据。

生成脚本为 `paper_source/make_closed_loop_figure.py`，网页插入与编号处理为 `paper_source/closed_loop_explanation.py`。SVG、PDF 和 PNG 在 `webpage/paper/closed_loop_simulation.*`。网页与 PDF 均已检查，字体、上下标和闭环箭头显示完整；桌面和窄屏无整页横向溢出。验证见 `revision/closed_loop_verification.json`，修改前版本见 `revision/closed_loop_snapshot.json`。
