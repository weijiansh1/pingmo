先看 [对象库核查与修正](对象库核查与修正.md)，再看 [强化学习逐步说明](强化学习逐步说明.md)。结论是保留滚转路线验证，修正对象覆盖和指标解释，补足控制接口与任务可达性的定义。

本目录是原 v3 交付的独立补充。原 PDF、5000 对象库、实验协议、权重和 v3 压缩包均未覆盖。页面图来自用户提供的本地 PDF；外部原始文献链接放在相应说明段落中。

| 文件 | 用途 |
|---|---|
| coverage.json / audit_library.py | 旧库的可复算覆盖统计 |
| reviewed_objects.jsonl | 对每个旧对象的修正解释层，按 plant_id 关联；不改变动力学参数 |
| quality_parameter_registry_reviewed.csv | 核对原 PDF 后修正的条款清单 |
| gjb_metric_corrections.py | 有符号峰谷、阻尼分支和调节时间截断处理 |
| build_joint_library.py / pilot_joint_v1 | 新联合采样及活动边界构造；190 个样例，未训练控制器 |
| verify_audit.py / verification.json | 科学指标和新数据的回归检查 |
| extract_pages.py / source.json / pages | 原扫描 PDF 的页面证据与校验 |

构造器和验证器使用同级目录 `flight_control_corrected_2026-10-24/pingmo_src` 中的原模型和对齐后的灵敏度函数。页面提取器另需同级用户提供的 GJB PDF。依赖为 Python、NumPy、SciPy；提取页面另需已有的 pypdf 和 Pillow。

新样例使用显式字段 `partial_screen_level`、`partial_components`、`full_gjb_level=null`，不是对旧训练入口的无缝替换。正式训练前，需要在新协议中指定训练对象选择、边界探针角色、等级/组合抽样权重和 OOD 套件，并让新入口读取这些字段。其作用是使构造问题及修正可以检查，不预先声称控制器性能改善。
