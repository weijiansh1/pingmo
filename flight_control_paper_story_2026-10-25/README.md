# 论文叙事重构版 · 2026-10-25

正文：`webpage/dhc6_report.md`。网页：`webpage/dhc6_report.html`。PDF：`论文_控制经验复用_2026-10-25.pdf`。

以“源域控制经验能否帮助新目标在有限预算下适配”为中心，重写摘要、引言、相关工作、方法、结果和讨论。数学解释与 GJB 判据保留于附录。见[行文重构与证据边界](行文重构与证据边界.md)。

本版没有新增策略训练。主结果仅来自 `controller-route-v1`；旧论文及历史结果保留于 `/home/swj/pingmo/flight_control_paper_revision_2026-10-25` 和 `/home/swj/pingmo/flight_control_corrected_2026-10-24`。前一稿 Markdown 副本及差异位于 `revision/`。

## 来源

- 冻结实验：`/home/swj/pingmo/flight_control_corrected_2026-10-24/experiments/controller_route_20261024`。
- 汇总与协议的原样副本：`webpage/evidence/controller_route_report_summary.json`、`controller_route_protocol.json`。
- 对象库核查：`webpage/evidence/gjb_audit/`；190 对象样例未训练。
- 目标损失依据：冻结实验目录的 `target_rollout.py` 中 `loss()` 和 `adapt_students.py`。
- 数学交互说明保持原服务 `/math/`，主论文仍含完整数学附录。

## 重建

在本目录执行：

```bash
python paper_source/make_story_figures.py
python paper_source/dhc6_paper.py
```

修改中文正文后，先运行 `python paper_source/prepare_fonts.py` 取得匹配字符的本地 OFL 字体子集，再运行上述构建。首次字体准备需要网络，日常 HTML 构建使用已随包提供的字体与 KaTeX，无外部资源依赖。

PDF 使用 Chromium 打印 HTML，A4、背景图形、嵌入字体；复查记录、文件哈希和页面截图位于 `revision/` 和 `output/playwright/`。图表为通用论文草稿样式，未声称符合特定期刊格式。制图流程的软件资料列于[修订说明](行文重构与证据边界.md)。

## 网页

常驻服务由 `/home/swj/pingmo/paper_web_proxy_2026-10-25/start.py` 管理，访问当前 Jupyter 域名下的 `/proxy/8320/`。`/paper.pdf` 对应本版 PDF；不开放工作区目录列表。
