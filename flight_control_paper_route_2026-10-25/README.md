# 飞控控制器构建路线与公式修订 · 2026-10-25

正文源文件：`webpage/dhc6_report.md`。网页：`webpage/dhc6_report.html`。PDF：`论文_飞控构建路线_2026-10-25.pdf`。

主线恢复为“品质相关对象族 → 特权强化学习教师 → 多轮蒸馏统一飞控控制器 → 未见对象闭环验证，必要时再适配”。滚转是当前可行性验证入口；目标适配作为路线中的后续环节。结果依次展示教师、学生能力保留、蒸馏轮次、未见对象跟踪与恢复，以及目标适配。

公式同时支持正文/表格中的行内 `\(...\)` 与独立 `$$...$$`，由本地 KaTeX 在构建时生成 HTML 和 MathML；数学字体随网页提供。所有变量、时间下标、教师/学生上标、导数及平方采用数学排版，代码标识和文献引用保持原用途。

## 文件与来源

- [本次主线及公式修订说明](主线与公式修订说明.md)。
- `revision/previous_manuscript.md`、`manuscript.diff`：上一稿与本次差异。
- `revision/math_manifest.json`、`math_verification.json`：逐个公式与渲染检查。
- `output/playwright/`：桌面、手机和公式局部截图。
- `webpage/evidence/`：原样保留的冻结协议、控制结果和 GJB 核查材料。

主结果仅来自 `controller-route-v1`，本次没有新增策略训练或修改实验数据。完整冻结实验位于 `/home/swj/pingmo/flight_control_corrected_2026-10-24/experiments/controller_route_20261024`；上一叙事稿保留于 `/home/swj/pingmo/flight_control_paper_story_2026-10-25`。

## 构建与访问

```bash
python paper_source/dhc6_paper.py
```

修改中文正文后，先执行 `python paper_source/prepare_fonts.py` 更新 OFL 字体子集；字体准备需要网络，普通 HTML 构建离线完成。PDF 由 Chromium 打印为 A4，并嵌入数学和中文字体。`revision/format_inline_math.py` 是本次一次性格式转换记录，日常直接编辑 Markdown 中的 TeX，无需重复自动转换。

常驻网页由 `/home/swj/pingmo/paper_web_proxy_2026-10-25/start.py` 提供；入口为当前 Jupyter 域名下的 `/proxy/8320/`，PDF 为 `/proxy/8320/paper.pdf`，交互数学说明继续位于 `/proxy/8320/math/`。
