# 六通道品模对象库可视化（当前为 v4）

通过 Jupyter 代理访问：<http://127.0.0.1:8888/proxy/8330/>。服务只监听 `127.0.0.1:8330`，只读访问对象库，不修改任何对象库、模型或已有服务。

两页：
- 主页“模型与公式”（`static/model.html`、`model.js`）：六通道低阶等效系统 → 纵向/横航向小扰动动力学（含由模态解导数、交叉通道一般式）→ 全量动力学（风轴重力、惯性耦合、阵风与 Dryden）→ 运动学 → 离散实现 → GJB/MIL 判据 → 对象库与验证。公式用本地 KaTeX 渲染；每节的“代入当前对象”框由 `/api/model` 给出所选对象的导数、状态矩阵、全部通道的因式分解传递函数、模态与特征向量、零极点图、频率响应和精确性校核。
- “对象库浏览”（`/library`）：定级来源、参数分布对照、5,000 个对象的散点浏览、选中对象的现场仿真与大机动。

- `serve.py`：静态文件（只限 `static/` 下已知类型）、`/api/summary`、`/api/object`、`/api/model`、`/api/steps`、`/api/maneuvers`。仿真接口用耦合模型在 1 kHz 下现场计算，返回 50 Hz 采样。
- `start.py`：已在运行则只记录状态，否则后台启动；健康检查绕开 shell 导出的 `HTTP_PROXY`。状态写入 `server_state.json`。
- `static/`：纯 SVG 页面，不依赖外网；所有地址都是相对路径，可在代理前缀下工作。
- 链接 `#id=<对象编号>` 直接选中对象，加 `&maneuvers` 自动计算大机动。

```bash
cd /home/swj/pingmo/six_channel_model_2026-10-29/web
PYTHONDONTWRITEBYTECODE=1 python start.py
```
