服务入口：[论文网页](http://127.0.0.1:8888/proxy/8320/)。通过当前 Jupyter 登录访问，代理路径为 `/proxy/8320/`；如果 Jupyter 使用其他外部域名，沿用该域名和同一代理路径。

页面顶部提供论文、PDF 和交互数学说明。服务绑定 127.0.0.1:8320，由已有 Jupyter 代理转发；服务独立运行，关闭本轮会话不会主动停止它。

```bash
python /home/swj/pingmo/paper_web_proxy_2026-10-25/start.py
```

启动脚本会复用已存在的同一服务。进程记录在 server_state.json，日志在 server.log。当前服务为 `flight_control_paper_route_2026-10-25` 的飞控构建路线与公式修订版，PDF 共 17 页，数学说明沿用原页面；冻结训练结果不变。

当前代理检查记录为 route_proxy_verification.json：正文、本地资料、中文和数学字体、插图、PDF 与数学页面均经过检查，PDF 与本地文件逐字节一致。前版记录保留在 story_proxy_verification.json 与 proxy_verification.json；本次服务更改前的配置和状态保留在 before_route_serve.py、before_route_server_state.json。
