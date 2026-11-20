# 飞控论文阅读服务

当前服务：`flow-2026-10-25`。论文目录：`/home/swj/pingmo/flight_control_paper_flow_2026-10-25`。服务监听 `127.0.0.1:8320`，通过已有 Jupyter 身份验证代理访问。

- [论文](http://127.0.0.1:8888/proxy/8320/)
- [21 页 PDF](http://127.0.0.1:8888/proxy/8320/paper.pdf)
- [全部 60 个对象的时域图册](http://127.0.0.1:8888/proxy/8320/webpage/time_domain_gallery.html)
- [数学说明](http://127.0.0.1:8888/proxy/8320/math/)

启动命令：`python /home/swj/pingmo/paper_web_proxy_2026-10-25/start.py`。进程标识在 `server_state.json`，日志在 `server.log`。更换服务代码时需要只重启经命令行核实属于本服务的进程；正文和 PDF 每次请求从当前目录读取。

`flow_proxy_verification.json` 记录本版真实代理验证：包括正文、图册 60 张 SVG、PDF、数学页面和字体在内的 135 个资源全部返回 200，代理 PDF 与本地文件字节一致。没有在报告、URL 或日志中记录 Jupyter 令牌。`before_flow_*` 保留切换前的配置与状态，旧稿目录继续保留。
