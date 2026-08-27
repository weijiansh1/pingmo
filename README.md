# Flight RL Control

本仓库正在按 `GJB_MoE_SAC_滚转品质实验手册_v1.md` 对齐：GJB-original `s^2` P-channel、分数延迟、IV-A profile、响应级增益标定、Reference/Oracle 与特权 SAC。当前历史数据和 GPU 短训仅是原型诊断，不能作 GJB 或正式 Teacher 结论；状态见 `docs/manual_alignment_status.md`。

不包含 Student、蒸馏、迁移或任何远端/GPU 启动行为。

## 本地验证

```powershell
$py = 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe'
Set-Location 'C:\Users\24307\Desktop\品模py\flight_rl_control'
& $py -m pytest -q
& $py scripts\99_local_smoke.py
```

烟测会写入 `checkpoints/smoke/` 与 `results/local_smoke_report.json`。它只验证数据、环境、反向传播、checkpoint 和 router 统计的端到端正确连接；正式 curriculum、长训练、消融和 ID/OOD/Extreme 报告将只在收到明确 GPU 训练授权后运行。
