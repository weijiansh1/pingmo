# Flight RL Control

本仓库正在核对单 `s` P-channel 候选模型、分数延迟、IV-A profile、响应级增益标定、Reference/Oracle 与特权 SAC。当前历史数据和 GPU 短训仅是原型诊断，不能作 GJB 或正式 Teacher 结论；状态见 `docs/manual_alignment_status.md`，GJB 2874-97 的全书项目记忆、延迟术语和采样参数边界见 `docs/references/gjb_2874_1997_project_memory.md` 与 `docs/references/gjb_2874_1997_iv_a_parameter_audit.md`。

不包含 Student、蒸馏、迁移或任何远端/GPU 启动行为。

## IV-A 机库生成

```bash
python scripts/01_generate_aircraft.py --seed 20260827 --candidates 16384
```

生成器固定选出 Level 1/2/3 各 900 和独立 OOD 300。产物位于
`data/aircraft/generated/p_channel_library_iv_a_manual_v1/`（不纳入 Git）；当前
`plants.jsonl` 期望 SHA-256 为
`479c16d7b5ddb363c0e2d069a7a1e2d510f0a77cf8335eb86da6bb359ca4b118`。等级口径和限制见
`docs/references/gjb_2874_1997_iv_a_parameter_audit.md`。

## 本地验证

```powershell
$py = 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe'
Set-Location 'C:\Users\24307\Desktop\品模py\flight_rl_control'
& $py -m pytest -q
& $py scripts\99_local_smoke.py
```

烟测会写入 `checkpoints/smoke/` 与 `results/local_smoke_report.json`。它只验证数据、环境、反向传播、checkpoint 和 router 统计的端到端正确连接；正式 curriculum、长训练、消融和 ID/OOD/Extreme 报告将只在收到明确 GPU 训练授权后运行。
