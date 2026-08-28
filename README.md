# Flight RL Control

本仓库当前研究单 `s` P-channel 上的 1 kHz SAC 滚转品质整形。Actor 根据当前响应、响应趋势、四类实时品质代价和飞机参数输出等效力修正，不再以固定 reference tracking 为主要目标。当前控制与训练契约见 `docs/第一阶段_SAC控制设计.md`；GJB 边界状态见 `docs/manual_alignment_status.md`，全书项目记忆和 IV-A 采样审计见 `docs/references/gjb_2874_1997_project_memory.md` 与 `docs/references/gjb_2874_1997_iv_a_parameter_audit.md`。

不包含 Student、蒸馏、迁移或任何远端/GPU 启动行为。

## IV-A 机库生成

```bash
python scripts/01_generate_aircraft.py --seed 20260827 --candidates 16384
```

生成器固定选出 Level 1/2/3 各 900 和独立 OOD 300。产物位于
`data/aircraft/generated/p_channel_library_iv_a_manual_v1/`（不纳入 Git）；当前
`plants.jsonl` 期望 SHA-256 为
重新生成并核对后写入 manifest 的 SHA-256。等级口径和限制见
`docs/references/gjb_2874_1997_iv_a_parameter_audit.md`。

## 本地验证

```powershell
$py = 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe'
Set-Location 'C:\Users\24307\Desktop\品模py\flight_rl_control'
& $py -m pytest -q
& $py scripts\99_local_smoke.py
```

烟测会写入 `checkpoints/smoke/` 与 `results/local_smoke_report.json`。它只验证数据、环境、反向传播、checkpoint 和 router 统计的端到端正确连接；正式 curriculum、长训练、消融和 ID/OOD/Extreme 报告将只在收到明确 GPU 训练授权后运行。
