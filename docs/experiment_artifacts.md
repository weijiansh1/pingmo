# Teacher / Student 实验产物格式

完整入口 `scripts/35_run_teacher_student_pipeline.py` 使用固定目录层级，避免 checkpoint、图和临时
数据混在一起：

```text
results/teacher_student_pipeline/
├── pipeline_report.json
├── completion_audit.json
├── artifact_index.csv
├── diagnostics/
├── benchmarks/
├── 01_teachers/
│   ├── teacher_bank.json
│   ├── summary/
│   │   ├── teacher_bank_summary.json
│   │   ├── teacher_bank_metrics.csv
│   │   └── teacher_bank_summary.png
│   └── <plant>-seed-<seed>-cfg-<hash>/
│       ├── teacher_actor.pt
│       ├── training_checkpoint.pt
│       ├── progress.json
│       ├── evaluation.json
│       ├── report.json
│       ├── response_comparison.png
│       ├── all_evaluation_commands.png
│       └── comparison_vs_pid/
│           ├── comparison.json
│           └── controller_comparison.png
└── 02_student_driven_distillation/
    ├── pipeline_report.json
    ├── round_metrics.csv
    ├── distillation_progress.png
    ├── round_000_teacher_driven/
    │   ├── dataset/
    │   ├── student/
    │   │   ├── student.pt
    │   │   ├── report.json
    │   │   ├── routing_report.json
    │   │   └── router_utilization.png
    │   └── evaluation/
    ├── round_001_student_driven/
    │   ├── dataset/
    │   ├── student/
    │   └── evaluation/
    ├── ...
    └── final/
        ├── student.pt
        └── evaluation/
└── 03_final_comparison/
    ├── comparison_report.json
    ├── controller_metrics.csv
    ├── raw_pid_teacher_student_summary.png
    └── aircraft/<plant>/raw_pid_teacher_student.png
```

## 完整性规则

- `pipeline_report.json` 是根索引；`status=complete` 只表示 Teacher 与 Student 两级门禁均通过。
- `completion_audit.json` 必须逐项验证 Teacher/Student 契约、Student-driven 分片哈希、
  部署与训练恢复 checkpoint、闭环改善以及 11 张最终时域图；不能只信任上游
  `status` 字段。
- `artifact_index.csv` 记录正式产物的绝对路径、字节数和 SHA-256，本地完整交付时索引中
  不允许有缺失或空文件。
- `teacher_bank.json` 只在每个独立 Teacher 的 `quality_gate.passed=true` 时标记为 `complete`。
- 每个 checkpoint 在引用它的 manifest 中记录 SHA-256；数据 shard 同样记录 hash。
- Teacher checkpoint 分别保存 Actor 与训练期 privileged Critic 的字段合同；只有
  `actor_observation_contract` 可进入蒸馏和部署，Critic 的时延 FIFO 不属于策略历史输入。
- 完整交付同时保留每架飞机的 `teacher_actor.pt` 和 `training_checkpoint.pt`：前者用于部署与
  蒸馏，后者用于恢复 Actor/Critic/target/优化器状态。
- PID-guided 正式流水线的 Teacher Actor 输入是当前 7 维控制 observation，
  `raw_history_steps=0`，且不读取 `theta`。纯奖励修订实验则使用统一 34 维合同：相同 7 维当前量、
  2 维执行器量和 25 个更早的 requested action（加上 7 维中的 `previous_force` 共 26 步）。两种
  checkpoint 不得放入同一个 Teacher Bank；进入蒸馏的全部 Teacher 必须具有完全相同的
  `actor_observation_contract`。
- 当前正式线性 MoE Student 路由器只读取归一化 8 维 `theta`，线性专家只读取当前 observation
  中的 `error, integrated error, p_dot`；checkpoint 必须记录索引 `[3, 4, 5]`。若正式采用 34 维
  纯奖励 Teacher，需单独验证 Student 是否读取并保留动作记忆，不能让数据集虽为 34 维而专家仍
  静默忽略新增字段。
- 被主动终止或仅作资源检查的运行放在 `diagnostics/`、`benchmarks/`，不得留在正式
  `01_teachers/` 或进入 `teacher_bank.json`。
- `round_000` 的 `driver` 必须是 `teacher`；后续轮次必须是 `student`，标签源始终是对应飞机的
  specialist Teacher。
- 正式 Teacher-Bank 蒸馏使用 `all_aircraft_command_holdout`：所有 Bank 飞机参与梯度训练，每架
  的 `eval-*` 命令只用于 validation。`aircraft_holdout` 是单独的零样本泛化诊断；二者不得混写，
  命令 holdout 结果不能作为跨飞机泛化证据。
- 最终 Student 按 validation 闭环 `Student RMSE - Teacher RMSE` 选择，不按训练 action MSE
  选择。

## 必查图表

- Teacher 的 `all_evaluation_commands.png` 必须检查全部留出命令以及 requested/applied force。
- `controller_comparison.png` 必须在相同命令与坐标中同时展示 `p_c, p_ref, Raw, PID, RL Teacher`；
  对应 `comparison.json` 提供逐命令和汇总指标。
- 每个蒸馏轮次的 `<plant>/teacher_student_comparison.png` 在同一坐标系对比 `p_ref`、Teacher、
  Student 及两者 requested/applied 控制力。
- `closed_loop_summary.png` 将每个飞机-命令对的 Teacher RMSE 与 Student RMSE 作散点比较。
- `distillation_progress.png` 同时展示离线 action RMSE 和闭环 Student-Teacher RMSE 差，防止用
  很小的行为克隆 loss 掩盖闭环分布偏移。
- `round_metrics.csv` 提供便于论文表格和外部分析读取的逐轮摘要。
- `router_utilization.png` 和 `routing_report.json` 必须报告 train/holdout 的专家均值占比、熵、
  最大占比和硬路由飞机数，用于排除所有飞机集中到单一专家。
- `03_final_comparison/aircraft/*/raw_pid_teacher_student.png` 是最终统一时域证据，必须用相同命令、
  相同种子和相同坐标同时展示 Raw、PID、RL Teacher 与 Student。
- Student 门禁同时检查闭环跟踪、峰值误差、requested force 总变差绝对值以及相对 Teacher 的
  总变差比例；执行机构限速后的平滑曲线不能掩盖网络请求动作抖动。
