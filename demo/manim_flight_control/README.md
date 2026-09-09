# Pingmo Manim Demo

这是一个按 NousResearch [`manim-video`](https://github.com/NousResearch/hermes-agent/tree/main/skills/creative/manim-video) 工作流制作的仓库原理动画。

## 成品

- `final.mp4`: 1920x1080、60 fps、约 2 分 11 秒的 H.264 视频
- `poster.png`: 成片封面帧
- `final.srt`: 完整中文字幕；`final.mp4` 内也封装了一条默认关闭的中文字幕轨
- `plan.md`: 叙事弧、逐幕规划、颜色语义和数据来源
- `script.py`: 单文件 Manim 实现，每一幕可独立渲染
- `concat.txt`: 1080p 场景拼接顺序

视频无配音，关键结论直接显示在画面中。支持字幕轨的播放器可以额外开启“中文解说”。

## 内容

动画解释了仓库当前的完整路线：

1. 同一滚转指令作用于不同飞机时，原始响应为何不同
2. 为什么通用控制追求“不同动作、统一响应”
3. 32 个逐机纯奖励 TD3 Teacher 如何蒸馏成一个 540,417 参数 Student
4. Student-driven / DAgger 如何处理闭环状态分布漂移
5. 冻结 Student 在 10 架未见飞机、60 个飞机-命令对上的真实结果
6. 为什么当前正式状态仍是 `quality_gate_failed`

## 渲染

依赖 Python 3.10+、Manim CE 0.20.1、ffmpeg 和中文字体。低清草稿：

```bash
manim -ql script.py \
  Scene1_SameCommand Scene2_SameQuality Scene3_TeacherBank \
  Scene4_DAgger Scene5_UnseenResults Scene6_HonestBoundary
```

高清成片：

```bash
manim -qh script.py \
  Scene1_SameCommand Scene2_SameQuality Scene3_TeacherBank \
  Scene4_DAgger Scene5_UnseenResults Scene6_HonestBoundary

ffmpeg -y -f concat -safe 0 -i concat.txt -c copy final.mp4
```

## 数据来源

画面指标直接取自：

- `docs/current_code_and_results_20260830.md`
- `results/pure_reward_teacher_bank_coverage_v3/student_driven_dense_balanced_holdout/round_metrics.csv`
- `results/pure_reward_teacher_bank_coverage_v3/student_driven_dense_balanced_holdout/pipeline_report.json`
- `results/pure_reward_teacher_bank_coverage_v3/unseen_aircraft_v3/report.json`
- `results/pure_reward_teacher_bank_coverage_v3/unseen_comparison_v3/comparison.json`

成片不声称 Student 已优于逐机 PID，也不声称满足完整 GJB 横航向飞行品质要求。
