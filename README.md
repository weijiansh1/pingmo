# Flight RL Control

本仓库当前实现一条单通道、分飞机的控制学习路线：先为每个固定 P-channel 训练独立 SAC
Teacher，再把多个 Teacher 的动作规律蒸馏为一个读取飞机参数的 Dense Student。历史全机群
Global SAC、G0-G4 和 MoE 对照保留作实验记录，不再是默认训练入口。

当前控制契约见 [`docs/第一阶段_SAC控制设计.md`](docs/第一阶段_SAC控制设计.md)。GJB 阅读边界见
[`docs/references/gjb_2874_1997_project_memory.md`](docs/references/gjb_2874_1997_project_memory.md)。

## 当前流水线

```text
固定飞机 G(theta_i) -> 独立 SAC Teacher pi_i(o)
                                  |
                                  v
                    (o, theta_i, pi_i(o)) 数据
                                  |
                                  v
                    Dense Student pi(o, theta)
```

- 命令 `p_c` 是滚转角速度指令。
- 二阶参考模型产生 `p_ref`。
- Teacher Actor 读取 `p_c, p_ref, p, error, p_dot, previous F_as` 和 250 ms 历史；不读取 `theta`。
- Teacher 动作是完整 `F_as`，不是 `F + delta_F`。
- Student 读取同一 observation 和归一化的八维 `theta`，输出同一定义的 `F_as`。
- plant 与策略均以 `0.001 s` 更新。

## 执行顺序

先在 1 架飞机上验证完整链路：

```bash
python scripts/12_train_specialists.py --count 1 --device cuda
python scripts/30_collect_distillation_data.py --device cuda
python scripts/31_distill_dense_student.py --device cuda
python scripts/33_evaluate_student.py --device cuda
```

确认 `p_c / p_ref / p_raw / p_SAC` 曲线、动作限制和闭环指标定义正确后，再把 `--count`
依次扩到 10、50/100。每架飞机有独立目录、checkpoint 和报告；Teacher Bank 只汇总完整产物。

## 验证

```bash
pytest -q
```

单元与烟测只证明接口、时序、反向传播、checkpoint、蒸馏 split 和闭环评测可以运行，不证明
Teacher 已收敛，也不构成正式 GJB 一致性结论。
