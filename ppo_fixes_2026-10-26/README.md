# PPO 概率与轨迹边界修复版

入口为本目录的 `src/teacher/ppo/trainer.py`，由当前项目 `FQControl_Model/src/` 独立复制。原交付、当前发布仓库、冻结实验与权重保留。本次只修复两处实现问题，没有启动教师训练。

1. Actor 新增 `sample_for_ppo()`，返回有界执行动作与原始高斯采样值。训练缓存保留原始采样值，采样和更新时的概率对应同一个值。原 `act()` 的三个返回值及确定性推理接口保持兼容。
2. 8 秒按连续控制任务的采样截断处理。PPO 使用 `step(..., auto_reset=False)`，在环境重置前计算末端价值；下一轮采样开始时显式重置。其他环境调用仍默认自动重置。

网络结构、权重键、动作限幅、奖励、对象生成、训练配置默认值与优化预算保留。新增日志、检查点和汇总字段 `ppo_protocol=ppo-raw-sample-final-state-v2`，用于标明修复后的实现。

本机 NumPy 1.26 使用 `np.trapz`，NumPy 2.x 使用 `np.trapezoid`。灵敏度标定增加两者的兼容选择，继续采用相同的梯形积分计算。

从本目录运行后续脚本，或把本目录置于 `PYTHONPATH` 首位，才能使用这份修复源码。新训练应使用新的结果目录。

CPU 回归验证：

```bash
cd /home/swj/pingmo/ppo_fixes_2026-10-26
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

`before/` 保存被修改文件的原件；`changes.json` 记录来源与修改前后哈希；`ppo_fixes.patch` 提供对原源码的差异。
