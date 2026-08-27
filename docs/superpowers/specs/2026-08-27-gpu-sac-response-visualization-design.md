# GPU SAC 响应可视化设计

本次只进行单架 `train_core-0000` 的探索性 SAC 训练。训练集为当前 IV-A
`p_channel_library_iv_a_manual_v1`，训练环境与现有奖励保持不变。

训练后使用同一随机种子、同一初值和同一驾驶员阶跃命令回放两次：一次强制
`Delta F = 0` 得到 raw；一次由确定性 SAC 策略输出得到 SAC。两次回放共同使用
参考模型产生 `p_ref`。结果图包含 `p_raw`、`p_ref`、`p_SAC`，以及 SAC 的
`Delta F` 和每步奖励；checkpoint 和 JSON 报告与图同时保存。

这是控制方向诊断，不能作为 GJB 认证、全局泛化或 MoE 结论。
