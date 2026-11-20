# 证据版本索引

| 材料 | 对应版本 | 当前使用方式 |
|---|---|---|
| teacher_admission_summary.json | 68 条目、32 accepted 的旧教师库 | 不能验证现稿 49/46 |
| finetune/unified_protocol.md 等 | 条件学生、64 次更新等旧协议 | 不能验证现稿 small 的 40 次更新 |
| sae_delay/ 与 experiments/.../results_large/ | 旧参考、不同初始化与追加采集条件 | 保留描述性结果；旧 source-only 列撤出有效跨时延证据 |
| gjb_tau_e/ | 未修复 20 ms 对齐的旧版本 | 历史追溯，不再作为当前 τe 主表 |
| ../../experiments/sae_delay_20261023/replay_legacy/probe_episode.json | 重新重建后严格时间截止的探针 | 当前因果探针主表 |
| ../../experiments/sae_delay_20261023/results_corrected/ | 公共参考、源时延范围、随机初始化与实际两轮采集 | 新训练对照 |
| ../../experiments/gjb_tau_e_20261023/results_phase_aligned/ | 归档频响相位修复与重新拟合 | 当前 τe 主表 |
| ../../revision/reference_check/ | 同一 tiny 权重、公共目标下的配对测试 | 新参考输入敏感性结果 |
| ../../revision/deep_review/audit_metrics.json | 5,000 对象规则及灵敏度检查、PPO 人工数值反例 | 当前品模/PPO 实现核查；非性能改进结果 |
| ../../pingmo_src/data/aircraft/generated/p_channel_library_iv_a_5000_v1/quality_label_review.json | 原库哈希、两个边界对象修正标签及未评品质条件 | 历史标签保留，另列修正结果 |
| ../../revision/deep_review/ppo_smoke_train_core_0000/ | 修订 PPO，一个对象、25,600 次转移 | 实现流程检查，未达到控制门限，不替代主线教师 |
| ../../revision/deep_review/ppo_smoke_executed/ | 小规模 PPO 实际执行的源码及协议哈希 | 不与旧教师源码或之后的注释修改混淆 |

归档主线缺少完整的教师准入、六轮蒸馏、800 对象幅值扫描、small 迁移原始记录及相应权重。本次没有据旧实验补造这些记录，论文相应数字保留为待独立核验的归档报告值。
