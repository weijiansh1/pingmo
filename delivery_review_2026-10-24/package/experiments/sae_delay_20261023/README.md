# 延迟表征实验：稀疏自编码器能否提取 τp（2026-10-23）

回应论文 6.1 计划里的“时延表征”一项：用稀疏自编码器（SAE）从 32×5 闭环历史中提取
延迟相关潜表征，并与原始历史、PCA、稠密 AE、互相关基线比较。实验分两层：

1. **表征层**：冻结 SAE，用因果历史前缀的线性探针估计 τp；
2. **闭环层**：用 16 维潜码替换原始历史训练策略，按延迟分层评估，并做跨时延测试。

## 数据

- 对象：IV-A 库 `train_core` / `validation` / `id_test`，按 τp 分三层（≤0.05、0.05–0.15、>0.15 s），
  各抽 100 / 30 / 40 架，共 510 架；
- 指令：8 条（±1、±2、±5、±10、±21、±31 deg/s 阶跃、双脉冲、多正弦）；
- 行为策略：`_report_review/weights/student_tiny.pt`（3,856 参数）；
- 窗口：32×5 历史，步长 3（16.7 Hz），共 489,600 个窗口；
- 留出：120 架、960 个“对象—指令”episode，按对象划分，无窗口泄漏。

## 表征层结果（留出 R² / MAE，1.2 s 因果前缀）

| 表征 | 维度 | R² | MAE |
|---|---:|---:|---:|
| 请求力—响应互相关 | 36 | **0.995** | **2.9 ms** |
| 稠密 AE | 16 | 0.608 | 32.8 ms |
| 原始历史 | 160 | 0.567 | 34.7 ms |
| PCA | 16 | 0.496 | 38.3 ms |
| L1-SAE（L0≈7） | 16 | 0.344 | 43.6 ms |
| Top-k SAE（k=4） | 16 | 0.122 | 53.2 ms |
| L1-SAE（L0≈3.5） | 16 | 0.075 | 56.1 ms |

结论：延迟信息主要存在于请求力与响应的时移；16 维稀疏瓶颈明显损失延迟信息，
稠密 AE/PCA 优于稀疏 SAE，但都远不如互相关基线。16 维本身不是主要瓶颈，稀疏约束才是。

## 闭环层结果（通过率，120 架留出）

| 变体 | 短延迟 | 中延迟 | 长延迟 | 长延迟 RMSE | 仅源域训练→长延迟 |
|---|---:|---:|---:|---:|---:|
| 原始历史（tiny 微调） | 93.1% | 95.3% | 90.0% | 0.408 | 90.0% |
| PCA-16 | 60.0% | 60.3% | 51.6% | 1.406 | 47.5% |
| 稠密 AE-16 | 69.1% | 72.5% | 63.7% | 0.731 | — |
| Top-k SAE-16 | 85.6% | 79.7% | 71.9% | 0.708 | 46.2% |
| SAE-aux（保留原始历史） | 93.1% | 95.6% | 90.0% | 0.409 | 90.0% |

- 除 raw 外，各变体均先在相同 BC 数据上训练，再用相同 DAgger 预算
  （96 架源域对象、β=0.5、8 轮）补充学生自身访问状态；
- 跨时延列只使用 τp ≤ 0.15 s 的源域对象训练，在 τp > 0.15 s 的 40 架留出对象上评价；
- 单种子；PCA 与稠密 AE 各重复一次，长延迟通过率分别落在 51.6–55.6% 与 52.5–63.8%，
  说明替换类的具体排序对种子敏感，但“明显低于 raw”在两次运行中一致。

结论：16 维替换类在闭环中普遍低于原始历史；Top-k SAE 是替换类中最好的，但跨时延迁移
反而明显下降。SAE 作为辅助输入可保持基线，但没有可测收益。当前证据不支持“用 SAE 替换
历史输入”。

## 复现

从 `pingmo/` 运行：

```powershell
python experiments/sae_delay_20261023/collect.py --n-train 100 --n-val 30 --n-test 40 --workers 12
python experiments/sae_delay_20261023/train_sae.py --suite --epochs 20
python experiments/sae_delay_20261023/features.py
python experiments/sae_delay_20261023/probe_episode.py
python experiments/sae_delay_20261023/train_policy.py train --variants raw_ft pca16 dense16 topk4 aux_topk4 --epochs 20 --dagger-epochs 8 --dagger-plants 96 --workers 12 --tag all_dagger2
python experiments/sae_delay_20261023/train_policy.py train --variants raw_ft pca16 topk4 aux_topk4 --epochs 20 --dagger-epochs 8 --dagger-plants 96 --workers 12 --source-only --tag cross
```

图表：

```powershell
python dhc6_blog_assets/make_delay_figures.py
```

## 文件

- `common.py`：环境 rollout、数据采样、tiny 学生；
- `collect.py`：采集闭环历史；
- `train_sae.py`：训练 dense/L1/top-k AE；
- `features.py`：预计算冻结特征；
- `probe.py` / `probe_linear.py` / `probe_episode.py`：延迟探针；
- `train_policy.py`：策略 BC + DAgger + 闭环评估；
- `results_large/`：数据、模型、探针与策略结果。
