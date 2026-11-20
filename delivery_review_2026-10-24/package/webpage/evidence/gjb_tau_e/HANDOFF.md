# GJB τe 评价：交接说明（2026-10-24 凌晨）

## 这次完成了什么

1. 实现 0.1–10 rad/s 长时闭环频响 harness（180 s 多正弦，50 Hz 采样）。
2. 实现 4 阶滚转 LOES + τe 拟合（滚转 + 螺旋 + 荷兰滚），以及 VAF/相位残差门。
3. 完成 **120 架 × 3 策略** 主评价：raw、Top-k SAE-16、SAE-aux。
4. 完成 **三档激励幅值** 0.5 / 1.0 / 2.0 deg/s 的稳健性评价。
5. 完成 **第二策略种子** 的重训与 τe 评价。
6. 完成拟合门限（相位 8°/10°/12°）敏感性与 Top-k SAE 时域/频谱诊断。
7. 论文新增 6.1.5、6.1.6，表 10–12，图 19–22；网页已同步。

## 主要结果（种子 1，1.0 deg/s）

| 策略 | 可评/120 | τe 中位 (s) | 短/中/长中位 (s) | 1 级 | 2 级 | 3 级 | >3 级 |
|---|---:|---:|---|---:|---:|---:|---:|
| 原始历史 | 73 | 0.073 | 0.042 / 0.127 / 0.299 | 43 | 15 | 4 | 11 |
| SAE-aux | 102 | 0.102 | 0.032 / 0.125 / 0.309 | 51 | 14 | 7 | 30 |
| Top-k SAE-16 | 0 | — | — | 0 | 0 | 0 | 0 |

- 长延迟可评对象的 τe 中位约 0.30 s，超过 A21 的 3 级上限 0.25 s。
- Top-k SAE 在三种幅值、两个种子下均为 0/120 可评；VAF 中位 0.73–0.82，
  相位 RMSE 中位 35–53°，时域低频段功率明显更高。它是 `not_assessable`，不是 τe=0。
- 幅值主要改变可评比例（raw 41/73/91；aux 78/102/111），但共同可评集上的
  τe 中位跨度约 0.01 s。
- 两种子的可评数、τe 中位和等级分布基本一致；raw 73→77，0.073→0.078；
  aux 102→106，0.102→0.106。

## 关键限制

- 拟合门限（VAF 0.95、相位 10°）是项目设定，不是 GJB 规定；可评比例依赖门限。
- 两种子、三幅值，仍不是多源/多机型统计。
- 参考模型被计入飞控系统；这是协议选择，需要在论文中明确。
- 不构成 A21 符合性结论。

## 复现命令

```powershell
cd pingmo

# 主评价（种子 1）
python experiments/gjb_tau_e_20261023/tau_e.py batch --workers 12

# 幅值稳健性
python experiments/gjb_tau_e_20261023/tau_e.py batch --amplitude 0.5 --workers 12 --tag batch_amp050
python experiments/gjb_tau_e_20261023/tau_e.py batch --amplitude 2.0 --workers 12 --tag batch_amp200

# 第二种子
python experiments/sae_delay_20261023/train_policy.py train --variants raw_ft topk4 aux_topk4 \
  --epochs 20 --dagger-epochs 8 --dagger-plants 96 --workers 12 --tag seed2 --seed 424242
python experiments/gjb_tau_e_20261023/tau_e.py batch \
  --policy-dir experiments/sae_delay_20261023/results_large/policies_seed2 \
  --tag batch_seed2 --workers 12

# 分析与图
python experiments/gjb_tau_e_20261023/analyze_amplitude.py
python experiments/gjb_tau_e_20261023/analyze_seed.py
python dhc6_blog_assets/make_tau_e_figure.py
python dhc6_blog_assets/make_tau_e_amplitude_figure.py
python dhc6_blog_assets/make_tau_e_seed_figure.py
python experiments/gjb_tau_e_20261023/diagnose_topk4.py
```

## 文件

- 评价器：`pingmo/experiments/gjb_tau_e_20261023/tau_e.py`
- 幅值/种子分析：`analyze_amplitude.py`、`analyze_seed.py`
- 诊断：`diagnose_topk4.py`
- 结果：`results/batch/`、`results/batch_amp050/`、`results/batch_amp200/`、
  `results/batch_seed2/`、`results/*_summary.json`
- 图：`dhc6_blog_assets/tau_e_eval.svg`、`tau_e_amplitude.svg`、
  `tau_e_seed.svg`、`topk4_diagnostic.svg`
- 网页：<http://127.0.0.1:8765/dhc6_report.html#s615>

## 明早建议优先看

1. 6.1.5 表 10：主评价；
2. 6.1.6 表 11/12：幅值与种子稳健性；
3. 图 22：Top-k SAE 为什么不能给 τe；
4. 如果时间允许，下一步是**第三种子**或**真实作动器/滤波器变体**，而不是继续调拟合门限。
