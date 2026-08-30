# Batch Smooth SAC Screening Design

## Purpose

Determine whether the observed damage to a naturally good held-out plant is
primarily caused by insufficient optimization, a single-plant training
distribution, or the relative strength of action penalties. The experiment is
exploratory only and cannot establish GJB compliance or a deployable controller.

## Controlled batch

All runs use the persisted IV-A manual bank, 22 N pilot step, 30% authority,
0.08 s actuator lag, 50 Hz control, deterministic evaluation, and the same six
held-out plants. Each configuration is repeated with seeds `20260828` and
`20260829`.

| ID | Training plants | Reward weights | Steps | Question |
| --- | --- | --- | --- | --- |
| single-40k | train_core-0000 | current | 40,000 | repeatable baseline |
| single-120k | train_core-0000 | current | 120,000 | does more optimization help? |
| multi-40k | 16 deterministic train_core plants | current | 40,000 | does plant diversity help? |
| multi-strong-40k | same 16 plants | stronger action costs | 40,000 | do stronger costs prevent needless control? |

The current reward weights are action energy `0.05`, command change `0.75`,
applied change `0.15`, and late error `0.50`. Strong weights are `0.20`,
`1.50`, `0.30`, and `0.50`; tracking error remains unchanged.

## Evaluation and decision rules

Each trained policy is rolled out on the same six `id_test` records. The batch
records per-plant raw and SAC tracking RMSE, applied augmentation RMS, and
commanded-force total variation. It also reports the fraction of held-out
plants whose RMSE increases over raw (`harm_rate`).

The principal comparison is median RMSE change and harm rate; a lower effort is
not counted as improvement if median tracking worsens. `id_test-2100` is kept
as a known naturally-good regression case. No configuration is promoted to
formal training based on this batch alone.
