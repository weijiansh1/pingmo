# G0–G4 Multi-Command Privileged Teacher Design

## Goal

Determine whether the reference response is achievable under the existing augmentation limits, then train and compare constrained global Privileged MLP-SAC and MoE-SAC Teachers on a split-safe multi-command aircraft population.

## Dataset contract

The persisted 3000-aircraft library remains immutable. G0 is a read-only physics audit across every split. Only `train_core` and `train_boundary` may generate learning transitions; `validation` selects fixed training settings. `id_test`, `ood_test`, and `extreme_test` are evaluated exactly once after the settings are frozen.

## Command contract

Every command is a deterministic 50 Hz sequence with a 200 Hz held plant input. The initial suite is: positive and negative steps at 0.25, 0.50, and 1.00 of the nominal 22 N; equal-amplitude doublets; sine waves at 0.25, 0.50, and 1.00 Hz; and a 0.10–1.50 Hz chirp. The same command identifier and force history drive raw, reference, constrained oracle, training, and evaluation.

## G0: feasibility audit

For every aircraft-command pair, run raw, unconstrained oracle, and constrained oracle through the same `±6.6 N` augmentation, `4 s^-1` normalized slew bound, and `0.08 s` actuator lag used by the learner. Persist per-pair metrics and a feasibility label. Feasibility is diagnostic, not a learned label: it separates reference error that is physically unavoidable from error a controller could reasonably improve.

## G1: constrained environment and metrics

An episode samples one allowed training aircraft and one command profile. Reward remains reference tracking but action magnitude, command total variation, applied total variation, and saturation are first-class diagnostic costs. Evaluation records raw/reference/controller/oracle metrics by command and aircraft split; no test result is passed back to training.

## G2: global MLP baseline

Train a parameter-conditioned, two-stream Privileged SAC on the train splits only. Its actor receives the deployable signal/history plus privileged theta for this Teacher experiment; its critics receive the full Markov simulator state. A validation-only configuration choice is frozen before final tests. GPU runs are independent `(controller, seed, configuration)` jobs with unique manifests, checkpoints, reports, and resume markers.

## G3: MoE comparison

Train a four-expert theta-routed MoE Teacher with the exact same data, command distribution, action limits, update budget, evaluation code, and seeds as G2. Record expert utilization and balance loss. MoE is a capacity comparison, not a mechanism for changing the control objective.

## G4: decision report

Report every split separately, with raw/oracle/MLP/MoE comparisons. A candidate is not eligible for formal claims unless it improves the feasible subset over raw across repeated seeds, keeps total variation and saturation within the configured budget, avoids harming near-reference aircraft, and has no expert collapse. The report explicitly remains an exploratory research result, not a GJB compliance conclusion.

## Parallel execution

CPU-bound G0 simulation is sharded across cores on the GPU host. GPU learning jobs are sharded by unique run ID; a dispatcher starts a measured safe concurrency after a short throughput probe and never allows shared output directories. The final aggregator reads only completed reports.
