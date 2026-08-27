# G0–G4 Multi-Command Privileged Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit constrained reference feasibility across the split 3000-aircraft library, then train and compare globally conditioned constrained MLP-SAC and MoE-SAC Teachers on a shared multi-command task.

**Architecture:** A reusable command-profile module drives Oracle, environment, training, and evaluator paths. G0 writes only diagnostics for all splits. G1 adds command sampling and explicit costs. G2 and G3 use a common global two-stream trainer plus different actor classes. A manifest-driven dispatcher runs independent seeds in parallel; G4 aggregates completed reports without training on test splits.

**Tech Stack:** Python, NumPy, SciPy, Gymnasium, PyTorch, pytest, JSONL, CUDA.

---

### Task 1: G0 command profiles and constrained-oracle audit

**Files:**

- Create: `src/envs/commands.py`
- Modify: `src/experiments/reference_oracle_check.py`
- Create: `src/experiments/feasibility_audit.py`
- Create: `scripts/13_run_g0_feasibility_audit.py`
- Modify: `tests/test_reference_oracle_check.py`
- Create: `tests/test_commands.py`

- [ ] Write failing tests that a command profile has deterministic 50 Hz samples, doublet sign reversal, and an oracle trace follows supplied samples rather than a hard-coded 22 N step.
- [ ] Implement immutable command profiles for signed amplitude steps, doublets, sine waves, and chirps; expose `samples(action_dt, duration_s)`.
- [ ] Change `simulate_reference_oracle` to accept a profile and apply the existing action rate/magnitude contract plus the same first-order actuator lag as the environment.
- [ ] Implement the all-split audit as a deterministic shardable map over `(plant_id, command_id)` pairs. Save per-pair raw/oracle/constrained RMSE, saturation, total variation, and feasibility fields; never modify the plant library.
- [ ] Run the targeted tests, then `pytest -q`; commit `feat: audit constrained oracle feasibility`.
- [ ] Submit the CPU-sharded audit to the GPU host and retrieve the immutable G0 JSONL report and summary.

### Task 2: G1 multi-command environment and split-safe evaluation

**Files:**

- Modify: `src/envs/p_channel_env.py`
- Modify: `src/experiments/exploratory_sac.py`
- Create: `src/experiments/teacher_evaluation.py`
- Modify: `tests/test_env.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] Write failing tests proving seeded reset picks a reproducible command profile, pilot input changes during doublet/sine episodes, and reports retain command ID and physical force metrics.
- [ ] Replace the `"step" | "sine"` pilot-signal switch with a supplied command profile while retaining compatibility wrappers for existing smoke tests.
- [ ] Add explicit episode diagnostics for command and applied total variation, augmentation saturation, raw/reference/oracle/controller tracking error, and feasible-subset membership.
- [ ] Implement split-safe evaluator input validation: training accepts only train splits; validation selection accepts only validation; final evaluator rejects any attempt to tune from ID/OOD/extreme metrics.
- [ ] Run focused and full tests; commit `feat: add split-safe multi-command evaluation`.

### Task 3: G2 global constrained Privileged MLP-SAC

**Files:**

- Modify: `src/experiments/privileged_sac.py`
- Modify: `src/teacher/sac/teacher.py`
- Create: `scripts/14_train_global_privileged_sac.py`
- Create: `scripts/15_evaluate_global_teacher.py`
- Modify: `tests/test_exploratory_training.py`
- Modify: `tests/test_teacher.py`

- [ ] Write failing tests for a global replay stream that observes more than one `train_core`/`train_boundary` plant and command profile, emits a checkpoint, and evaluates deterministically without a test split.
- [ ] Generalize the fixed-plant trainer into a global trainer that samples only named training IDs, uses the two-stream replay state, moves all tensors and modules to the requested device, and writes a run manifest with library hash, command-suite hash, seed, split list, and source revision.
- [ ] Make action variation and saturation constraints configurable from the frozen validation-only configuration; persist raw/oracle/controller validation metrics.
- [ ] Run local smoke tests, then dispatch at least two independent MLP seeds to the P4 with unique run directories and resumes.
- [ ] Select one configuration using validation only; freeze its manifest and commit `feat: train global constrained privileged SAC`.

### Task 4: G3 fair global MoE-SAC comparison

**Files:**

- Modify: `src/teacher/moe/teacher.py`
- Create: `src/experiments/moe_sac.py`
- Replace: `scripts/11_train_moe_teacher.py`
- Modify: `tests/test_teacher.py`

- [ ] Write failing tests that MoE training runs on the requested device, returns action samples, records finite actor/critic/balance losses, and reports nonempty router weights.
- [ ] Add device-aware `act` and update operations to `MoETeacher`; use the same global trainer contract, replay, command suite, and seed manifest as G2.
- [ ] Write the GPU MoE runner and launch the same seed count and frozen configuration as MLP with unique output directories.
- [ ] Verify router-use entropy, maximum mean route weight, total variation, saturation, and held-out metrics are recorded for every completed run.
- [ ] Run full tests; commit `feat: compare global constrained MoE SAC`.

### Task 5: G4 locked evaluation and decision report

**Files:**

- Create: `scripts/16_run_g4_locked_evaluation.py`
- Create: `src/experiments/g4_report.py`
- Create: `tests/test_g4_report.py`
- Generated, ignored: `results/G0-G4_受限多指令Teacher报告.json`
- Generated, ignored: `img/G0-G4_受限多指令Teacher对比.png`

- [ ] Write failing tests that the aggregator rejects incomplete manifests, keeps ID/OOD/extreme separate, and never merges validation metrics into test metrics.
- [ ] Implement the report to compare raw, constrained oracle, MLP, and MoE by split, command, seed, and oracle-feasibility label; compute confidence intervals across seeds and report harm rate, RMSE change, total variation, saturation, and router diagnostics.
- [ ] Generate Chinese plots for command-wise and split-wise comparisons and label the report exploratory/non-formal.
- [ ] Dispatch independent evaluation jobs in parallel on the P4; retrieve artifacts locally; run full tests and commit `feat: report constrained multi-command teacher results`.
