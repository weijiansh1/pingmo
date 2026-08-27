# Privileged Critic Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the custom SAC Teacher use deployment observations for its actor and full simulator states for its critics, then verify the corrected path locally before a bounded P4 run.

**Architecture:** The environment emits both actor and critic state at reset/step.  A small NumPy replay buffer stores the two streams.  PyTorch actor and twin target critics consume their respective streams; the fixed-plant trainer owns collection, optimization, evaluation and checkpoints.

**Tech Stack:** Python 3.12/3.10, NumPy, PyTorch, Gymnasium, pytest.

---

### Task 1: Expose a fixed-width privileged environment state

**Files:** Modify `src/aircraft/delay.py`, `src/aircraft/p_channel.py`, `src/aircraft/reference.py`, `src/envs/p_channel_env.py`; test `tests/test_env.py`.

- [ ] Write tests requiring reset and step info to contain the same finite `critic_state`, and requiring the state to include the plant and reference FIFO histories.
- [ ] Run `pytest tests/test_env.py -q` and observe the missing/reset-state failure.
- [ ] Add read-only padded FIFO state accessors and build the fixed-width critic state from actor state, plant state/history and reference state/history.
- [ ] Re-run `pytest tests/test_env.py -q` and observe success.

### Task 2: Add two-stream replay and privileged SAC

**Files:** Create `src/teacher/sac/replay.py`; modify `src/teacher/sac/actor.py`, `src/teacher/sac/critic.py`, `src/teacher/sac/teacher.py`; test `tests/test_teacher.py`.

- [ ] Write tests requiring replay samples to retain both state streams and requiring Q output to change when only privileged state changes, while actor actions do not.
- [ ] Run `pytest tests/test_teacher.py -q` and observe failures for missing APIs.
- [ ] Implement replay, squashed Gaussian actor, twin Q/target critics, Polyak updates and a two-stream update interface.
- [ ] Re-run `pytest tests/test_teacher.py -q` and observe success.

### Task 3: Build a bounded corrected fixed-plant training path

**Files:** Create `src/experiments/privileged_sac.py`; modify `scripts/10_train_sac_teacher.py`; test `tests/test_exploratory_training.py`.

- [ ] Write a test requiring a tiny CPU run to collect transitions, perform updates, save a checkpoint and report actor/critic state dimensions.
- [ ] Run the test and observe the missing trainer failure.
- [ ] Implement fixed-plant collection, seeded replay sampling, deterministic action evaluation and checkpoint/report persistence.
- [ ] Run the focused test, then `pytest -q`.

### Task 4: Local smoke and bounded P4 experiment

**Files:** Modify `scripts/97_exploratory_gpu_sac.py`; artifacts under `checkpoints/`, `results/`, `img/`.

- [ ] Run the corrected CPU smoke and record deterministic reward/action diagnostics.
- [ ] Copy only source, script and the persisted library to P4; verify the remote script imports the custom learner.
- [ ] Run the bounded fixed-plant P4 training and reproduce raw/reference/oracle/SAC curves with the two-stream checkpoint.
