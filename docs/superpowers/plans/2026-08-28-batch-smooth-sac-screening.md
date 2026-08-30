# Batch Smooth SAC Screening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a reproducible eight-run GPU screening batch that separates extra optimization, training-plant diversity, and reward smoothness effects.

**Architecture:** Reward coefficients become an immutable configuration passed from `RollQualityEnv` into the reward function. Persisted library loading supports a deterministic collection of train records. The batch script trains each configuration sequentially and emits a JSON summary across six held-out records.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium, Stable-Baselines3 SAC, pytest, PyTorch CUDA.

---

### Task 1: Make reward weights configurable

**Files:**
- Modify: `src/envs/reward.py`
- Modify: `src/envs/p_channel_env.py`
- Modify: `tests/test_reward.py`

- [ ] **Step 1: Add a failing configurable-weight test**

```python
from src.envs.reward import RewardWeights, roll_quality_reward


def test_reward_uses_supplied_action_energy_weight() -> None:
    low_cost = RewardWeights(action_energy=0.01)
    high_cost = RewardWeights(action_energy=0.20)
    assert roll_quality_reward(0.0, 1.0, 0.0, weights=high_cost) < roll_quality_reward(0.0, 1.0, 0.0, weights=low_cost)
```

- [ ] **Step 2: Verify red**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_reward.py -q
```

Expected: import failure because `RewardWeights` is absent.

- [ ] **Step 3: Implement immutable reward configuration**

```python
@dataclass(frozen=True)
class RewardWeights:
    action_energy: float = 0.05
    command_delta: float = 0.75
    applied_delta: float = 0.15
    late_error: float = 0.50
```

Add `weights: RewardWeights = RewardWeights()` as a keyword-only argument to
`roll_quality_reward`, substitute its fields for the four current constants,
and pass `self.reward_weights` from `RollQualityEnv`. Add
`reward_weights: RewardWeights = RewardWeights()` to the environment
constructor.

- [ ] **Step 4: Verify green and commit**

Run the command from Step 2; expected: all reward tests pass.

```powershell
git add src/envs/reward.py src/envs/p_channel_env.py tests/test_reward.py
git commit -m "feat: configure SAC reward weights"
```

### Task 2: Build deterministic multi-plant environments

**Files:**
- Modify: `src/experiments/exploratory_sac.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Add a failing multi-plant loader test**

```python
def test_build_multi_env_cycles_through_requested_persisted_plants() -> None:
    root = Path(__file__).parents[1]
    env = build_multi_env(
        root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl",
        ["train_core-0000", "train_core-0001"],
        horizon_steps=8,
    )
    seen = {env.reset(seed=seed)[1]["plant_id"] for seed in range(8)}
    assert seen == {"train_core-0000", "train_core-0001"}
```

- [ ] **Step 2: Verify red**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_exploratory_training.py::test_build_multi_env_cycles_through_requested_persisted_plants -q
```

Expected: import failure because `build_multi_env` is absent.

- [ ] **Step 3: Implement `load_persisted_records` and `build_multi_env`**

`load_persisted_records(library_path, plant_ids)` must preserve the requested
order, construct `PlantRecord` objects using the existing eight P-channel
parameters, and raise `ValueError` listing unknown IDs. `build_multi_env` must
pass its record list, reward weights, actuator settings, and step pilot signal
to `RollQualityEnv`.

- [ ] **Step 4: Verify green and commit**

Run the command from Step 2; expected: `1 passed`.

```powershell
git add src/experiments/exploratory_sac.py tests/test_exploratory_training.py
git commit -m "feat: build multi-plant SAC environments"
```

### Task 3: Implement the sequential batch runner and summary contract

**Files:**
- Create: `scripts/08_run_gpu_sac_screening_batch.py`
- Modify: `src/experiments/exploratory_sac.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Add a failing aggregate-metric test**

```python
def test_summarize_held_out_metrics_reports_harm_rate_and_median_change() -> None:
    summary = summarize_held_out_metrics([
        {"raw": {"tracking_rmse": 0.01}, "sac": {"tracking_rmse": 0.02}},
        {"raw": {"tracking_rmse": 0.20}, "sac": {"tracking_rmse": 0.10}},
    ])
    assert summary["harm_rate"] == 0.5
    assert summary["median_rmse_change"] == pytest.approx(-0.045)
```

- [ ] **Step 2: Verify red**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_exploratory_training.py::test_summarize_held_out_metrics_reports_harm_rate_and_median_change -q
```

Expected: import failure because `summarize_held_out_metrics` is absent.

- [ ] **Step 3: Implement evaluation summary and GPU runner**

`summarize_held_out_metrics` computes SAC-minus-raw per-plant RMSE deltas,
their median, and the fraction strictly greater than zero. The runner uses
`train_core-0000` plus 15 deterministic evenly spaced `train_core` IDs; held-out
IDs are `id_test-2100` through `id_test-2105`; and creates the four
configuration dictionaries in the design document with the two prescribed
seeds. It trains one model at a time, writes each run report, and writes one
`results/GPU批量SAC筛选报告.json` containing all per-plant metrics and aggregate
values. It exits when CUDA is unavailable and never starts more than one job.

- [ ] **Step 4: Verify green, full suite, and commit**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_exploratory_training.py::test_summarize_held_out_metrics_reports_harm_rate_and_median_change -q
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest -q
```

Expected: the focused and complete test suites have zero failures.

```powershell
git add src/experiments/exploratory_sac.py scripts/08_run_gpu_sac_screening_batch.py tests/test_exploratory_training.py
git commit -m "feat: add GPU SAC screening batch"
```

### Task 4: Run and validate the authorized P4 batch

**Files:**
- Generated, ignored: `checkpoints/gpu_sac_screening_batch/`
- Generated, ignored: `results/GPU批量SAC筛选报告.json`

- [ ] **Step 1: Synchronize committed source and manual plant bank to P4**

Archive `HEAD` to `/root/pingmo`, copy the ignored manual `plants.jsonl`, and
write the source revision to `/root/pingmo/.source_revision`.

- [ ] **Step 2: Run one sequential GPU batch**

Run `/root/flight_rl_control/.venv/bin/python scripts/08_run_gpu_sac_screening_batch.py`.
Expected: eight finite run reports and one aggregate report; no concurrent GPU
processes.

- [ ] **Step 3: Synchronize and validate the report**

Copy the JSON locally, check every run has six held-out records and finite
metrics, and report the configuration ranking without claiming GJB success.
