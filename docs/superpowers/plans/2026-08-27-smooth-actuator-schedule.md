# Smooth Actuator SAC Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a non-formal SAC controller whose actions pass through a visible first-order actuator and whose reward makes high-frequency command changes costly.

**Architecture:** `RollQualityEnv` will maintain distinct commanded and applied normalized actuator states.  The applied state drives the P-channel, while both states become part of the policy observation and the environment diagnostics.  The reward accepts optional applied-action and late-step-error terms while retaining backwards-compatible defaults.  The GPU diagnostic will report matched raw/reference/SAC traces for a training plant and a held-out plant.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 SAC, PyTorch CUDA, pytest, Matplotlib.

---

### Task 1: Extend the dense reward with explicit smoothness terms

**Files:**
- Modify: `src/envs/reward.py`
- Modify: `tests/test_reward.py`

- [ ] **Step 1: Write the failing reward tests**

```python
def test_reward_penalizes_applied_action_motion_and_late_error() -> None:
    quiet = roll_quality_reward(
        error=0.0, action=0.0, action_delta=0.0,
        applied_action_delta=0.0, late_error=0.0,
    )
    chattering = roll_quality_reward(
        error=0.0, action=0.0, action_delta=0.2,
        applied_action_delta=0.1, late_error=0.0,
    )
    residual = roll_quality_reward(
        error=0.0, action=0.0, action_delta=0.0,
        applied_action_delta=0.0, late_error=0.3,
    )
    assert quiet > chattering
    assert quiet > residual
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_reward.py -q
```

Expected: `TypeError` because `applied_action_delta` and `late_error` do not yet exist.

- [ ] **Step 3: Implement the minimal backwards-compatible reward API**

Replace the reward implementation with:

```python
"""Dense tracking reward for non-formal control diagnostics."""


def roll_quality_reward(
    error: float,
    action: float,
    action_delta: float,
    constraint: float = 0.0,
    applied_action_delta: float = 0.0,
    late_error: float = 0.0,
) -> float:
    return -(
        error * error
        + 0.05 * action * action
        + 0.75 * action_delta * action_delta
        + 0.15 * applied_action_delta * applied_action_delta
        + 0.50 * late_error * late_error
        + constraint
    )
```

- [ ] **Step 4: Run the reward tests to verify they pass**

Run the command from Step 2.  Expected: `2 passed`.

- [ ] **Step 5: Commit the reward contract**

```powershell
git add src/envs/reward.py tests/test_reward.py
git commit -m "feat: penalize SAC action chatter"
```

### Task 2: Model command, actuator lag, and applied effort in the environment

**Files:**
- Modify: `src/envs/p_channel_env.py`
- Modify: `tests/test_env.py`

- [ ] **Step 1: Write the failing actuator behaviour test**

Add this test to `tests/test_env.py`:

```python
def test_environment_exposes_rate_limited_command_and_lagged_applied_action() -> None:
    env = RollQualityEnv(
        generate_plant_library(19, {"train_core": 1}),
        horizon_steps=3,
        pilot_signal="step",
        correction_ratio=0.3,
        pilot_force_scale_n=22.0,
        normalized_rate_limit_s_inv=4.0,
        actuator_time_constant_s=0.08,
    )
    observation, _ = env.reset(seed=19)
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))

    rate_limited_command = 4.0 * 0.02
    expected_applied = (1.0 - np.exp(-0.02 / 0.08)) * rate_limited_command
    assert observation.shape == (142,)
    assert info["commanded_action"] == pytest.approx(rate_limited_command)
    assert info["applied_action"] == pytest.approx(expected_applied)
    assert info["delta_f"] == pytest.approx(expected_applied * 0.3 * 22.0)
    assert info["command_delta"] == pytest.approx(rate_limited_command)
    assert info["applied_action_delta"] == pytest.approx(expected_applied)
```

Update the previous 141-dimensional observation expectations in this file to 142.

- [ ] **Step 2: Run the environment tests to verify the new test fails**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_env.py -q
```

Expected: `TypeError` because `actuator_time_constant_s` is not accepted.

- [ ] **Step 3: Add actuator state and use it to drive the plant**

In `RollQualityEnv.__init__`, add `actuator_time_constant_s: float = 0.08`, validate it is positive, save it, expand `observation_space` to `6 + history_steps * 4 + 8`, and set `_actuator_alpha = 1.0 - math.exp(-self._action_dt / actuator_time_constant_s)`.

In `reset`, reset both states:

```python
self._previous_commanded_action = 0.0
self._applied_action = 0.0
```

Replace the direct-action section in `step` with:

```python
requested_action = float(np.clip(np.asarray(action, dtype=float)[0], -self.action_limit, self.action_limit))
previous_command = self._previous_commanded_action
previous_applied = self._applied_action
max_action_increment = self.normalized_rate_limit_s_inv * self._action_dt
commanded_action = float(np.clip(
    requested_action,
    previous_command - max_action_increment,
    previous_command + max_action_increment,
))
applied_action = previous_applied + self._actuator_alpha * (commanded_action - previous_applied)
command_delta = commanded_action - previous_command
applied_action_delta = applied_action - previous_applied
delta = applied_action * self.correction_ratio * self.pilot_force_scale_n
previous_delta = previous_applied * self.correction_ratio * self.pilot_force_scale_n
```

Use `applied_action` in `delta`, `f_eq`, and the history.  Use `commanded_action`, `command_delta`, `applied_action_delta`, and `late_error` in `roll_quality_reward`, where `late_error` equals normalized error only for `pilot_signal == "step" and self._episode_step * self._action_dt >= 1.0`, otherwise zero.  Save the two state variables before incrementing the episode and expose `commanded_action`, `applied_action`, `command_delta`, and `applied_action_delta` in the step info.

Change `_observation` so its final current-state fields are command force and applied force:

```python
current = np.array([
    self._f_pilot, self._p, self._p_ref, error,
    self._previous_commanded_action * self.correction_ratio * self.pilot_force_scale_n,
    self._applied_action * self.correction_ratio * self.pilot_force_scale_n,
], dtype=np.float32)
```

- [ ] **Step 4: Run all environment tests to verify they pass**

Run the command from Step 2.  Expected: `4 passed`.

- [ ] **Step 5: Commit the environment model**

```powershell
git add src/envs/p_channel_env.py tests/test_env.py
git commit -m "feat: model lagged SAC actuator"
```

### Task 3: Retain commanded and applied effort in the response diagnostic

**Files:**
- Modify: `src/experiments/exploratory_sac.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Write the failing trace test**

Extend `test_collect_response_trace_records_raw_reference_and_control_effort`:

```python
assert trace["commanded_delta_f"].shape == (8,)
assert np.allclose(trace["commanded_delta_f"], 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_exploratory_training.py::test_collect_response_trace_records_raw_reference_and_control_effort -q
```

Expected: `KeyError: 'commanded_delta_f'`.

- [ ] **Step 3: Store command effort and show it separately from applied effort**

Add `commanded_delta_f` to the trace arrays and append:

```python
traces["commanded_delta_f"].append(
    float(info["commanded_action"]) * env.correction_ratio * env.pilot_force_scale_n
)
```

Update `save_response_comparison` so the lower panel contains a dashed gray
`ΔF_命令` line and a purple `ΔF_RL（实际）` line.  Keep raw, ref, and SAC as
the three response curves in the upper panel and preserve Chinese labels.

- [ ] **Step 4: Run the trace test to verify it passes**

Run the command from Step 2.  Expected: `1 passed`.

- [ ] **Step 5: Commit the diagnostic change**

```powershell
git add src/experiments/exploratory_sac.py tests/test_exploratory_training.py
git commit -m "feat: plot SAC command and applied effort"
```

### Task 4: Produce matched training-plant and held-out GPU reports

**Files:**
- Modify: `scripts/07_run_gpu_exploratory_sac.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Write the failing report-helper test**

Add a pure helper test that evaluates the test plant without training:

```python
def test_response_metrics_include_tracking_and_effort() -> None:
    trace = {
        "p": np.array([0.0, 1.0]),
        "p_ref": np.array([0.0, 0.0]),
        "delta_f": np.array([0.0, 3.0]),
        "commanded_delta_f": np.array([0.0, 4.0]),
    }
    metrics = response_metrics(trace)
    assert metrics == {
        "tracking_rmse": 2 ** -0.5,
        "applied_delta_f_rms_n": 3 / 2 ** 0.5,
        "commanded_delta_f_total_variation_n": 4.0,
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_exploratory_training.py::test_response_metrics_include_tracking_and_effort -q
```

Expected: import failure because `response_metrics` does not exist.

- [ ] **Step 3: Implement `response_metrics` and use it for both plants**

Define this helper in `src/experiments/exploratory_sac.py`:

```python
def response_metrics(trace: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "tracking_rmse": float(np.sqrt(np.mean(np.square(trace["p"] - trace["p_ref"]))),),
        "applied_delta_f_rms_n": float(np.sqrt(np.mean(np.square(trace["delta_f"]))),),
        "commanded_delta_f_total_variation_n": float(np.sum(np.abs(np.diff(trace["commanded_delta_f"]))),),
    }
```

In `scripts/07_run_gpu_exploratory_sac.py`, retain `train_core-0000`, add
`held_out_plant_id = "id_test-2100"`, and roll out the trained model against
that held-out record.  Save two named plots and report nested `train` and
`held_out` metric dictionaries.  The script must keep the 40,000-step setting,
the IV-A manual plant bank, fixed seed `20260827`, `correction_ratio = 0.3`, and
the CUDA availability check.

- [ ] **Step 4: Run the helper test and the full local suite**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest tests/test_exploratory_training.py::test_response_metrics_include_tracking_and_effort -q
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' -m pytest -q
```

Expected: helper test passes, then the full suite has zero failures.

- [ ] **Step 5: Commit the reproducible GPU diagnostic**

```powershell
git add src/experiments/exploratory_sac.py scripts/07_run_gpu_exploratory_sac.py tests/test_exploratory_training.py
git commit -m "feat: evaluate smooth SAC on held-out plant"
```

### Task 5: Smoke test locally and run the authorized GPU diagnostic

**Files:**
- Generated, ignored locally: `checkpoints/gpu_sac_smooth_train_core_0000/`
- Generated, ignored locally: `results/GPU平滑SAC_训练报告.json`
- Generated, ignored locally: `img/GPU平滑SAC_训练机阶跃响应.png`
- Generated, ignored locally: `img/GPU平滑SAC_留出机阶跃响应.png`

- [ ] **Step 1: Run the local smoke test**

Run:

```powershell
& 'C:\Users\24307\Desktop\品模py\.venv\Scripts\python.exe' scripts\99_local_smoke.py
```

Expected: exit code 0 and a smoke report without NaN/Inf.

- [ ] **Step 2: Copy the committed source and manual IV-A bank to the P4 host**

Run `git archive HEAD` through SSH to `/root/pingmo`, then copy the ignored
`p_channel_library_iv_a_manual_v1/plants.jsonl` file.  Record the source commit
in `/root/pingmo/.source_revision`.

- [ ] **Step 3: Run the GPU script and inspect its JSON output**

Run:

```powershell
ssh flight-rl-p4 'cd /root/pingmo && .venv/bin/python scripts/07_run_gpu_exploratory_sac.py'
```

Expected: CUDA identifies the Tesla P4; report contains finite `train` and
`held_out` tracking/effort metrics; checkpoint and two plots exist.

- [ ] **Step 4: Synchronize the ignored report and plots back to the local workspace**

Copy them into `results/` and `img/`, then inspect each PNG locally to ensure
Chinese text is rendered by the local font configuration.

- [ ] **Step 5: Commit no generated artifacts and report the comparison honestly**

Run `git status --short`; it must contain no source changes.  Report both
training and held-out metrics, identify any loss of tracking, and explicitly
call out remaining command or applied-action chatter.  Do not claim GJB
compliance or a completed global/MoE controller.
