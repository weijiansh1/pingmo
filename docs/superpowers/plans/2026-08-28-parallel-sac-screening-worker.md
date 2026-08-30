# Parallel SAC Screening Worker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible CLI worker that executes one frozen SAC screening run, so the two final strong-penalty runs can proceed concurrently with the current sequential batch.

**Architecture:** Move batch configuration resolution and one-run execution into importable helpers in `08_run_gpu_sac_screening_batch.py`. Keep that script as the sequential summary producer. Add `09_run_gpu_sac_screening_worker.py` as a narrow CLI that executes exactly one ID and uses the same report format and checkpoint path.

**Tech Stack:** Python 3.10+, argparse, pathlib, pytest, NumPy, Stable-Baselines3 SAC, PyTorch CUDA.

---

### Task 1: Resolve a frozen run ID

**Files:**

- Modify: `scripts/08_run_gpu_sac_screening_batch.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Write the failing test**

```python
def test_screening_run_spec_resolves_frozen_strong_seed() -> None:
    module = _load_batch_module()
    run = module.resolve_screening_run("multi-strong-40k-seed-20260829", LIBRARY)

    assert run.run_id == "multi-strong-40k-seed-20260829"
    assert run.seed == 20260829
    assert run.timesteps == 40_000
    assert len(run.plant_ids) == 16
    assert run.reward_weights.action_energy == 0.20
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exploratory_training.py::test_screening_run_spec_resolves_frozen_strong_seed -q`

Expected: FAIL because `resolve_screening_run` is absent.

- [ ] **Step 3: Add the minimal resolver**

Add a frozen `ScreeningRun` dataclass and `resolve_screening_run(run_id, library)`. Move the existing four configurations and two seeds into `screening_configurations(library)` and `SCREENING_SEEDS`; resolve only IDs of the form `configuration-id-seed-YYYYMMDD`. Raise `ValueError("unknown screening run ID: ...")` for every other string. Update the sequential loop to use these helpers.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exploratory_training.py::test_screening_run_spec_resolves_frozen_strong_seed -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/08_run_gpu_sac_screening_batch.py tests/test_exploratory_training.py
git commit -m "feat: resolve frozen SAC screening runs"
```

### Task 2: Reuse the one-run execution path

**Files:**

- Modify: `scripts/08_run_gpu_sac_screening_batch.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Write the failing test**

```python
def test_execute_screening_run_returns_existing_report(tmp_path: Path) -> None:
    module = _load_batch_module()
    run = module.ScreeningRun(
        run_id="single-40k-seed-20260828",
        configuration_id="single-40k",
        seed=20260828,
        timesteps=40_000,
        plant_ids=["train_core-0000"],
        reward_weights=RewardWeights(),
    )
    run_dir = tmp_path / run.run_id
    run_dir.mkdir()
    (run_dir / "screening_report.json").write_text('{"run_id": "single-40k-seed-20260828"}', encoding="utf-8")

    report, skipped = module.execute_screening_run(run, LIBRARY, tmp_path)

    assert skipped is True
    assert report["run_id"] == run.run_id
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exploratory_training.py::test_execute_screening_run_returns_existing_report -q`

Expected: FAIL because `execute_screening_run` is absent.

- [ ] **Step 3: Add the minimal executor**

Implement `execute_screening_run(run, library, output_root) -> tuple[dict[str, object], bool]`. It must load and return an existing completed report with `True`; otherwise reuse the existing `train_short_experiment`, `SAC.load`, `_evaluate`, `summarize_held_out_metrics`, and JSON write. Preserve report keys `run_id`, `configuration_id`, `seed`, `timesteps`, `training_plant_ids`, `reward_weights`, `train_report`, `held_out`, and `held_out_summary`.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exploratory_training.py::test_execute_screening_run_returns_existing_report -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/08_run_gpu_sac_screening_batch.py tests/test_exploratory_training.py
git commit -m "feat: execute individual SAC screening runs"
```

### Task 3: Add the single-run CLI

**Files:**

- Create: `scripts/09_run_gpu_sac_screening_worker.py`
- Modify: `tests/test_exploratory_training.py`

- [ ] **Step 1: Write the failing test**

```python
def test_screening_worker_rejects_unknown_run_id() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/09_run_gpu_sac_screening_worker.py", "--run-id", "unknown"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unknown screening run ID" in result.stderr
```

- [ ] **Step 2: Verify RED**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exploratory_training.py::test_screening_worker_rejects_unknown_run_id -q`

Expected: FAIL because the worker file is absent.

- [ ] **Step 3: Add the CLI**

Implement `--run-id` with argparse, load the numbered batch script through `importlib.util.spec_from_file_location`, require CUDA, resolve one run, and call `execute_screening_run`. Print a JSON `run_skipped` or `run_finished` event containing the ID and held-out summary. Do not write the aggregate report.

- [ ] **Step 4: Verify GREEN**

Run: `.venv\\Scripts\\python.exe -m pytest tests/test_exploratory_training.py::test_screening_worker_rejects_unknown_run_id -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add scripts/09_run_gpu_sac_screening_worker.py tests/test_exploratory_training.py
git commit -m "feat: add parallel SAC screening worker"
```

### Task 4: Verify and launch only the two disjoint strong runs

**Files:**

- Generated, ignored: `checkpoints/gpu_sac_screening_batch/multi-strong-40k-seed-20260828/`
- Generated, ignored: `checkpoints/gpu_sac_screening_batch/multi-strong-40k-seed-20260829/`
- Generated, ignored: `logs/gpu_sac_screening_worker-20260828.log`
- Generated, ignored: `logs/gpu_sac_screening_worker-20260829.log`

- [ ] **Step 1: Run local verification**

Run: `.venv\\Scripts\\python.exe -m pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Push and sync the tested revision**

```powershell
git push origin feature/smooth-actuator-schedule
git archive --format=tar HEAD | ssh -i C:\\Users\\24307\\.ssh\\id_ed25519_flight_rl_p4 -p 50776 root@llrnwtdupzmu4jgwsnow.deepln.com "tar -xf - -C /root/pingmo"
```

Expected: the remote worker script exists and `.source_revision` records the pushed SHA.

- [ ] **Step 3: Launch the two workers**

```bash
nohup /root/pingmo/.venv/bin/python /root/pingmo/scripts/09_run_gpu_sac_screening_worker.py --run-id multi-strong-40k-seed-20260828 > /root/pingmo/logs/gpu_sac_screening_worker-20260828.log 2>&1 < /dev/null &
nohup /root/pingmo/.venv/bin/python /root/pingmo/scripts/09_run_gpu_sac_screening_worker.py --run-id multi-strong-40k-seed-20260829 > /root/pingmo/logs/gpu_sac_screening_worker-20260829.log 2>&1 < /dev/null &
```

Expected: two workers use distinct IDs, logs, and checkpoint directories while the sequential batch continues.

- [ ] **Step 4: Verify isolation and throughput**

Run: `nvidia-smi; tail -n 20 /root/pingmo/logs/gpu_sac_screening_worker-20260828.log; tail -n 20 /root/pingmo/logs/gpu_sac_screening_worker-20260829.log`

Expected: no CUDA errors, only the expected run ID per log, and the sequential runner later skips their completed reports.
