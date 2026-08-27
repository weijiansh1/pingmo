# Privileged MoE-SAC Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and locally smoke-test the parameterized P-channel through a single Global Privileged MoE-SAC Teacher, without Student, distillation, transfer, GPU deployment, or long training.

**Architecture:** The project is a small Python package rooted at `src`, with deterministic dynamics/sampling/metrics below Gymnasium environment and PyTorch SAC components. All datasets and reports have explicit provenance and seeds. Training scripts use a CPU smoke profile and write to `results/` and `checkpoints/` only.

**Tech Stack:** Python 3.12, NumPy, SciPy, PyTorch, Gymnasium, PyYAML, pytest, Matplotlib.

---

## File map

* `src/aircraft/parameters.py`, `delay.py`, `p_channel.py`, `sampler.py`, `library.py`: plant contract, fractional delay, Sobol constrained sampling, persisted library.
* `src/benchmark/time_domain.py`, `gjb.py`, `evaluator.py`: response metrics and shared deterministic evaluation.
* `src/envs/commands.py`, `observation.py`, `reward.py`, `p_channel_env.py`: command/reference signals and 50 Hz RL environment.
* `src/teacher/sac/*`, `src/teacher/moe/*`: baseline and feature-mixture Teacher networks/training.
* `configs/gjb_roll.yaml`, `configs/*/*.yaml`: reproducible source, simulation, data, training, and smoke settings.
* `scripts/00_check_p_channel.py` through `scripts/20_evaluate_teacher.py`: cookbook entry points.
* `tests/test_*.py`: behavior-level tests; test-first for every production behavior.

### Task 1: Package and configuration foundation

**Files:** create `src/__init__.py`, `src/utils/config.py`; modify `pyproject.toml`, `requirements.txt`, `README.md`; test `tests/test_config.py`.

- [ ] Write a failing test asserting `load_yaml(path)` returns a mapping and rejects a non-mapping root.
- [ ] Run `pytest tests/test_config.py -v`; expected failure: `ModuleNotFoundError` or missing `load_yaml`.
- [ ] Implement `def load_yaml(path: Path) -> dict[str, Any]` with `yaml.safe_load`, a mapping check, and readable error messages.
- [ ] Add exact runtime dependencies: `numpy`, `scipy`, `torch`, `gymnasium`, `PyYAML`, `matplotlib`, `pytest`.
- [ ] Re-run `pytest tests/test_config.py -v`; expected: pass.

### Task 2: Parameter schema and fractional delay

**Files:** modify `src/aircraft/parameters.py`, `delay.py`; test `tests/test_delay.py`, `tests/test_p_channel.py`.

- [ ] Write failing tests for `PChannelParameters(...).derived()` and `FractionalDelay(dt=0.005, delay_s=0.0125).push(value)` proving derived omega/zeta and interpolation between the two preceding samples.
- [ ] Run the two tests; expected failure: missing classes.
- [ ] Implement frozen parameter dataclass, validation (`T_R>0`, `0<zeta_d<1`, `omega_d>0`, `lambda_s!=0`, `tau_p>=0`), and `FractionalDelay.reset/push` using a fixed FIFO and linear interpolation.
- [ ] Re-run `pytest tests/test_delay.py tests/test_p_channel.py -v`; expected: pass.

### Task 3: Exact P-channel simulation and response metrics

**Files:** modify `src/aircraft/p_channel.py`, `src/benchmark/time_domain.py`, `src/benchmark/gjb.py`; test `tests/test_p_channel.py`, `tests/test_metrics.py`.

- [ ] Write failing tests for: twice the input produces twice the no-saturation response; `R_omega=R_zeta=1` produces cancellation-compatible roll response; an inserted 12.5 ms delay shifts first response; `evaluate_roll_response` returns finite P1/P2/P3/ratio.
- [ ] Run `pytest tests/test_p_channel.py tests/test_metrics.py -v`; expected failure: missing simulator/metrics.
- [ ] Implement ZOH state-space realization of the specified fourth-order numerator and denominator; calculate p-dot from state output difference; integrate delay before plant input; implement peak sequence and settling calculations in one shared metrics module.
- [ ] Run the same test command and `python scripts/00_check_p_channel.py`; expected: tests pass and 1N/2N/5N report is finite.

### Task 4: Joint Sobol sampler and plant library

**Files:** modify `src/aircraft/sampler.py`, `library.py`, `configs/gjb_roll.yaml`; test `tests/test_aircraft_sampler.py`.

- [ ] Write failing tests that generate a small seeded library, assert exact split counts, deterministic IDs, non-independent derived quantities, and valid quality/source metadata.
- [ ] Run `pytest tests/test_aircraft_sampler.py -v`; expected failure: missing library generation.
- [ ] Implement scenario-first scrambled Sobol candidate generation, log sampling for T_R, rejection gates, computed `L_Fa`, and deterministic split allocation. Store JSONL/CSV metadata in `data/aircraft/generated/`.
- [ ] Run `python scripts/01_generate_aircraft.py --profile smoke` and `pytest tests/test_aircraft_sampler.py -v`; expected: generated library and passing tests.

### Task 5: Raw benchmark and reference model

**Files:** modify `src/benchmark/scenarios.py`, `evaluator.py`, `compare.py`; create `src/aircraft/reference.py`; test `tests/test_metrics.py`.

- [ ] Write failing tests that a reference model exposes the same 50 Hz simulation interface and raw/reference benchmark output contains NRMSE, IAE, ITAE, peak, Posc/Pav, action RMS, total variation, and saturation ratio.
- [ ] Run `pytest tests/test_metrics.py -v`; expected failure: missing reference/evaluator behavior.
- [ ] Implement target model matching with feasibility-limited action bounds and a shared evaluator; ensure validation selection never reads ID/OOD/extreme records.
- [ ] Run `python scripts/50_run_benchmark.py --profile smoke`; expected: timestamped JSON/CSV report with finite fields.

### Task 6: Gymnasium environment

**Files:** modify `src/envs/commands.py`, `observation.py`, `reward.py`, `p_channel_env.py`, `randomization.py`; test `tests/test_env.py`, `tests/test_reward.py`.

- [ ] Write failing tests for seeded reset, four plant substeps per action, action addition `F_pilot + delta_F`, privileged theta shape, full critic state shape, finite reward, and terminated/truncated behavior.
- [ ] Run `pytest tests/test_env.py tests/test_reward.py -v`; expected failure: missing `RollQualityEnv`.
- [ ] Implement deterministic pilot signal families, a feasible reference, normalized tracking/action/delta-action reward, and episode diagnostics without Posc/Pav dense reward.
- [ ] Run `pytest tests/test_env.py tests/test_reward.py -v`; expected: pass.

### Task 7: Baseline privileged SAC

**Files:** modify `src/teacher/sac/actor.py`, `critic.py`, `teacher.py`; test `tests/test_teacher.py`.

- [ ] Write failing tests that actor output is bounded to action space, twin critics return scalar Q values, one replay update has finite losses, and deterministic evaluation writes a checkpoint-compatible state dictionary.
- [ ] Run `pytest tests/test_teacher.py -v`; expected failure: missing SAC components.
- [ ] Implement replay buffer, squashed Gaussian actor, twin Q critic, target networks, entropy temperature, and fixed/global sampling trainers with explicit seeded CPU device selection.
- [ ] Run `python scripts/10_train_sac_teacher.py --profile smoke --mode fixed`; expected: finite loss lines and a local checkpoint.

### Task 8: Feature-level MoE-SAC Teacher

**Files:** modify `src/teacher/moe/router.py`, `expert.py`, `actor.py`, `critic.py`, `teacher.py`; test `tests/test_teacher.py`.

- [ ] Write failing tests for theta-only router weights summing to one, feature mixture shape, bounded Gaussian action, finite balance loss, and a smoke update exposing router usage.
- [ ] Run `pytest tests/test_teacher.py -v`; expected failure: missing MoE classes.
- [ ] Implement theta 8->64->32 encoder, router 32->64->E, state/history encoder, E feature experts, weighted feature mix, standard squashed Gaussian head, plain twin critics, and batch-average KL balance loss with 1e-3 default.
- [ ] Run `python scripts/11_train_moe_teacher.py --profile smoke --experts 4`; expected: finite losses, nonempty router use, and local checkpoint.

### Task 9: Curriculum, evaluation, and Cookbook

**Files:** modify `scripts/02_visualize_aircraft.py`, `03_split_aircraft_dataset.py`, `20_evaluate_teacher.py`, `52_run_gjb_evaluation.py`; create `Cookbook/README.md`, `Cookbook/01_gjb_extract.md` through `Cookbook/15_teacher_freeze.md`; test `tests/test_teacher.py`.

- [ ] Write failing tests for curriculum stages `[1,32,256,1200,1800]`, validation-only checkpoint selection, and an evaluation report that separates validation/ID/OOD/extreme partitions.
- [ ] Run `pytest tests/test_teacher.py -v`; expected failure: missing curriculum/evaluation report behavior.
- [ ] Implement continuation checkpoints, core/boundary weighted sampling, deterministic evaluation, router diagnostics, and CPU smoke cookbook commands. Document that formal curriculum and benchmarks require future explicit GPU authorization.
- [ ] Run `pytest -q`, then execute Cookbook smoke commands 02, 03, 04, 05, 06, 08, 09, 10, 11, and 14; expected: all tests pass and every command writes local artifacts.

## Review checklist

* The `s^2` numerator is retained and no sampled parameter crosses `lambda_s=0`.
* Delay is fractional, plant/RL rates are exactly 200/50 Hz, and action hold is four substeps.
* Sampler is scenario-first and preserves provenance, split, and quality labels.
* Actor receives theta; critic receives full state; router receives theta only; experts mix features, not actions.
* Student/distillation/transfer and remote/GPU execution are absent from runtime entry points.
* All completion claims are backed by a fresh full `pytest -q` run and logged smoke commands.
