# GJB Manual v1 Alignment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the provisional research prototype with the G0–G3-compatible IV-A diagnostic path defined by `GJB_MoE_SAC_滚转品质实验手册_v1.md`; no global or MoE training occurs until the single-plant gate passes.

**Architecture:** Plant records are sampled in response coordinates using `S_1s` and numerically calibrated to `L_Fa`.  The diagnostic reference preserves each raw plant's sensitivity and physical delay while removing only pole-zero mismatch.  A constrained oracle, environment, and evaluator use the same force authority, rate limit, and waveform sequence.

**Tech Stack:** Python, NumPy, SciPy, PyTorch, Gymnasium, pytest.

---

### Task 1: Freeze IV-A profile and invalidate provisional artifacts

**Files:** Create `data/gjb_roll_spec.yaml`, `docs/manual_alignment_status.md`; modify `README.md`; test `tests/test_config.py`.

- [x] Add profile constants: IV-A, 22 N pilot scale, A18/A19/A21/A31 values, model version `GJB_s2_original`, and explicit `A116=not_digitized` status.
- [x] Mark every existing `*_v2_stratified` library and exploratory GPU checkpoint as provisional/invalid for GJB conclusions.
- [x] Test that the profile loader exposes the IV-A action-scale constants and refuses unknown profiles.

### Task 2: Implement response-level gain calibration

**Files:** Create `src/aircraft/gain_calibration.py`; modify `parameters.py`, `sampler.py`; test `tests/test_gain_calibration.py`.

- [x] Write failing tests showing that a target `S_1s` in degree/N is met to 0.5%, and that doubling `L_Fa` doubles measured sensitivity.
- [x] Simulate a 1 N step at 200 Hz, integrate `p` to bank angle, convert radian to degree, and solve `L_Fa` by linear scaling.
- [x] Replace direct `L_Fa` proposals with IV-A `S_1s` proposals and persist both target/measured sensitivity plus profile provenance.

### Task 3: Correct diagnostic reference and constrained oracle

**Files:** Modify `src/aircraft/reference.py`; create `src/aircraft/constrained_oracle.py`; test `tests/test_reference.py`, `tests/test_constrained_oracle.py`.

- [x] Write failing tests proving `R_omega=R_zeta=1`, retained delay, and reference `S_1s=raw S_1s`.
- [x] Implement diagnostic reference gain calibration and force/rate limited oracle with the same 50 Hz hold as RL.
- [x] Test that the oracle cannot exceed force or rate limits and reports its achievable objective.

### Task 4: Align actions, observations, excitation, and reward

**Files:** Modify `src/envs/p_channel_env.py`, `reward.py`, `src/experiments/privileged_sac.py`; create `src/envs/excitation.py`; test `tests/test_env.py`, `tests/test_reward.py`.

- [ ] Use force units: pilot amplitudes are sampled from 5.5/11/16.5/22 N and augmentation is `rho*22 N`.
- [x] Enforce normalized augmentation rate limit 4/s and expose `F_pilot`, `F_eq`, `p_ref`, tracking error, history, cancel index and saturation diagnostics.
- [ ] Begin with the required single step smoke, then add the formal excitation mixture only after G3 passes.

### Task 5: Rebuild IV-A plant bank and G0–G2 reports

**Files:** Modify `scripts/01_generate_aircraft.py`, `scripts/00_check_p_channel.py`; create `scripts/05_verify_reference_oracle.py`; tests for split counts and provenance.

- [x] Generate 16,384 candidates and select the 3,000 documented split sizes without ID/OOD leakage.
- [ ] Produce raw S1s, delay, and oscillation diagnostics.  Leave A116 level as `not_available`, never a fabricated threshold.
- [x] Produce a same-input raw/reference/constrained-oracle four-line report and stop if authority cannot attain the reference.

### Task 6: Re-run G3 only after gates pass

**Files:** Modify `scripts/10_train_sac_teacher.py`, `scripts/98_privileged_gpu_sac.py`, evaluation scripts; tests for manifest and four-line outputs.

- [ ] Start fresh weights/replay, run 0.1–0.3 M single-plant step diagnostic on P4.
- [ ] Select only from Validation after later global training; ID/OOD/Stress remain sealed.
- [ ] Require reference tracking, oracle-gap, rate/saturation and cancellation evidence before allowing SAC-1/SAC-2/MoE.
