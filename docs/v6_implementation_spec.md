# V6 — Incremental-action Student (implementation spec)

## 1. Problem being fixed

v4 distillation (own-delta convention) produces a Student whose requested-force
total variation is too high relative to the Teacher:

```
TV_S / TV_T = 2.133  >  quality gate 1.25
```

The v4 loss `teacher_action_rate_mse` penalizes the Student's own
step-to-step delta against the Teacher's own step-to-step delta:

```
(u^S_t − u^S_{t−1}) ≈ (u^T_t − u^T_{t−1})        # v4 own-delta
```

This lets the Student match the Teacher's *increment* while sliding its
*absolute* force around a drifting baseline, inflating total variation.
v5 (a slew-rate clamp) fixed the TV ratio but reintroduced doublet-reversal
tracking degradation, so it was reverted.

**V6 changes the action head itself**: the Student emits a *residual delta*
rather than a full action.

```
Δu_t = π^Δ(o_t, θ, u_{t−1})
u_t  = clip(u_{t−1} + Δu_t, −1, 1)
```

and the distillation label becomes the residual against the Student's actual
previous action:

```
label_t = u^T_t − u_{t−1}          # residual convention
```

During round-zero teacher-forcing `u_{t−1} = u^T_{t−1}` (driver = Teacher), so
`label_t` coincides with the own-delta. During Student-driven DAgger rounds
`u_{t−1} = u^S_{t−1}` (driver = Student), so `label_t` is a true residual. The
dataset already stores `driver_actions` and `previous_driver_actions`, so **no
new data collection is required** — only relabelling.

## 2. Two corrections vs. the original V6 plan

1. **`u_{t−1}` is the *requested* action, not the rate-limited *applied* action.**
   The controller's own previous *requested* force is already in the observation
   (`previous_force_normalized`, index 6) and in the dataset
   (`driver_actions` / `previous_driver_actions`), so it is observable, deployable,
   and present in the data. The 88 N/s actuator rate limit
   (`force_rate_limit_n_s=88.0`) remains a uniform environment-layer backstop and
   is *not* what the incremental head conditions on.

2. **V6 requires stride-1 collection.** `previous` must be exactly one policy
   step (`policy_step_delta == 1`) so `u_{t−1}` is the immediately preceding
   requested action. Set `--initial-sample-stride 1 --student-sample-stride 1`.
   The dataset's `previous_driver_actions` self-references at episode boundaries;
   V6 zeroes these boundary predecessors (`incremental_previous_actions`) to
   match the deployed policy's reset state `u_{t−1} = 0` at episode start.

## 3. Odd-policy symmetry (incremental form)

The base dense Student enforces `π(o) = −π(−o)`. For the incremental head the
symmetry must mirror both the observation *and* the previous action:

```
delta(o, u) = −delta(−o, −u)      ⇒      u_t = clip(u_{t−1} + delta(o, u_{t−1}))
```

`IncrementalDenseStudent` implements this by antisymmetrizing the delta logits:

```
delta_logits = 0.5 * (f(o, θ, u) − f(−o, θ, −u))
delta         = delta_scale * tanh(delta_logits)
action        = clip(prev + delta, −1, 1)
```

## 4. Losses

`incremental_student_losses` (in `src/distillation/losses.py`) returns three
per-row-weighted terms:

- `action_mse`  = weighted MSE of `action_t` vs `teacher_action` (L_u)
- `delta_mse`   = weighted MSE of `delta_t` vs `residual_delta_label` (L_du)
- `excess`      = `clamp(|delta_t| − |label_t| − margin, min=0)²` (L_excess),
  penalizing only Student deltas that overshoot the Teacher's necessary
  magnitude by more than `excess_margin`.

Training loss (incremental branch of `distill.py`):

```
L = action_mse + incremental_delta_weight * delta_mse + excess_weight * excess
```

The validation objective mirrors this with `validation["excess"]`.

## 5. File map (all edits on `main` @ 52fc0e63)

| File | Change |
|------|--------|
| `src/student/dense/network.py` | `_small_head` + `IncrementalDenseStudent` (residual head, odd symmetry, `clip(prev+delta)`) |
| `src/student/dense/policy.py` | `IncrementalDenseStudentPolicy` (`reset()` zeroes `_previous_action`; `predict()` stores it), `load_dense_student` branch for schema `incremental_dense_student_v1` |
| `src/distillation/losses.py` | `incremental_student_losses` |
| `src/distillation/dataset.py` | `incremental_previous_actions` (boundary-zeroed) + `residual_delta_labels`, exposed as `previous_driver_action` / `residual_delta_label` |
| `src/distillation/distill.py` | config fields (`delta_scale`, `incremental_delta_weight`, `excess_weight`, `excess_margin`), `_build_student_model` branch, training/validation objective branches, checkpoint payload `incremental_dense_student_v1` |
| `src/distillation/validate.py` | `imitation_metrics` incremental branch (delta vs residual label, `excess`), `evaluate_dense_student_bank` incremental policy |
| `src/distillation/student_driven.py` | policy `reset()` on env reset + incremental policy construction in student-driven collection |
| `scripts/34_distill_student_driven.py` | `--student-architecture incremental`, new CLI flags |
| `tests/test_v6_incremental_smoke.py` | gymnasium-free smoke test (forward + odd symmetry, dataset fields, losses) |
| `configs/distillation/v6_incremental.yaml` | contract record mirroring the CLI defaults |

## 6. How to run (GPU, executed by the user)

```
python scripts/34_distill_student_driven.py \
  --student-architecture incremental \
  --initial-sample-stride 1 \
  --student-sample-stride 1 \
  --output results/v6_incremental \
  --delta-scale 1.0 \
  --incremental-delta-weight 1.0 \
  --excess-weight 1.0 \
  --excess-margin 0.0
```

Then run E1 (single-seed comparison of V6 vs v4) and check:

- `student_teacher_requested_force_variation_ratio` ≤ 1.25
- no doublet-reversal degradation vs v4 (tracking RMSE not worse)

## 7. Verification status

- `python tests/test_v6_incremental_smoke.py` → **PASS**
- All edited modules byte-compile (`py_compile`) clean.
- Not yet run end-to-end (requires GPU + full Teacher Bank); this is the user's
  next step.
