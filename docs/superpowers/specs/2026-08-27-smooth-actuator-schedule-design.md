# Smooth Actuator SAC Diagnostic Design

## Scope

This is the second, explicitly non-formal SAC diagnostic.  It retains the persisted
IV-A manual plant bank, the existing reference model, a 22 N pilot step, and the
30% augmentation authority.  It does not claim GJB compliance, train MoE, or
change the aircraft/reference dynamics.

## Decision

Use an environment-level actuator model plus an explicit smoothness-oriented
reward, then compare the resulting controller with the existing short-run
single-plant SAC result.  This is preferred over post-filtering the learned
action: the policy must learn within the actuator limitations rather than seeing
an unrealistically instantaneous action during training.

Alternatives considered:

1. Reward-only tuning leaves the policy trained against an unrealistically direct
   actuator and is therefore rejected.
2. Post-training filtering changes deployment behaviour without retraining and is
   therefore rejected.
3. Train with actuator lag and rate limits, and make jitter expensive.  This is
   selected because the training and evaluated systems remain identical.

## Actuator and observation model

At every 50 Hz control tick, the requested normalized action is clipped and
rate-limited relative to the prior commanded action.  The applied normalized
action then follows the commanded action through a first-order actuator:

`a_applied[k+1] = a_applied[k] + (1 - exp(-dt/tau_a)) * (a_command[k+1] - a_applied[k])`

where `dt = 0.02 s` and the exploratory default is `tau_a = 0.08 s`.  The plant
receives `Delta F_RL = a_applied * correction_ratio * pilot_force_scale_n`.
The observation exposes both prior command and applied action, so actuator state
is not hidden from the policy.

## Reward

The existing normalized reference-tracking error remains the primary term.  The
diagnostic reward additionally penalizes command energy, command change, applied
action change, and—only after the step has had 1.0 s to develop—residual tracking
error.  Initial coefficients are deliberately recorded as experimental settings:

- command energy: `0.05`
- command-change penalty: `0.75`
- applied-action-change penalty: `0.15`
- post-1.0 s tracking-error multiplier: `0.50`

The action-change terms target the observed 50 Hz sawtooth; the late tracking
term prevents a low-frequency residual from being treated as free once the
transient is over.

## Evidence and acceptance criteria

The GPU script will train a new 40,000-step exploratory controller only after
local tests pass.  It will save a Chinese-labelled raw/reference/SAC plot and a
report for the training plant `train_core-0000` plus held-out plant `id_test-2100`.
The plot must show both commanded and applied augmentation.  The report must
include tracking RMSE, RMS augmentation, and action total variation for both
plants.

This iteration is considered informative—not formally successful—if it reports
all metrics without NaN/Inf.  A smoother action trace alone is not accepted as
an improvement unless tracking RMSE is also reported for the same rollouts.
