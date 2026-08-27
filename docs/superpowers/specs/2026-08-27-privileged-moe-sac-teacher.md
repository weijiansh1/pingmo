# Privileged MoE-SAC Teacher Design

## Scope

This local phase ends after one globally conditioned, privileged MoE-SAC Teacher has been trained and benchmarked. It explicitly excludes Student policies, distillation, transfer, remote deployment, and GPU training.

## Dynamics contract

Each plant uses `theta = [L_Fa, lambda_s, T_R, zeta_d, omega_d, R_omega, R_zeta, tau_p]` and derives `omega_phi = R_omega * omega_d`, `zeta_phi = R_zeta * zeta_d`:

`p/F_as = L_Fa*s^2*(s^2+2*zeta_phi*omega_phi*s+omega_phi^2) / ((s-lambda_s)*(s+1/T_R)*(s^2+2*zeta_d*omega_d*s+omega_d^2)) * exp(-tau_p*s)`.

The numerator's `s^2` is preserved. The continuous model is discretized at 200 Hz with ZOH. Delay is implemented by a FIFO history with linear fractional interpolation; the RL action rate is 50 Hz and each action is held for four plant steps.

## Dataset and quality contract

Candidate parameter vectors are generated with scrambled Sobol points only after choosing scenario, aircraft class, flight phase, and quality region. Derived values are never sampled independently. Feasible candidates pass stability/numeric/parameter gates and retain scenario, source, split, and quality labels. The deterministic split target is: train core 1200, train boundary 600, validation 300, ID test 450, OOD test 300, extreme test 150.

`evaluate_roll_response(t, p)` returns P1, P2, P3, P_osc/P_av, peak magnitude, settling time, IAE, ITAE, and NRMSE. The same metric implementation is used by raw benchmarks, the environment diagnostics, and controller evaluation.

## Environment and controller contract

At every action step, `F_eq = F_pilot + delta_F_RL`; the latter is an equivalent research input augmentation, not a physical second pilot force. The actor receives deployable observation `[F_pilot, p, p_dot, previous_action, command_history]` plus privileged theta. The critic additionally receives full simulator state including delay history. Reward uses normalized reference tracking error, action magnitude, action increment, and explicit constraint penalties. Posc/Pav is evaluation-only rather than dense reward.

The baseline is a single privileged SAC. The Teacher is a feature-mixture Gaussian SAC actor: theta encoder 8->64->32, observation/history encoder ->128->128, theta-only router 32->64->E, E experts 128->128->64, weighted feature mixture, and one squashed Gaussian action head. Twin critics are plain 256->256 MLPs. The actor adds KL(mean router weight || uniform) with initial coefficient 1e-3. Initial E is four.

## Local smoke-test contract

Smoke tests are CPU-only and have two layers: deterministic unit/integration checks for dynamics, delay, sampling, metrics, reference model, and environment; then tiny fixed-budget optimization runs for single SAC and MoE-SAC that prove replay, losses, checkpoints, deterministic evaluation, and router diagnostics are wired. They do not establish training convergence or research performance.

## Acceptance gates

* A: plant linearity, cancellation, delay, state update, and metric extraction tests pass.
* B: a CPU fixed-plant SAC smoke run emits a checkpoint and has finite metrics/actions.
* C: a CPU global conditioned SAC smoke run samples both core and boundary plants and completes evaluation.
* D: a CPU 4-expert MoE run emits finite actor/critic/balance losses and nonempty router usage.
* E: local benchmark scripts produce a reproducible report for validation/ID/OOD/extreme partitions; formal long training remains deferred to explicit GPU authorization.
