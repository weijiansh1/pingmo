# Privileged Critic Repair Design

## Goal

Replace the exploratory SB3 path with a trainable PyTorch SAC Teacher whose actor consumes only the 13-element Teacher observation and whose twin critics consume a separate Markovian privileged state.  Keep the work limited to Teacher training; Student, distillation and transfer remain out of scope.

## State contract

`actor_obs` remains `[F_pilot, p, p_dot, previous_action, action_history, theta(8)]` with dimension 13.  It is the only input accepted by the actor and is therefore the future Teacher deployment interface.

`critic_obs` is constructed by the environment as `actor_obs + plant continuous state + plant delay FIFO + reference continuous state + reference delay FIFO`.  Each FIFO is padded to the largest delay history required by the environment's configured plant population, so batches have a fixed dimension.  The critic state is provided at reset and after every step.

## Learner and replay

The replay buffer stores `actor_obs`, `critic_obs`, `action`, `reward`, `next_actor_obs`, `next_critic_obs`, and `done` separately.  Both Q networks and their target copies take `(critic_obs, action)`; the actor takes only `actor_obs`.  The actor loss evaluates its sampled action through the current critics with the matching current `critic_obs`.  Critic targets use target Q networks and `next_critic_obs`.  Target parameters use Polyak updates.

## Training and evidence

A bounded fixed-plant trainer will collect real environment transitions, update the custom SAC from replay, save an actor/critic checkpoint, and run deterministic evaluation.  Tests prove the two observation streams have different dimensions, critic gradients depend on privileged features while actor outputs do not, replay preserves both streams, and a short CPU train loop emits finite diagnostics.  Only after these checks pass will the same bounded script be copied to P4 for a short experiment.

## Non-goals

This repair does not claim convergence, does not add MoE routing to the first corrected experiment, and does not change the frozen plant, reference model, or aircraft splits.
