"""Train one deterministic TD3 Teacher using environment reward only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.exploratory_sac import load_persisted_records  # noqa: E402
from src.envs.roll_rate_commands import RandomCommandDistribution  # noqa: E402
from src.teacher.specialist.pure_td3_trainer import (  # noqa: E402
    PureRewardTD3Config,
    train_pure_reward_td3,
)
from src.teacher.specialist.trainer import SpecialistTrainingConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    defaults = PureRewardTD3Config()
    environment_defaults = SpecialistTrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=ROOT
        / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl",
    )
    parser.add_argument("--plant-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=defaults.total_steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--network-width", type=int, default=defaults.network_width)
    parser.add_argument("--residual-blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument(
        "--exploration-std-initial",
        type=float,
        default=defaults.exploration_std_initial,
    )
    parser.add_argument(
        "--exploration-std-final",
        type=float,
        default=defaults.exploration_std_final,
    )
    parser.add_argument(
        "--exploration-decay-steps",
        type=int,
        default=defaults.exploration_decay_steps,
    )
    parser.add_argument(
        "--actor-learning-rate",
        type=float,
        default=defaults.actor_learning_rate,
    )
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=defaults.critic_learning_rate,
    )
    parser.add_argument(
        "--critic-warmup-updates",
        type=int,
        default=defaults.critic_warmup_updates,
    )
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument(
        "--episode-duration",
        type=float,
        default=environment_defaults.episode_duration_s,
    )
    parser.add_argument(
        "--command-mode",
        choices=("step", "extended"),
        default="extended",
    )
    parser.add_argument(
        "--requested-action-history-steps",
        type=int,
        default=26,
        help="Fixed policy-rate action history; 26 covers the canonical library's 0.498 s maximum delay.",
    )
    parser.add_argument(
        "--include-reference-derivative",
        action="store_true",
        help="Expose the causal reference-rate derivative to the deployable Actor.",
    )
    parser.add_argument("--fixed-command-bank", action="store_true")
    parser.add_argument(
        "--random-command-sequence",
        action="store_true",
        help="Concatenate random command segments within each episode without resetting the plant.",
    )
    parser.add_argument("--segment-duration-min", type=float, default=2.0)
    parser.add_argument("--segment-duration-max", type=float, default=5.0)
    parser.add_argument("--long-dwell-step-probability", type=float, default=0.0)
    parser.add_argument("--long-dwell-duration-min", type=float, default=15.0)
    parser.add_argument("--long-dwell-duration-max", type=float, default=30.0)
    parser.add_argument("--random-duration-min", type=float, default=4.0)
    parser.add_argument("--random-duration-max", type=float, default=8.0)
    parser.add_argument("--random-frequency-min", type=float, default=0.20)
    parser.add_argument("--random-frequency-max", type=float, default=1.50)
    parser.add_argument(
        "--evaluation-interval-steps",
        type=int,
        default=defaults.evaluation_interval_steps,
    )
    parser.add_argument(
        "--force-limit", type=float, default=environment_defaults.force_limit_n
    )
    parser.add_argument(
        "--force-rate-limit",
        type=float,
        default=environment_defaults.force_rate_limit_n_s,
    )
    parser.add_argument(
        "--tracking-weight",
        type=float,
        default=environment_defaults.tracking_error_weight,
    )
    parser.add_argument(
        "--force-weight",
        type=float,
        default=environment_defaults.force_energy_weight,
    )
    parser.add_argument(
        "--force-delta-weight",
        type=float,
        default=environment_defaults.force_delta_weight,
    )
    parser.add_argument(
        "--reward-scale", type=float, default=environment_defaults.reward_scale
    )
    parser.add_argument("--skip-quality-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    records = load_persisted_records(args.library, [args.plant_id])
    if len(records) != 1:
        raise ValueError("pure-reward TD3 requires exactly one matching aircraft")
    environment_config = SpecialistTrainingConfig(
        episode_duration_s=args.episode_duration,
        command_mode=args.command_mode,
        history_steps=0,
        requested_action_history_steps=args.requested_action_history_steps,
        include_actor_actuator_state=True,
        include_reference_derivative=args.include_reference_derivative,
        critic_include_episode_progress=False,
        critic_include_command_context=not args.random_command_sequence,
        force_limit_n=args.force_limit,
        force_rate_limit_n_s=args.force_rate_limit,
        reference_delay_mode="match_plant_transport_delay",
        tracking_error_weight=args.tracking_weight,
        force_energy_weight=args.force_weight,
        force_delta_weight=args.force_delta_weight,
        reward_scale=args.reward_scale,
        enforce_odd_policy=True,
        odd_policy_projection_stage="training",
        enforce_quality_gate=not args.skip_quality_gate,
        seed=args.seed,
        device=args.device,
    )
    report = train_pure_reward_td3(
        records[0],
        args.output,
        environment_config,
        PureRewardTD3Config(
            total_steps=args.steps,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            replay_capacity=args.replay_capacity,
            exploration_std_initial=args.exploration_std_initial,
            exploration_std_final=args.exploration_std_final,
            exploration_decay_steps=args.exploration_decay_steps,
            network_width=args.network_width,
            residual_blocks=args.residual_blocks,
            gamma=args.gamma,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            critic_warmup_updates=args.critic_warmup_updates,
            evaluation_interval_steps=args.evaluation_interval_steps,
            randomize_training_commands=not args.fixed_command_bank,
            random_command_sequence=args.random_command_sequence,
            random_sequence_segment_duration_range_s=(
                args.segment_duration_min,
                args.segment_duration_max,
            ),
            long_dwell_step_probability=args.long_dwell_step_probability,
            long_dwell_duration_range_s=(
                args.long_dwell_duration_min,
                args.long_dwell_duration_max,
            ),
            random_command_distribution=RandomCommandDistribution(
                duration_range_s=(
                    args.random_duration_min,
                    args.random_duration_max,
                ),
                frequency_range_hz=(
                    args.random_frequency_min,
                    args.random_frequency_max,
                ),
            ),
            seed=args.seed,
            device=args.device,
        ),
        library_path=args.library,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "plant_id": report["plant_id"],
                "seed": args.seed,
                "steps": report["steps"],
                "actor_updates": report["actor_updates"],
                "actor_observation_dim": report["actor_observation_dim"],
                "training_command_scheduler": report["training_command_scheduler"],
                "best_validation": report["best_validation"],
                "evaluation": report["quality_gate"]["observed"],
                "actor_checkpoint": report["actor_checkpoint"],
                "best_validation_actor_checkpoint": report[
                    "best_validation_actor_checkpoint"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
