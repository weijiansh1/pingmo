"""Train one direct-force TD3 specialist initialized from a tuned PID."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import PIDGains  # noqa: E402
from src.experiments.exploratory_sac import load_persisted_records  # noqa: E402
from src.teacher.specialist.td3_trainer import (  # noqa: E402
    PIDGuidedTD3Config,
    train_pid_guided_td3,
)
from src.teacher.specialist.trainer import SpecialistTrainingConfig  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid-report", type=Path, required=True)
    parser.add_argument(
        "--library",
        type=Path,
        default=(
            ROOT
            / "data"
            / "aircraft"
            / "generated"
            / "p_channel_library_iv_a_manual_v1"
            / "plants.jsonl"
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "pid_guided_td3"
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--updates-per-interval", type=int, default=1)
    parser.add_argument("--exploration-std", type=float, default=0.01)
    parser.add_argument("--network-width", type=int, default=128)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--behavior-weight", type=float, default=100.0)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--reward-multiplier", type=float, default=1.0)
    parser.add_argument("--q-normalization-scale", type=float, default=0.05)
    parser.add_argument("--maximum-q-coefficient", type=float, default=1.0)
    parser.add_argument("--critic-warmup-updates", type=int, default=500)
    parser.add_argument("--actor-trust-region-l2", type=float, default=0.002)
    parser.add_argument("--residual-action-limit", type=float, default=0.05)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument("--disable-odd-policy", action="store_true")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pid_report = json.loads(args.pid_report.read_text(encoding="utf-8"))
    if pid_report.get("status") != "complete":
        raise ValueError("PID report must be complete")
    plant_id = str(pid_report["plant_id"])
    record = load_persisted_records(args.library, [plant_id])[0]
    gains = PIDGains(**pid_report["gains"])
    environment_config = SpecialistTrainingConfig(
        episode_duration_s=float(pid_report["episode_duration_s"]),
        plant_dt_s=float(pid_report["plant_dt_s"]),
        policy_dt_s=float(pid_report["policy_dt_s"]),
        reference_natural_frequency_rad_s=float(
            pid_report["reference_natural_frequency_rad_s"]
        ),
        reference_damping_ratio=float(pid_report["reference_damping_ratio"]),
        reference_delay_mode=str(pid_report["reference_delay_mode"]),
        command_mode="extended",
        history_steps=0,
        enforce_odd_policy=not args.disable_odd_policy,
        odd_policy_projection_stage="inference",
        seed=args.seed,
        device=args.device,
    )
    td3_config = PIDGuidedTD3Config(
        total_steps=args.steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        behavior_clone_epochs=args.bc_epochs,
        behavior_clone_batch_size=args.bc_batch_size,
        update_interval_steps=args.update_interval,
        updates_per_interval=args.updates_per_interval,
        exploration_std=args.exploration_std,
        network_width=args.network_width,
        residual_blocks=args.residual_blocks,
        behavior_regularization_weight=args.behavior_weight,
        behavior_clone_learning_rate=args.bc_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        reward_multiplier=args.reward_multiplier,
        q_normalization_scale=args.q_normalization_scale,
        maximum_q_coefficient=args.maximum_q_coefficient,
        critic_warmup_updates=args.critic_warmup_updates,
        actor_trust_region_l2=args.actor_trust_region_l2,
        residual_action_limit=args.residual_action_limit,
        progress_interval_steps=args.progress_interval,
        seed=args.seed,
        device=args.device,
    )
    destination = args.output / plant_id
    report = train_pid_guided_td3(
        record,
        gains,
        destination,
        environment_config,
        td3_config,
        library_path=args.library,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "algorithm": report["algorithm"],
                "plant_id": report["plant_id"],
                "parameter_counts": report["parameter_counts"],
                "demonstration_steps": report["demonstration_steps"],
                "behavior_cloning": report["behavior_cloning"],
                "online_steps": report["online_steps"],
                "online_updates": report["online_updates"],
                "online_episodes": report["online_episodes"],
                "training_elapsed_s": report["training_elapsed_s"],
                "evaluation": {
                    key: value
                    for key, value in report["evaluation"].items()
                    if key != "rows"
                },
                "quality_gate": report["quality_gate"],
                "accepted_for_distillation": report[
                    "accepted_for_distillation"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={destination / 'report.json'}")
    print(f"plot={destination / 'response_comparison.png'}")


if __name__ == "__main__":
    main()
