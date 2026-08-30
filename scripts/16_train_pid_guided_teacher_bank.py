"""Train a gated PID-guided TD3 Teacher Bank on explicit aircraft IDs."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.experiments.exploratory_sac import load_persisted_records
from src.teacher.specialist.td3_manager import train_pid_guided_teacher_bank
from src.teacher.specialist.td3_trainer import PIDGuidedTD3Config
from src.teacher.specialist.trainer import SpecialistTrainingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=(
            ROOT
            / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
        ),
    )
    parser.add_argument(
        "--pid-report-root",
        type=Path,
        default=ROOT / "results/teacher_student_pipeline/00_pid_oracles",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/teacher_student_pipeline/01_teachers",
    )
    parser.add_argument("--plant-id", action="append", required=True)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--bc-epochs", type=int, default=100)
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--updates-per-interval", type=int, default=1)
    parser.add_argument("--exploration-std", type=float, default=0.01)
    parser.add_argument("--network-width", type=int, default=704)
    parser.add_argument("--residual-blocks", type=int, default=10)
    parser.add_argument("--behavior-weight", type=float, default=100.0)
    parser.add_argument("--bc-learning-rate", type=float, default=3e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--q-normalization-scale", type=float, default=0.05)
    parser.add_argument("--maximum-q-coefficient", type=float, default=1.0)
    parser.add_argument("--critic-warmup-updates", type=int, default=500)
    parser.add_argument("--actor-trust-region-l2", type=float, default=0.002)
    parser.add_argument("--residual-action-limit", type=float, default=0.05)
    parser.add_argument("--progress-interval", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records = load_persisted_records(args.library, args.plant_id)
    environment_config = SpecialistTrainingConfig(
        command_mode="extended",
        history_steps=0,
        enforce_odd_policy=True,
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
        q_normalization_scale=args.q_normalization_scale,
        maximum_q_coefficient=args.maximum_q_coefficient,
        critic_warmup_updates=args.critic_warmup_updates,
        actor_trust_region_l2=args.actor_trust_region_l2,
        residual_action_limit=args.residual_action_limit,
        progress_interval_steps=args.progress_interval,
        seed=args.seed,
        device=args.device,
    )
    result = train_pid_guided_teacher_bank(
        args.library,
        args.pid_report_root,
        args.output,
        records,
        environment_config,
        td3_config,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "algorithm": result["algorithm"],
                "teacher_count": result["teacher_count"],
                "accepted_teacher_count": result["accepted_teacher_count"],
                "quality_region_counts": result["quality_region_counts"],
                "teacher_bank": str(args.output / "teacher_bank.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
