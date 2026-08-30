"""Train one independent specialist SAC Teacher per selected aircraft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.exploratory_sac import load_persisted_records
from src.teacher.specialist.manager import select_specialist_records, train_teacher_bank
from src.teacher.specialist.trainer import SpecialistTrainingConfig


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    defaults = SpecialistTrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl",
    )
    parser.add_argument("--output", type=Path, default=root / "results/specialist_teachers")
    parser.add_argument("--plant-id", action="append", default=[])
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=defaults.total_steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--episode-duration", type=float, default=defaults.episode_duration_s)
    parser.add_argument("--command-mode", choices=("step", "extended"), default=defaults.command_mode)
    parser.add_argument("--network-width", type=int, default=defaults.network_width)
    parser.add_argument("--residual-blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument("--history-steps", type=int, default=defaults.history_steps)
    parser.add_argument("--plant-dt", type=float, default=defaults.plant_dt_s)
    parser.add_argument("--policy-dt", type=float, default=defaults.policy_dt_s)
    parser.add_argument("--force-limit", type=float, default=defaults.force_limit_n)
    parser.add_argument("--force-rate-limit", type=float, default=defaults.force_rate_limit_n_s)
    parser.add_argument("--reference-frequency", type=float, default=defaults.reference_natural_frequency_rad_s)
    parser.add_argument("--reference-damping", type=float, default=defaults.reference_damping_ratio)
    parser.add_argument(
        "--reference-delay-mode",
        choices=("match_plant_transport_delay", "none"),
        default=defaults.reference_delay_mode,
    )
    parser.add_argument("--tracking-weight", type=float, default=defaults.tracking_error_weight)
    parser.add_argument("--force-weight", type=float, default=defaults.force_energy_weight)
    parser.add_argument("--force-delta-weight", type=float, default=defaults.force_delta_weight)
    parser.add_argument("--reward-scale", type=float, default=defaults.reward_scale)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--initial-alpha", type=float, default=defaults.initial_alpha)
    parser.add_argument("--disable-odd-policy", action="store_true")
    parser.add_argument(
        "--odd-policy-stage",
        choices=("training", "inference"),
        default=defaults.odd_policy_projection_stage,
    )
    parser.add_argument("--skip-quality-gate", action="store_true")
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configuration = SpecialistTrainingConfig(
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        episode_duration_s=args.episode_duration,
        command_mode=args.command_mode,
        network_width=args.network_width,
        residual_blocks=args.residual_blocks,
        history_steps=args.history_steps,
        plant_dt_s=args.plant_dt,
        policy_dt_s=args.policy_dt,
        force_limit_n=args.force_limit,
        force_rate_limit_n_s=args.force_rate_limit,
        reference_natural_frequency_rad_s=args.reference_frequency,
        reference_damping_ratio=args.reference_damping,
        reference_delay_mode=args.reference_delay_mode,
        tracking_error_weight=args.tracking_weight,
        force_energy_weight=args.force_weight,
        force_delta_weight=args.force_delta_weight,
        reward_scale=args.reward_scale,
        gamma=args.gamma,
        initial_alpha=args.initial_alpha,
        enforce_odd_policy=not args.disable_odd_policy,
        odd_policy_projection_stage=args.odd_policy_stage,
        enforce_quality_gate=not args.skip_quality_gate,
        seed=args.seed,
        device=args.device,
    )
    if args.plant_id:
        selected = load_persisted_records(args.library, args.plant_id)
    else:
        selected = select_specialist_records(args.library, args.count, seed=args.seed)
    result = train_teacher_bank(
        args.library,
        args.output,
        configuration,
        records=selected,
        count=len(selected),
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
