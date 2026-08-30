"""Run Teacher initialization and multi-round Student-driven policy distillation."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.distill import DenseStudentTrainingConfig
from src.distillation.student_driven import (
    StudentDrivenDistillationConfig,
    reselect_student_driven_distillation,
    run_student_driven_distillation,
)


def parse_args() -> argparse.Namespace:
    student_defaults = DenseStudentTrainingConfig()
    pipeline_defaults = StudentDrivenDistillationConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-bank",
        type=Path,
        default=ROOT / "results/specialist_teachers/teacher_bank.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results/student_driven_distillation",
    )
    parser.add_argument(
        "--reselect-existing",
        action="store_true",
        help="Reapply the current round-selection rule without retraining.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted pipeline from verified round artifacts.",
    )
    parser.add_argument("--dagger-rounds", type=int, default=pipeline_defaults.dagger_rounds)
    parser.add_argument(
        "--initial-sample-stride",
        type=int,
        default=pipeline_defaults.initial_sample_stride,
    )
    parser.add_argument(
        "--student-sample-stride",
        type=int,
        default=pipeline_defaults.student_sample_stride,
    )
    parser.add_argument(
        "--validation-aircraft-fraction",
        type=float,
        default=pipeline_defaults.validation_aircraft_fraction,
    )
    parser.add_argument(
        "--split-strategy",
        choices=("aircraft_holdout", "all_aircraft_command_holdout"),
        default=pipeline_defaults.split_strategy,
    )
    parser.add_argument(
        "--validation-plant-id",
        action="append",
        default=list(pipeline_defaults.validation_plant_ids),
        help=(
            "Explicit whole-aircraft validation holdout; repeat once per aircraft. "
            "Only valid with --split-strategy aircraft_holdout."
        ),
    )
    parser.add_argument("--epochs", type=int, default=student_defaults.epochs)
    parser.add_argument(
        "--student-architecture",
        choices=("dense", "theta_routed_linear_moe"),
        default=student_defaults.architecture,
    )
    parser.add_argument("--batch-size", type=int, default=student_defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=student_defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=student_defaults.weight_decay)
    parser.add_argument("--network-width", type=int, default=student_defaults.network_width)
    parser.add_argument("--residual-blocks", type=int, default=student_defaults.residual_blocks)
    parser.add_argument("--patience-epochs", type=int, default=student_defaults.patience_epochs)
    parser.add_argument(
        "--moe-experts", type=int, default=student_defaults.moe_expert_count
    )
    parser.add_argument(
        "--moe-router-temperature",
        type=float,
        default=student_defaults.moe_router_temperature,
    )
    parser.add_argument(
        "--moe-prototype-movement-limit",
        type=float,
        default=student_defaults.moe_prototype_movement_limit,
    )
    parser.add_argument(
        "--moe-router-balance-weight",
        type=float,
        default=student_defaults.moe_router_balance_weight,
    )
    parser.add_argument(
        "--moe-router-z-loss-weight",
        type=float,
        default=student_defaults.moe_router_z_loss_weight,
    )
    parser.add_argument(
        "--moe-prototype-anchor-weight",
        type=float,
        default=student_defaults.moe_prototype_anchor_weight,
    )
    parser.add_argument("--disable-odd-policy", action="store_true")
    parser.add_argument(
        "--max-student-teacher-rmse-gap",
        type=float,
        default=pipeline_defaults.max_student_teacher_rmse_gap_deg_s,
    )
    parser.add_argument(
        "--minimum-student-improvement-rate",
        type=float,
        default=pipeline_defaults.minimum_student_improvement_rate,
    )
    parser.add_argument(
        "--maximum-student-harm-rate",
        type=float,
        default=pipeline_defaults.maximum_student_harm_rate,
    )
    parser.add_argument(
        "--maximum-student-peak-error",
        type=float,
        default=pipeline_defaults.maximum_student_peak_error_deg_s,
    )
    parser.add_argument(
        "--maximum-student-requested-force-variation",
        type=float,
        default=(
            pipeline_defaults.maximum_mean_student_requested_force_variation_n
        ),
    )
    parser.add_argument(
        "--maximum-student-teacher-force-variation-ratio",
        type=float,
        default=pipeline_defaults.maximum_student_teacher_force_variation_ratio,
    )
    parser.add_argument("--seed", type=int, default=pipeline_defaults.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.reselect_existing:
        result = reselect_student_driven_distillation(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0)
    student_config = DenseStudentTrainingConfig(
        architecture=args.student_architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        network_width=args.network_width,
        residual_blocks=args.residual_blocks,
        patience_epochs=args.patience_epochs,
        enforce_odd_policy=not args.disable_odd_policy,
        moe_expert_count=args.moe_experts,
        moe_router_temperature=args.moe_router_temperature,
        moe_prototype_movement_limit=args.moe_prototype_movement_limit,
        moe_router_balance_weight=args.moe_router_balance_weight,
        moe_router_z_loss_weight=args.moe_router_z_loss_weight,
        moe_prototype_anchor_weight=args.moe_prototype_anchor_weight,
        seed=args.seed,
        device=args.device,
    )
    result = run_student_driven_distillation(
        args.teacher_bank,
        args.output,
        StudentDrivenDistillationConfig(
            dagger_rounds=args.dagger_rounds,
            initial_sample_stride=args.initial_sample_stride,
            student_sample_stride=args.student_sample_stride,
            validation_aircraft_fraction=args.validation_aircraft_fraction,
            split_strategy=args.split_strategy,
            validation_plant_ids=tuple(args.validation_plant_id),
            student_training=student_config,
            seed=args.seed,
            device=args.device,
            max_student_teacher_rmse_gap_deg_s=args.max_student_teacher_rmse_gap,
            minimum_student_improvement_rate=args.minimum_student_improvement_rate,
            maximum_student_harm_rate=args.maximum_student_harm_rate,
            maximum_student_peak_error_deg_s=args.maximum_student_peak_error,
            maximum_mean_student_requested_force_variation_n=(
                args.maximum_student_requested_force_variation
            ),
            maximum_student_teacher_force_variation_ratio=(
                args.maximum_student_teacher_force_variation_ratio
            ),
        ),
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
