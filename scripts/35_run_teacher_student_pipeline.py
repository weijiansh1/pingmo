"""Run the gated RL-Teacher to student-driven Student pipeline end to end."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.distillation.distill import DenseStudentTrainingConfig
from src.distillation.student_driven import (
    StudentDrivenDistillationConfig,
    run_student_driven_distillation,
)
from src.experiments.exploratory_sac import load_persisted_records
from src.teacher.specialist.manager import train_teacher_bank
from src.teacher.specialist.td3_manager import train_pid_guided_teacher_bank
from src.teacher.specialist.td3_trainer import PIDGuidedTD3Config
from src.teacher.specialist.trainer import SpecialistTrainingConfig
from src.utils.provenance import git_source_revision, sha256_file


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    teacher = SpecialistTrainingConfig()
    student = DenseStudentTrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=ROOT
        / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/teacher_student_pipeline"
    )
    parser.add_argument("--teacher-count", type=int, default=6)
    parser.add_argument(
        "--teacher-algorithm",
        choices=("pid-guided-td3", "sac"),
        default="pid-guided-td3",
    )
    parser.add_argument("--plant-id", action="append")
    parser.add_argument(
        "--pid-report-root",
        type=Path,
        default=ROOT / "results/teacher_student_pipeline/00_pid_oracles",
    )
    parser.add_argument("--teacher-workers", type=int, default=3)
    parser.add_argument("--teacher-steps", type=int, default=teacher.total_steps)
    parser.add_argument("--td3-steps", type=int, default=5_000)
    parser.add_argument("--teacher-warmup-steps", type=int, default=teacher.warmup_steps)
    parser.add_argument("--teacher-batch-size", type=int, default=teacher.batch_size)
    parser.add_argument("--teacher-network-width", type=int, default=704)
    parser.add_argument("--teacher-residual-blocks", type=int, default=10)
    parser.add_argument("--teacher-initial-alpha", type=float, default=teacher.initial_alpha)
    parser.add_argument("--td3-bc-epochs", type=int, default=100)
    parser.add_argument("--td3-update-interval", type=int, default=4)
    parser.add_argument("--td3-exploration-std", type=float, default=0.01)
    parser.add_argument("--td3-behavior-weight", type=float, default=100.0)
    parser.add_argument("--td3-actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--td3-critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--td3-q-scale", type=float, default=0.05)
    parser.add_argument("--td3-maximum-q-coefficient", type=float, default=1.0)
    parser.add_argument("--td3-critic-warmup-updates", type=int, default=500)
    parser.add_argument("--td3-actor-trust-region-l2", type=float, default=0.002)
    parser.add_argument("--td3-residual-action-limit", type=float, default=0.05)
    parser.add_argument("--history-steps", type=int, default=teacher.history_steps)
    parser.add_argument("--force-delta-weight", type=float, default=teacher.force_delta_weight)
    parser.add_argument("--reward-scale", type=float, default=teacher.reward_scale)
    parser.add_argument("--skip-teacher-quality-gate", action="store_true")
    parser.add_argument("--disable-odd-policy", action="store_true")
    parser.add_argument(
        "--teacher-odd-policy-stage",
        choices=("training", "inference"),
        default=teacher.odd_policy_projection_stage,
    )
    parser.add_argument("--dagger-rounds", type=int, default=3)
    parser.add_argument("--initial-sample-stride", type=int, default=2)
    parser.add_argument("--student-sample-stride", type=int, default=1)
    parser.add_argument(
        "--distillation-split-strategy",
        choices=("aircraft_holdout", "all_aircraft_command_holdout"),
        default="aircraft_holdout",
    )
    parser.add_argument("--student-epochs", type=int, default=student.epochs)
    parser.add_argument(
        "--student-architecture",
        choices=("dense", "theta_routed_linear_moe"),
        default=student.architecture,
    )
    parser.add_argument("--student-batch-size", type=int, default=student.batch_size)
    parser.add_argument("--student-network-width", type=int, default=student.network_width)
    parser.add_argument("--student-residual-blocks", type=int, default=student.residual_blocks)
    parser.add_argument("--student-patience-epochs", type=int, default=student.patience_epochs)
    parser.add_argument("--student-moe-experts", type=int, default=student.moe_expert_count)
    parser.add_argument(
        "--student-moe-router-temperature",
        type=float,
        default=student.moe_router_temperature,
    )
    parser.add_argument(
        "--student-moe-prototype-movement-limit",
        type=float,
        default=student.moe_prototype_movement_limit,
    )
    parser.add_argument("--seed", type=int, default=teacher.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    root_report_path = args.output / "pipeline_report.json"
    teacher_config = SpecialistTrainingConfig(
        total_steps=(
            args.td3_steps
            if args.teacher_algorithm == "pid-guided-td3"
            else args.teacher_steps
        ),
        warmup_steps=(
            0
            if args.teacher_algorithm == "pid-guided-td3"
            else args.teacher_warmup_steps
        ),
        batch_size=args.teacher_batch_size,
        replay_capacity=max(200_000, args.teacher_batch_size),
        command_mode="extended",
        history_steps=args.history_steps,
        network_width=args.teacher_network_width,
        residual_blocks=args.teacher_residual_blocks,
        initial_alpha=args.teacher_initial_alpha,
        force_delta_weight=args.force_delta_weight,
        reward_scale=args.reward_scale,
        enforce_odd_policy=not args.disable_odd_policy,
        odd_policy_projection_stage=args.teacher_odd_policy_stage,
        enforce_quality_gate=not args.skip_teacher_quality_gate,
        seed=args.seed,
        device=args.device,
    )
    student_config = DenseStudentTrainingConfig(
        architecture=args.student_architecture,
        epochs=args.student_epochs,
        batch_size=args.student_batch_size,
        network_width=args.student_network_width,
        residual_blocks=args.student_residual_blocks,
        patience_epochs=args.student_patience_epochs,
        enforce_odd_policy=not args.disable_odd_policy,
        moe_expert_count=args.student_moe_experts,
        moe_router_temperature=args.student_moe_router_temperature,
        moe_prototype_movement_limit=args.student_moe_prototype_movement_limit,
        seed=args.seed,
        device=args.device,
    )
    td3_config = PIDGuidedTD3Config(
        total_steps=args.td3_steps,
        batch_size=args.teacher_batch_size,
        replay_capacity=max(50_000, args.teacher_batch_size),
        behavior_clone_epochs=args.td3_bc_epochs,
        update_interval_steps=args.td3_update_interval,
        exploration_std=args.td3_exploration_std,
        network_width=args.teacher_network_width,
        residual_blocks=args.teacher_residual_blocks,
        behavior_regularization_weight=args.td3_behavior_weight,
        actor_learning_rate=args.td3_actor_learning_rate,
        critic_learning_rate=args.td3_critic_learning_rate,
        q_normalization_scale=args.td3_q_scale,
        maximum_q_coefficient=args.td3_maximum_q_coefficient,
        critic_warmup_updates=args.td3_critic_warmup_updates,
        actor_trust_region_l2=args.td3_actor_trust_region_l2,
        residual_action_limit=args.td3_residual_action_limit,
        seed=args.seed,
        device=args.device,
    )
    _write_json(
        root_report_path,
        {
            "schema_version": "teacher_student_pipeline_v1",
            "status": "training_teachers",
            "source": git_source_revision(),
            "library": str(args.library.resolve()),
            "teacher_config": asdict(teacher_config),
            "teacher_algorithm": args.teacher_algorithm,
            "td3_config": (
                asdict(td3_config)
                if args.teacher_algorithm == "pid-guided-td3"
                else None
            ),
            "student_config": asdict(student_config),
        },
    )
    teacher_dir = args.output / "01_teachers"
    if args.teacher_algorithm == "pid-guided-td3":
        if not args.plant_id:
            raise ValueError(
                "pid-guided-td3 requires one or more explicit --plant-id values"
            )
        teacher_records = load_persisted_records(args.library, args.plant_id)
        teacher_bank = train_pid_guided_teacher_bank(
            args.library,
            args.pid_report_root,
            teacher_dir,
            teacher_records,
            teacher_config,
            td3_config,
        )
    else:
        teacher_bank = train_teacher_bank(
            args.library,
            teacher_dir,
            teacher_config,
            count=args.teacher_count,
            workers=args.teacher_workers,
        )
    teacher_bank_path = teacher_dir / "teacher_bank.json"
    if teacher_bank["status"] != "complete":
        report = {
            "schema_version": "teacher_student_pipeline_v1",
            "status": "teacher_quality_gate_failed",
            "source": git_source_revision(),
            "teacher_bank": {
                "path": str(teacher_bank_path),
                "sha256": sha256_file(teacher_bank_path),
                "accepted": teacher_bank.get("accepted_teacher_count", 0),
                "total": teacher_bank.get("teacher_count", 0),
            },
            "elapsed_s": time.perf_counter() - started,
        }
        _write_json(root_report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)

    _write_json(
        root_report_path,
        {
            "schema_version": "teacher_student_pipeline_v1",
            "status": "distilling_student",
            "source": git_source_revision(),
            "teacher_bank": {
                "path": str(teacher_bank_path),
                "sha256": sha256_file(teacher_bank_path),
            },
        },
    )
    distillation = run_student_driven_distillation(
        teacher_bank_path,
        args.output / "02_student_driven_distillation",
        StudentDrivenDistillationConfig(
            dagger_rounds=args.dagger_rounds,
            initial_sample_stride=args.initial_sample_stride,
            student_sample_stride=args.student_sample_stride,
            split_strategy=args.distillation_split_strategy,
            student_training=student_config,
            seed=args.seed,
            device=args.device,
        ),
    )
    report = {
        "schema_version": "teacher_student_pipeline_v1",
        "status": distillation["status"],
        "source": git_source_revision(),
        "library": {
            "path": str(args.library.resolve()),
            "sha256": sha256_file(args.library),
        },
        "teacher_bank": {
            "path": str(teacher_bank_path),
            "sha256": sha256_file(teacher_bank_path),
            "teacher_count": teacher_bank["teacher_count"],
        },
        "distillation_report": str(
            args.output / "02_student_driven_distillation/pipeline_report.json"
        ),
        "final_student_checkpoint": distillation["final_checkpoint"],
        "quality_gate": distillation["quality_gate"],
        "elapsed_s": time.perf_counter() - started,
        "artifact_layout": {
            "teachers": "01_teachers/",
            "distillation_rounds": "02_student_driven_distillation/round_*/",
            "final_student": "02_student_driven_distillation/final/",
            "round_metrics": "02_student_driven_distillation/round_metrics.csv",
            "distillation_plot": "02_student_driven_distillation/distillation_progress.png",
        },
    }
    _write_json(root_report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
