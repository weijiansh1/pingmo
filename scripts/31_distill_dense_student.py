"""Distill a parameter-conditioned dense Student from specialist actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distillation.distill import DenseStudentTrainingConfig, train_dense_student


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    defaults = DenseStudentTrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=root / "results/distillation_data/dataset.json",
    )
    parser.add_argument("--output", type=Path, default=root / "results/dense_student")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument(
        "--student-architecture",
        choices=("dense", "theta_routed_linear_moe"),
        default=defaults.architecture,
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--network-width", type=int, default=defaults.network_width)
    parser.add_argument("--residual-blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument("--patience-epochs", type=int, default=defaults.patience_epochs)
    parser.add_argument("--moe-experts", type=int, default=defaults.moe_expert_count)
    parser.add_argument(
        "--moe-router-temperature",
        type=float,
        default=defaults.moe_router_temperature,
    )
    parser.add_argument(
        "--moe-prototype-movement-limit",
        type=float,
        default=defaults.moe_prototype_movement_limit,
    )
    parser.add_argument(
        "--moe-router-balance-weight",
        type=float,
        default=defaults.moe_router_balance_weight,
    )
    parser.add_argument(
        "--moe-router-z-loss-weight",
        type=float,
        default=defaults.moe_router_z_loss_weight,
    )
    parser.add_argument(
        "--moe-prototype-anchor-weight",
        type=float,
        default=defaults.moe_prototype_anchor_weight,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_dense_student(
        args.dataset,
        args.output,
        DenseStudentTrainingConfig(
            architecture=args.student_architecture,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            network_width=args.network_width,
            residual_blocks=args.residual_blocks,
            patience_epochs=args.patience_epochs,
            moe_expert_count=args.moe_experts,
            moe_router_temperature=args.moe_router_temperature,
            moe_prototype_movement_limit=args.moe_prototype_movement_limit,
            moe_router_balance_weight=args.moe_router_balance_weight,
            moe_router_z_loss_weight=args.moe_router_z_loss_weight,
            moe_prototype_anchor_weight=args.moe_prototype_anchor_weight,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
