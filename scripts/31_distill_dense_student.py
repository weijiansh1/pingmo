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
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--network-width", type=int, default=defaults.network_width)
    parser.add_argument("--residual-blocks", type=int, default=defaults.residual_blocks)
    parser.add_argument("--patience-epochs", type=int, default=defaults.patience_epochs)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = train_dense_student(
        args.dataset,
        args.output,
        DenseStudentTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            network_width=args.network_width,
            residual_blocks=args.residual_blocks,
            patience_epochs=args.patience_epochs,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
