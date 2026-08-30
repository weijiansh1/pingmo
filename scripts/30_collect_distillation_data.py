"""Collect specialist Teacher trajectories for conditional Student distillation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distillation.collect_data import DistillationCollectionConfig, collect_teacher_bank_data


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    defaults = DistillationCollectionConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-bank",
        type=Path,
        default=root / "results/specialist_teachers/teacher_bank.json",
    )
    parser.add_argument("--output", type=Path, default=root / "results/distillation_data")
    parser.add_argument("--sample-stride", type=int, default=defaults.sample_stride)
    parser.add_argument(
        "--validation-aircraft-fraction",
        type=float,
        default=defaults.validation_aircraft_fraction,
    )
    parser.add_argument(
        "--split-strategy",
        choices=("aircraft_holdout", "all_aircraft_command_holdout"),
        default=defaults.split_strategy,
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = collect_teacher_bank_data(
        args.teacher_bank,
        args.output,
        DistillationCollectionConfig(
            sample_stride=args.sample_stride,
            validation_aircraft_fraction=args.validation_aircraft_fraction,
            split_strategy=args.split_strategy,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
