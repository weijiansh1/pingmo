"""Evaluate raw, specialist Teachers, and the dense Student in closed loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.distillation.validate import evaluate_dense_student_bank


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "results/dense_student/dense_student.pt",
    )
    parser.add_argument(
        "--teacher-bank",
        type=Path,
        default=root / "results/specialist_teachers/teacher_bank.json",
    )
    parser.add_argument("--output", type=Path, default=root / "results/dense_student_evaluation")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = evaluate_dense_student_bank(
        args.checkpoint,
        args.teacher_bank,
        args.output,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
