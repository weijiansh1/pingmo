"""Run a bounded multi-aircraft MLP-SAC or MoE-SAC GPU trial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.short_teacher_trial import ShortTrialConfig, run_short_teacher_trial


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("mlp", "moe"), default="mlp")
    parser.add_argument("--steps", type=int, default=6_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--update-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--library",
        type=Path,
        default=root / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = args.output or Path("results/short_teacher_trials") / f"{args.teacher}-seed-{args.seed}"
    configuration = ShortTrialConfig(
        teacher_kind=args.teacher,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        update_every_steps=args.update_every,
        seed=args.seed,
        device=args.device,
    )
    result = run_short_teacher_trial(args.library, output, configuration)
    print(json.dumps(result["evaluation_summary"], ensure_ascii=False, indent=2))
