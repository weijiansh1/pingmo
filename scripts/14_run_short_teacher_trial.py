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
    defaults = ShortTrialConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", choices=("mlp", "moe"), default="mlp")
    parser.add_argument("--steps", type=int, default=defaults.total_steps)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--update-every", type=int, default=defaults.update_every_steps)
    parser.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--parallel-envs", type=int, default=defaults.parallel_envs)
    parser.add_argument("--evaluation-plants", type=int, default=defaults.evaluation_plants)
    parser.add_argument("--evaluation-batch-size", type=int, default=defaults.evaluation_batch_size)
    parser.add_argument("--progress-interval", type=int, default=defaults.progress_interval_steps)
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
        replay_capacity=args.replay_capacity,
        parallel_envs=args.parallel_envs,
        evaluation_plants=args.evaluation_plants,
        evaluation_batch_size=args.evaluation_batch_size,
        progress_interval_steps=args.progress_interval,
        seed=args.seed,
        device=args.device,
    )
    result = run_short_teacher_trial(args.library, output, configuration)
    print(json.dumps(result["evaluation_summary"], ensure_ascii=False, indent=2))
