"""Run the read-only G0 constrained-reference feasibility audit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.commands import default_command_suite
from src.experiments.feasibility_audit import audit_library, summarize_audit_rows, write_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=ROOT / "data" / "aircraft" / "generated" / "p_channel_library_iv_a_manual_v1" / "plants.jsonl",
        help="immutable 3,000-aircraft parameter library",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "g0_feasibility_audit",
        help="directory for pair-level records and split summaries",
    )
    parser.add_argument("--duration-s", type=float, default=10.0, help="duration of every command response")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(24, os.cpu_count() or 1)),
        help="parallel CPU processes; G0 is a deterministic simulation, not GPU training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least one")
    rows = audit_library(args.library, default_command_suite(), duration_s=args.duration_s, workers=args.workers)
    destination = write_audit(args.output, rows)
    print(summarize_audit_rows(rows))
    print(f"wrote {len(rows)} aircraft-command records to {destination}")


if __name__ == "__main__":
    main()
