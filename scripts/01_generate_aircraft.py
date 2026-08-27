"""Generate the manual-v1 IV-A response-calibrated plant bank."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.aircraft.sampler import build_iv_a_library


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--candidates", type=int, default=16384)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/aircraft/generated/p_channel_library_iv_a_manual_v1")
    args = parser.parse_args()
    print(build_iv_a_library(args.output, seed=args.seed, candidate_count=args.candidates))
