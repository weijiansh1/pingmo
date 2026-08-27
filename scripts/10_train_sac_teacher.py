"""Run a bounded local privileged-SAC Teacher smoke experiment."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.privileged_sac import train_fixed_privileged_sac


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = train_fixed_privileged_sac(
        root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl",
        "train_core-0000",
        root / "checkpoints/privileged_sac_smoke",
        timesteps=512,
        warmup_steps=64,
        batch_size=32,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
