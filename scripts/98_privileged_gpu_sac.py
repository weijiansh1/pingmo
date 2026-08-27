"""Run a bounded P4 experiment with the custom two-stream privileged SAC."""

from pathlib import Path
import json
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments.privileged_sac import train_fixed_privileged_sac


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("this bounded GPU script requires CUDA")
    root = Path(__file__).resolve().parents[1]
    report = train_fixed_privileged_sac(
        root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl",
        "train_core-0000",
        root / "checkpoints/privileged_sac_train_core_0000",
        timesteps=20_000,
        warmup_steps=1_000,
        batch_size=128,
        seed=20260827,
        device="cuda",
        correction_ratio=0.5,
    )
    print(json.dumps({"cuda": torch.cuda.get_device_name(0), **report}, ensure_ascii=False, indent=2))
