"""Run a bounded fixed-plant SAC experiment; not a formal Teacher run."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiments.exploratory_sac import DEFAULT_TRAIN_PLANT_ID, train_short_experiment

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = train_short_experiment(root / "data/aircraft/generated/p_channel_library_20260827_v2_stratified/plants.jsonl", DEFAULT_TRAIN_PLANT_ID, root / "checkpoints/exploratory_sac_train_core_0000", timesteps=20_000, device="cuda")
    print(json.dumps(report, ensure_ascii=False, indent=2))
