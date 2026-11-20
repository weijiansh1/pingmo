"""Build the coupled v2 library on top of the v1 joint library (CPU only, no training)."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pinmo6.library_v2 import build_coupled_library  # noqa: E402

if __name__ == "__main__":
    out = build_coupled_library(HERE / "library/joint_six_channel_5000_v1/plants.jsonl",
                                HERE / "library/coupled_six_channel_5000_v2")
    print(out)
