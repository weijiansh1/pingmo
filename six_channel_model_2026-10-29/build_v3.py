"""Build the v3 library: GJB-level modes, real-aircraft coupling shape, joint grading (CPU only)."""
from pathlib import Path
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pinmo6.sampler_v3 import build_v3_library  # noqa: E402

if __name__ == "__main__":
    print(build_v3_library(HERE / "library/six_channel_5000_v3", workers=32))
