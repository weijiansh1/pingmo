"""Build the v4 library: v3 plus the 2026-10-31 review corrections (CPU only)."""
from pathlib import Path
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pinmo6.sampler_v4 import build_v4_library  # noqa: E402

if __name__ == "__main__":
    print(build_v4_library(HERE / "library/six_channel_5000_v4", workers=32))
