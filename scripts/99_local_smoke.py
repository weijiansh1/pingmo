"""Run the CPU-only integration smoke test; never a long training job."""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.smoke import run_local_smoke


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = run_local_smoke(root / "checkpoints" / "smoke", updates=4, seed=20260827)
    target = root / "results" / "local_smoke_report.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
