"""Generate Raw/PID/RL-Teacher comparisons for every accepted Teacher."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-bank",
        type=Path,
        default=ROOT
        / "results/teacher_student_pipeline/01_teachers/teacher_bank.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    args = _parse_args()
    bank_path = args.teacher_bank.resolve()
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    if bank.get("schema_version") != "specialist_teacher_bank_v1":
        raise ValueError("unsupported Teacher Bank schema")
    if bank.get("status") != "complete":
        raise ValueError("Teacher Bank must be complete before comparison")

    rows: list[dict[str, object]] = []
    for entry in bank["teachers"]:
        if entry.get("status") != "complete":
            raise ValueError("Teacher Bank contains a rejected Teacher")
        run_dir = bank_path.parent / str(entry["run_dir"])
        actor_path = bank_path.parent / str(entry["actor_checkpoint"])
        pid_reference = entry.get("pid_oracle")
        if not isinstance(pid_reference, dict):
            raise ValueError(f"missing PID oracle: {entry['plant_id']}")
        pid_path = Path(str(pid_reference["path"]))
        if sha256_file(pid_path) != str(pid_reference["sha256"]):
            raise ValueError(f"PID oracle hash mismatch: {pid_path}")
        output = run_dir / "comparison_vs_pid"
        report_path = output / "comparison.json"
        plot_path = output / "controller_comparison.png"
        skipped = report_path.is_file() and plot_path.is_file() and not args.force
        if not skipped:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/15_compare_teacher_pid.py"),
                    "--teacher-checkpoint",
                    str(actor_path),
                    "--pid-report",
                    str(pid_path),
                    "--output",
                    str(output),
                    "--device",
                    args.device,
                ],
                cwd=ROOT,
                check=True,
            )
        rows.append(
            {
                "plant_id": entry["plant_id"],
                "quality_region": entry["quality_region"],
                "skipped_existing": skipped,
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "plot": str(plot_path),
                "plot_sha256": sha256_file(plot_path),
            }
        )

    report = {
        "schema_version": "teacher_bank_pid_comparison_index_v1",
        "status": "complete",
        "source": git_source_revision(),
        "teacher_bank": {
            "path": str(bank_path),
            "sha256": sha256_file(bank_path),
        },
        "comparison_count": len(rows),
        "rows": rows,
    }
    index_path = bank_path.parent / "comparison_index.json"
    _write_json(index_path, report)
    print(json.dumps({"status": "complete", "comparisons": len(rows)}, indent=2))
    print(f"index={index_path}")


if __name__ == "__main__":
    main()
