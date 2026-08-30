"""Audit staged Teacher/Student results and write the root artifact index."""

# ruff: noqa: E402 -- direct path execution needs the repository root first.

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.provenance import git_source_revision, sha256_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=ROOT / "results/teacher_student_pipeline",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return payload


def _write_results_markdown(
    path: Path,
    teacher_bank: dict[str, object],
    distillation: dict[str, object],
    comparison: dict[str, object],
) -> None:
    overall = comparison["overall"]
    rows = []
    for label in ("raw", "PID", "RL Teacher", "Student"):
        metrics = overall[label]
        rows.append(
            f"| {label} | {metrics['mean_tracking_rmse_deg_s']:.6f} | "
            f"{metrics['maximum_peak_error_deg_s']:.6f} | "
            f"{metrics['mean_requested_force_total_variation_n']:.6f} |"
        )
    student = comparison["student_checkpoint"]
    content = "\n".join(
        [
            "# P-channel Teacher-to-Student Results",
            "",
            "## Scope",
            "",
            f"- {teacher_bank['accepted_teacher_count']} fixed-aircraft PID-guided TD3 Teachers: Level 1/2/3 = 5/3/3.",
            f"- One theta-routed linear MoE Student with {student['parameter_count']} trainable parameters.",
            "- Round 0 is Teacher-driven; rounds 1 and 2 are Student-driven and Teacher-labeled.",
            "- Validation contains 6 unseen commands per Teacher-Bank aircraft, 66 closed-loop pairs total.",
            "- Raw observation history windows, TCN, and GRU are not used.",
            "- This run does not claim zero-shot generalization to unseen aircraft parameters.",
            "",
            "## Closed-loop result",
            "",
            "| Controller | Mean tracking RMSE (deg/s) | Maximum peak error (deg/s) | Mean requested-force TV (N) |",
            "| --- | ---: | ---: | ---: |",
            *rows,
            "",
            "The Student is within 10% of its matching Teacher on all 66 validation pairs. The",
            "current RL Teachers are effectively equivalent to the tuned PID baselines; this run",
            "demonstrates unified distillation, not a meaningful RL-over-PID performance gain.",
            "",
            f"Selected Student-driven round: {distillation['best_round']}.",
            "",
            "## Open first",
            "",
            "- [All-aircraft summary](03_final_comparison/raw_pid_teacher_student_summary.png)",
            "- [Challenging aircraft time-domain curves](03_final_comparison/aircraft/train_core-0867/raw_pid_teacher_student.png)",
            "- [Student-driven distillation progress](02_student_driven_distillation/distillation_progress.png)",
            "- [Final controller metrics](03_final_comparison/controller_metrics.csv)",
            "- [Machine-readable comparison](03_final_comparison/comparison_report.json)",
            "- [Completion audit](completion_audit.json)",
            "- [Artifact index](artifact_index.csv)",
            "",
            "## Directory layout",
            "",
            "```text",
            "01_teachers/                    Teacher reports, deployment actors, recovery checkpoints, plots",
            "02_student_driven_distillation/ round datasets, Students, routing reports, closed-loop evaluations",
            "03_final_comparison/            Raw/PID/RL Teacher/Student plots and aggregate metrics",
            "diagnostics/                    rejected or historical experiments, excluded from formal conclusions",
            "```",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    root = args.pipeline_root.resolve()
    repository_root = root.parent.parent

    def repository_path(value: object) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else repository_root / path

    teacher_bank_path = root / "01_teachers/teacher_bank.json"
    teacher_summary_path = root / "01_teachers/summary/teacher_bank_summary.json"
    distillation_path = root / "02_student_driven_distillation/pipeline_report.json"
    comparison_path = root / "03_final_comparison/comparison_report.json"
    teacher_bank = _load_json(teacher_bank_path)
    teacher_summary = _load_json(teacher_summary_path)
    distillation = _load_json(distillation_path)
    comparison = _load_json(comparison_path)

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, evidence: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    teachers = teacher_bank.get("teachers")
    if not isinstance(teachers, list):
        raise ValueError("Teacher Bank has no teacher list")
    check(
        "teacher_bank_complete",
        teacher_bank.get("status") == "complete"
        and teacher_bank.get("accepted_teacher_count") == 11
        and len(teachers) == 11,
        {
            "status": teacher_bank.get("status"),
            "accepted": teacher_bank.get("accepted_teacher_count"),
            "total": len(teachers),
        },
    )
    check(
        "teacher_quality_regions_covered",
        teacher_bank.get("quality_region_counts")
        == {"level_1": 5, "level_2": 3, "level_3": 3},
        teacher_bank.get("quality_region_counts"),
    )
    teacher_contract_rows: list[dict[str, object]] = []
    teacher_artifact_rows: list[dict[str, object]] = []
    for entry in teachers:
        report_path = teacher_bank_path.parent / str(entry["report"])
        report = _load_json(report_path)
        contract = report.get("actor_observation_contract")
        actor_path = teacher_bank_path.parent / str(entry["actor_checkpoint"])
        training_checkpoint_path = report_path.parent / "training_checkpoint.pt"
        evaluation_path = report_path.parent / "evaluation.json"
        comparison_plot_path = (
            report_path.parent / "comparison_vs_pid/controller_comparison.png"
        )
        actor_payload_ok = False
        actor_checkpoint_error: str | None = None
        if actor_path.is_file():
            try:
                actor_payload = torch.load(
                    actor_path, map_location="cpu", weights_only=True
                )
                actor_payload_contract = actor_payload.get("actor_observation_contract")
                actor_payload_ok = (
                    actor_payload.get("schema_version") == "specialist_actor_v1"
                    and actor_payload.get("algorithm") == "pid_guided_td3"
                    and actor_payload.get("plant", {}).get("plant_id")
                    == entry["plant_id"]
                    and int(actor_payload.get("actor_observation_dim", -1)) == 7
                    and isinstance(actor_payload_contract, dict)
                    and int(actor_payload_contract.get("raw_history_steps", -1)) == 0
                    and not bool(
                        actor_payload_contract.get("uses_raw_history_window", True)
                    )
                    and isinstance(actor_payload.get("actor"), dict)
                    and bool(actor_payload["actor"])
                )
            except Exception as error:  # pragma: no cover - audit failure path
                actor_checkpoint_error = f"{type(error).__name__}: {error}"
        training_checkpoint_ok = False
        training_checkpoint_error: str | None = None
        if training_checkpoint_path.is_file():
            try:
                training_payload = torch.load(
                    training_checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )
                training_checkpoint_ok = (
                    training_payload.get("schema_version")
                    == "pid_guided_td3_checkpoint_v1"
                    and training_payload.get("algorithm") == "pid_guided_td3"
                    and training_payload.get("plant", {}).get("plant_id")
                    == entry["plant_id"]
                    and int(training_payload.get("actor_observation_dim", -1)) == 7
                    and isinstance(training_payload.get("actor"), dict)
                    and bool(training_payload["actor"])
                    and isinstance(training_payload.get("critic"), dict)
                    and bool(training_payload["critic"])
                    and isinstance(training_payload.get("target_actor"), dict)
                    and bool(training_payload["target_actor"])
                    and isinstance(training_payload.get("target_critic"), dict)
                    and bool(training_payload["target_critic"])
                    and int(training_payload.get("updates", 0)) > 0
                    and int(training_payload.get("actor_updates", 0)) > 0
                )
                del training_payload
                gc.collect()
            except Exception as error:  # pragma: no cover - audit failure path
                training_checkpoint_error = f"{type(error).__name__}: {error}"
        contract_ok = (
            isinstance(contract, dict)
            and int(contract.get("raw_history_steps", -1)) == 0
            and not bool(contract.get("uses_raw_history_window", True))
            and int(report.get("actor_observation_dim", -1)) == 7
            and report.get("actor_receives_theta") is False
            and report.get("accepted_for_distillation") is True
            and report.get("online_actor_updates", 0) > 0
        )
        teacher_contract_rows.append(
            {
                "plant_id": entry["plant_id"],
                "passed": contract_ok,
                "actor_observation_dim": report.get("actor_observation_dim"),
                "raw_history_steps": (
                    contract.get("raw_history_steps")
                    if isinstance(contract, dict)
                    else None
                ),
                "actor_receives_theta": report.get("actor_receives_theta"),
                "online_actor_updates": report.get("online_actor_updates"),
                "actor_checkpoint_loadable": actor_payload_ok,
            }
        )
        teacher_artifact_rows.append(
            {
                "plant_id": entry["plant_id"],
                "actor_checkpoint": str(actor_path),
                "actor_checkpoint_exists": actor_path.is_file(),
                "actor_checkpoint_loadable": actor_payload_ok,
                "actor_checkpoint_error": actor_checkpoint_error,
                "training_checkpoint": str(training_checkpoint_path),
                "training_checkpoint_exists": training_checkpoint_path.is_file(),
                "training_checkpoint_loadable": training_checkpoint_ok,
                "training_checkpoint_error": training_checkpoint_error,
                "training_checkpoint_bytes": (
                    training_checkpoint_path.stat().st_size
                    if training_checkpoint_path.is_file()
                    else 0
                ),
                "evaluation_exists": evaluation_path.is_file(),
                "comparison_plot_exists": comparison_plot_path.is_file(),
            }
        )
    check(
        "teacher_actor_contracts",
        all(
            row["passed"] and row["actor_checkpoint_loadable"]
            for row in teacher_contract_rows
        ),
        teacher_contract_rows,
    )
    check(
        "teacher_artifact_completeness",
        all(
            row["actor_checkpoint_exists"]
            and row["actor_checkpoint_loadable"]
            and row["training_checkpoint_exists"]
            and row["training_checkpoint_loadable"]
            and int(row["training_checkpoint_bytes"]) > 0
            and row["evaluation_exists"]
            and row["comparison_plot_exists"]
            for row in teacher_artifact_rows
        ),
        teacher_artifact_rows,
    )
    check(
        "teacher_summary_complete",
        teacher_summary.get("status") == "complete"
        and teacher_summary.get("aggregate", {}).get("aircraft_count") == 11,
        teacher_summary.get("aggregate"),
    )

    distillation_config = distillation.get("config")
    if not isinstance(distillation_config, dict):
        raise ValueError("distillation report has no config")
    rounds = distillation.get("rounds")
    if not isinstance(rounds, list):
        raise ValueError("distillation report has no rounds")
    check(
        "student_distillation_quality_gate",
        distillation.get("status") == "complete"
        and bool(distillation.get("quality_gate", {}).get("passed", False)),
        {
            "status": distillation.get("status"),
            "quality_gate": distillation.get("quality_gate"),
            "best_round": distillation.get("best_round"),
        },
    )
    check(
        "command_holdout_scope",
        distillation_config.get("split_strategy") == "all_aircraft_command_holdout",
        distillation_config.get("split_strategy"),
    )
    expected_drivers = ["teacher", "student", "student"]
    observed_drivers = [row.get("driver") for row in rounds]
    check(
        "student_driven_rounds",
        observed_drivers == expected_drivers
        and all(
            int(row.get("dataset", {}).get("new_rows", 1)) > 0 for row in rounds[1:]
        ),
        observed_drivers,
    )

    dataset_audit_rows: list[dict[str, object]] = []
    expected_teacher_actor_hashes: dict[str, set[str]] = {}
    for round_index, round_row in enumerate(rounds):
        manifest_path = repository_path(round_row.get("dataset_manifest"))
        manifest_exists = manifest_path.is_file()
        manifest = _load_json(manifest_path) if manifest_exists else {}
        shards = manifest.get("shards")
        shard_rows: list[dict[str, object]] = []
        if isinstance(shards, list):
            for shard in shards:
                plant_id = str(shard.get("plant_id", ""))
                teacher_actor_hash = str(shard.get("teacher_actor_sha256", ""))
                if plant_id and teacher_actor_hash:
                    expected_teacher_actor_hashes.setdefault(plant_id, set()).add(
                        teacher_actor_hash
                    )
                shard_path = (
                    manifest_path.parent / str(shard.get("path", ""))
                ).resolve()
                exists = shard_path.is_file()
                expected_hash = str(shard.get("sha256", ""))
                hash_matches = (
                    exists
                    and bool(expected_hash)
                    and sha256_file(shard_path) == expected_hash
                )
                shard_rows.append(
                    {
                        "path": str(shard_path),
                        "exists": exists,
                        "sha256_matches": hash_matches,
                        "rows": int(shard.get("rows", 0)),
                        "collection_round": int(shard.get("collection_round", -1)),
                        "driver": str(shard.get("driver", "")),
                    }
                )
        current_round_shards = [
            shard for shard in shard_rows if shard["collection_round"] == round_index
        ]
        embedded_dataset = round_row.get("dataset")
        if not isinstance(embedded_dataset, dict):
            embedded_dataset = {}
        total_rows = sum(int(shard["rows"]) for shard in shard_rows)
        current_round_rows = sum(int(shard["rows"]) for shard in current_round_shards)
        expected_new_rows = embedded_dataset.get("new_rows")
        new_rows_match = (
            round_index == 0 or int(expected_new_rows or -1) == current_round_rows
        )
        row_counts_match = (
            int(manifest.get("row_count", -1)) == total_rows
            and int(manifest.get("train_rows", -1))
            + int(manifest.get("validation_rows", -1))
            == total_rows
        )
        expected_driver = expected_drivers[round_index]
        dataset_audit_rows.append(
            {
                "round": round_index,
                "manifest": str(manifest_path),
                "manifest_exists": manifest_exists,
                "status": manifest.get("status"),
                "split_strategy": manifest.get("split_strategy"),
                "shard_count": len(shard_rows),
                "current_round_shard_count": len(current_round_shards),
                "all_shards_exist_and_match_hash": bool(shard_rows)
                and all(
                    shard["exists"] and shard["sha256_matches"] for shard in shard_rows
                ),
                "row_counts_match": row_counts_match,
                "new_rows_match": new_rows_match,
                "current_round_driver_matches": bool(current_round_shards)
                and all(
                    shard["driver"] == expected_driver for shard in current_round_shards
                ),
            }
        )
    check(
        "student_driven_dataset_integrity",
        len(dataset_audit_rows) == 3
        and all(
            row["manifest_exists"]
            and row["status"] == "complete"
            and row["split_strategy"] == "all_aircraft_command_holdout"
            and row["all_shards_exist_and_match_hash"]
            and row["row_counts_match"]
            and row["new_rows_match"]
            and row["current_round_driver_matches"]
            for row in dataset_audit_rows
        ),
        dataset_audit_rows,
    )
    teacher_actor_hash_rows: list[dict[str, object]] = []
    for entry in teachers:
        plant_id = str(entry["plant_id"])
        actor_path = teacher_bank_path.parent / str(entry["actor_checkpoint"])
        expected_hashes = sorted(expected_teacher_actor_hashes.get(plant_id, set()))
        actual_hash = sha256_file(actor_path) if actor_path.is_file() else None
        teacher_actor_hash_rows.append(
            {
                "plant_id": plant_id,
                "actor_checkpoint": str(actor_path),
                "expected_sha256_values": expected_hashes,
                "actual_sha256": actual_hash,
                "matches": len(expected_hashes) == 1
                and actual_hash == expected_hashes[0],
            }
        )
    check(
        "teacher_actor_hashes_match_distillation_labels",
        len(teacher_actor_hash_rows) == 11
        and all(row["matches"] for row in teacher_actor_hash_rows),
        teacher_actor_hash_rows,
    )

    final_checkpoint_reference = distillation.get("final_checkpoint")
    if not isinstance(final_checkpoint_reference, dict):
        raise ValueError("distillation report has no final checkpoint")
    final_checkpoint = repository_path(final_checkpoint_reference["path"])
    checkpoint_hash_matches = final_checkpoint.is_file() and sha256_file(
        final_checkpoint
    ) == str(final_checkpoint_reference["sha256"])
    student_payload = torch.load(
        final_checkpoint, map_location="cpu", weights_only=True
    )
    temporal_contract = student_payload.get("temporal_contract")
    check(
        "student_checkpoint_contract",
        checkpoint_hash_matches
        and student_payload.get("schema_version")
        == "theta_routed_linear_moe_student_v2"
        and student_payload.get("student_architecture") == "theta_routed_linear_moe"
        and student_payload.get("control_feature_indices") == [3, 4, 5]
        and isinstance(temporal_contract, dict)
        and int(temporal_contract.get("raw_history_steps", -1)) == 0
        and not bool(temporal_contract.get("uses_raw_history_window", True))
        and temporal_contract.get("router_input") == "normalized_aircraft_theta_only",
        {
            "schema_version": student_payload.get("schema_version"),
            "parameter_count": student_payload.get("parameter_count"),
            "expert_count": student_payload.get("expert_count"),
            "control_feature_indices": student_payload.get("control_feature_indices"),
            "temporal_contract": temporal_contract,
            "sha256_matches": checkpoint_hash_matches,
        },
    )
    check(
        "final_controller_comparison",
        comparison.get("status") == "complete"
        and bool(comparison.get("self_check", {}).get("passed", False))
        and comparison.get("teacher_bank", {}).get("teacher_count") == 11,
        {
            "status": comparison.get("status"),
            "self_check": comparison.get("self_check", {}).get("passed"),
            "distillation": comparison.get("distillation"),
        },
    )
    overall = comparison.get("overall")
    distillation_effect = comparison.get("distillation", {}).get("overall", {})
    if not isinstance(overall, dict):
        overall = {}
    raw_metrics = overall.get("raw")
    student_metrics = overall.get("Student")
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}
    if not isinstance(student_metrics, dict):
        student_metrics = {}
    if not isinstance(distillation_effect, dict):
        distillation_effect = {}
    check(
        "distillation_effect_demonstrated",
        int(raw_metrics.get("pair_count", -1)) == 66
        and int(student_metrics.get("pair_count", -1)) == 66
        and float(student_metrics.get("mean_tracking_rmse_deg_s", float("inf")))
        < float(raw_metrics.get("mean_tracking_rmse_deg_s", float("-inf")))
        and float(student_metrics.get("maximum_peak_error_deg_s", float("inf"))) <= 10.0
        and float(student_metrics.get("mean_force_saturation_fraction", float("inf")))
        <= 0.01
        and float(
            distillation_effect.get("student_within_10pct_teacher_rate", float("-inf"))
        )
        == 1.0,
        {
            "raw": raw_metrics,
            "student": student_metrics,
            "distillation": distillation_effect,
        },
    )

    _write_results_markdown(root / "RESULTS.md", teacher_bank, distillation, comparison)

    artifact_rows = [
        ("human_results_summary", root / "RESULTS.md"),
        ("teacher_bank", teacher_bank_path),
        ("teacher_summary", teacher_summary_path),
        (
            "teacher_summary_plot",
            root / "01_teachers/summary/teacher_bank_summary.png",
        ),
        ("distillation_report", distillation_path),
        (
            "distillation_progress_plot",
            root / "02_student_driven_distillation/distillation_progress.png",
        ),
        (
            "distillation_round_metrics",
            root / "02_student_driven_distillation/round_metrics.csv",
        ),
        ("final_student", final_checkpoint),
        ("final_comparison", comparison_path),
        (
            "final_summary_plot",
            root / "03_final_comparison/raw_pid_teacher_student_summary.png",
        ),
        (
            "final_controller_metrics",
            root / "03_final_comparison/controller_metrics.csv",
        ),
    ]
    for entry in teachers:
        run_dir = teacher_bank_path.parent / str(entry["run_dir"])
        plant_id = str(entry["plant_id"])
        artifact_rows.extend(
            [
                (f"teacher_report:{plant_id}", run_dir / "report.json"),
                (f"teacher_actor:{plant_id}", run_dir / "teacher_actor.pt"),
                (
                    f"teacher_training_checkpoint:{plant_id}",
                    run_dir / "training_checkpoint.pt",
                ),
                (f"teacher_evaluation:{plant_id}", run_dir / "evaluation.json"),
                (
                    f"teacher_pid_comparison_plot:{plant_id}",
                    run_dir / "comparison_vs_pid/controller_comparison.png",
                ),
                (
                    f"final_response_plot:{plant_id}",
                    root
                    / "03_final_comparison/aircraft"
                    / plant_id
                    / "raw_pid_teacher_student.png",
                ),
            ]
        )
    for round_index, round_row in enumerate(rounds):
        round_dir = (
            root
            / "02_student_driven_distillation"
            / (
                "round_000_teacher_driven"
                if round_index == 0
                else f"round_{round_index:03d}_student_driven"
            )
        )
        artifact_rows.extend(
            [
                (
                    f"distillation_dataset:{round_index}",
                    repository_path(round_row["dataset_manifest"]),
                ),
                (
                    f"distillation_student:{round_index}",
                    round_dir / "student/student.pt",
                ),
                (
                    f"distillation_routing_report:{round_index}",
                    round_dir / "student/routing_report.json",
                ),
                (
                    f"distillation_closed_loop:{round_index}",
                    round_dir / "evaluation/evaluation.json",
                ),
            ]
        )
    artifacts = [
        {
            "name": name,
            "path": str(path),
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else 0,
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in artifact_rows
    ]
    check(
        "required_artifacts_exist",
        all(row["exists"] and int(row["bytes"]) > 0 for row in artifacts),
        artifacts,
    )
    final_response_artifacts = [
        row for row in artifacts if str(row["name"]).startswith("final_response_plot:")
    ]
    check(
        "final_response_plots_complete",
        len(final_response_artifacts) == 11
        and all(
            row["exists"] and int(row["bytes"]) > 10_000
            for row in final_response_artifacts
        ),
        final_response_artifacts,
    )

    passed = all(row["passed"] for row in checks)
    audit = {
        "schema_version": "teacher_student_completion_audit_v2",
        "status": "complete" if passed else "self_check_failed",
        "source": git_source_revision(),
        "scope": {
            "channel": "P-channel roll-rate control",
            "teacher_bank_aircraft": 11,
            "validation": "unseen commands on all Teacher-Bank aircraft",
            "zero_shot_unseen_aircraft_claimed": False,
            "raw_history_window_used": False,
        },
        "checks": checks,
        "artifacts": artifacts,
        "known_diagnostics": {
            "dense_aircraft_holdout_failure": str(
                root / "diagnostics/dense_baseline_6_teacher_6168f64a"
            ),
            "sparse_moe_aircraft_holdout_failure": str(
                root / "diagnostics/partial_theta_moe_sparse_temperature_002"
            ),
            "damping_floor_rejected": str(root / "benchmarks/damping_floor_probe_002"),
        },
    }
    audit_path = root / "completion_audit.json"
    _write_json(audit_path, audit)
    index_path = root / "artifact_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("name", "path", "exists", "bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(artifacts)
    root_report = {
        "schema_version": "teacher_student_pipeline_v1",
        "status": audit["status"],
        "source": git_source_revision(),
        "scope": audit["scope"],
        "teacher_bank": {
            "path": str(teacher_bank_path),
            "sha256": sha256_file(teacher_bank_path),
            "accepted_teacher_count": teacher_bank.get("accepted_teacher_count"),
        },
        "distillation": {
            "path": str(distillation_path),
            "sha256": sha256_file(distillation_path),
            "best_round": distillation.get("best_round"),
            "quality_gate": distillation.get("quality_gate"),
        },
        "final_comparison": {
            "path": str(comparison_path),
            "sha256": sha256_file(comparison_path),
            "distillation": comparison.get("distillation"),
        },
        "completion_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
        },
        "artifact_index": str(index_path),
    }
    _write_json(root / "pipeline_report.json", root_report)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "checks_passed": sum(row["passed"] for row in checks),
                "check_count": len(checks),
            },
            indent=2,
        )
    )
    print(f"audit={audit_path}")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
