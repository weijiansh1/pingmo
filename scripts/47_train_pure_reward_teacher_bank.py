"""Train and assemble a small, consistent pure-reward TD3 Teacher Bank."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.roll_rate_commands import RandomCommandDistribution  # noqa: E402
from src.experiments.exploratory_sac import load_persisted_records  # noqa: E402
from src.teacher.specialist.pure_td3_trainer import (  # noqa: E402
    PureRewardTD3Config,
    train_pure_reward_td3,
)
from src.teacher.specialist.trainer import (  # noqa: E402
    SpecialistTrainingConfig,
    load_specialist_actor,
)
from src.utils.provenance import git_source_revision, sha256_file  # noqa: E402


DEFAULT_PLANT_IDS = (
    "train_core-0306",
    "train_core-0312",
    "train_core-0334",
    "train_core-0760",
    "train_core-0803",
    "train_core-1187",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=(
            ROOT
            / "data/aircraft/generated/p_channel_library_iv_a_manual_v1/plants.jsonl"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant-id", action="append")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=80_000)
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--evaluation-interval-steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--maximum-mean-rmse-deg-s", type=float, default=1.5)
    parser.add_argument("--maximum-peak-error-deg-s", type=float, default=5.0)
    parser.add_argument("--maximum-mean-tv-rate-n-s", type=float, default=12.0)
    parser.add_argument("--maximum-saturation-fraction", type=float, default=0.01)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _training_request(
    *,
    library: Path,
    output: Path,
    plant_id: str,
    seed: int,
    steps: int,
    warmup_steps: int,
    evaluation_interval_steps: int,
    device: str,
) -> dict[str, object]:
    return {
        "library": str(library),
        "output": str(output),
        "plant_id": plant_id,
        "seed": seed,
        "steps": steps,
        "warmup_steps": warmup_steps,
        "evaluation_interval_steps": evaluation_interval_steps,
        "device": device,
    }


def _report_matches_request(
    report: dict[str, object], request: dict[str, object]
) -> bool:
    environment = report.get("environment_config")
    td3 = report.get("td3_config")
    if not isinstance(environment, dict) or not isinstance(td3, dict):
        return False
    return (
        report.get("status") == "complete"
        and report.get("plant_id") == request["plant_id"]
        and int(report.get("steps", -1)) == request["steps"]
        and int(td3.get("seed", -1)) == request["seed"]
        and int(td3.get("warmup_steps", -1)) == request["warmup_steps"]
        and int(td3.get("evaluation_interval_steps", -1))
        == request["evaluation_interval_steps"]
        and float(environment.get("episode_duration_s", -1.0)) == 30.0
        and int(environment.get("requested_action_history_steps", -1)) == 26
        and environment.get("include_reference_derivative") is True
    )


def _train_worker(request: dict[str, object]) -> dict[str, object]:
    output = Path(str(request["output"]))
    report_path = output / "report.json"
    if report_path.is_file():
        existing = _read_json(report_path)
        if _report_matches_request(existing, request):
            return {
                "plant_id": request["plant_id"],
                "report": str(report_path),
                "resumed": True,
            }
        raise ValueError(f"existing run does not match request: {report_path}")

    library = Path(str(request["library"]))
    plant_id = str(request["plant_id"])
    record = load_persisted_records(library, [plant_id])[0]
    seed = int(request["seed"])
    steps = int(request["steps"])
    warmup_steps = int(request["warmup_steps"])
    device = str(request["device"])
    environment = SpecialistTrainingConfig(
        episode_duration_s=30.0,
        command_mode="extended",
        history_steps=0,
        requested_action_history_steps=26,
        include_actor_actuator_state=True,
        include_reference_derivative=True,
        critic_include_episode_progress=False,
        critic_include_command_context=False,
        reference_delay_mode="match_plant_transport_delay",
        enforce_odd_policy=True,
        odd_policy_projection_stage="training",
        seed=seed,
        device=device,
    )
    td3 = PureRewardTD3Config(
        total_steps=steps,
        warmup_steps=warmup_steps,
        batch_size=256,
        replay_capacity=200_000,
        exploration_std_initial=0.20,
        exploration_std_final=0.02,
        exploration_decay_steps=max(steps - warmup_steps, 1),
        network_width=128,
        residual_blocks=2,
        gamma=0.9995,
        actor_learning_rate=3e-5,
        critic_learning_rate=3e-4,
        critic_warmup_updates=500,
        evaluation_interval_steps=int(request["evaluation_interval_steps"]),
        randomize_training_commands=True,
        random_command_sequence=True,
        random_sequence_segment_duration_range_s=(2.0, 5.0),
        long_dwell_step_probability=0.5,
        long_dwell_duration_range_s=(15.0, 30.0),
        random_command_distribution=RandomCommandDistribution(
            frequency_range_hz=(0.05, 1.50)
        ),
        seed=seed,
        device=device,
    )
    train_pure_reward_td3(
        record,
        output,
        environment,
        td3,
        library_path=library,
    )
    return {
        "plant_id": plant_id,
        "report": str(report_path),
        "resumed": False,
    }


def _resolve_artifact(path_value: object) -> Path:
    path = Path(str(path_value))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _validation_quality(
    validation: dict[str, object],
    environment: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    duration_s = float(environment["episode_duration_s"])
    tv_rate = float(validation["mean_requested_force_total_variation_n"]) / duration_s
    checks = {
        "tracking_improvement_rate": float(validation["tracking_improvement_rate"])
        >= 1.0,
        "mean_tracking_rmse": float(validation["mean_tracking_rmse_deg_s"])
        <= args.maximum_mean_rmse_deg_s,
        "peak_tracking_error": float(validation["maximum_peak_error_deg_s"])
        <= args.maximum_peak_error_deg_s,
        "requested_force_tv_rate": tv_rate <= args.maximum_mean_tv_rate_n_s,
        "force_saturation": float(validation["mean_force_saturation_fraction"])
        <= args.maximum_saturation_fraction,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            **validation,
            "mean_requested_force_total_variation_rate_n_s": tv_rate,
        },
        "thresholds": {
            "minimum_tracking_improvement_rate": 1.0,
            "maximum_mean_tracking_rmse_deg_s": args.maximum_mean_rmse_deg_s,
            "maximum_peak_error_deg_s": args.maximum_peak_error_deg_s,
            "maximum_mean_requested_force_total_variation_rate_n_s": (
                args.maximum_mean_tv_rate_n_s
            ),
            "maximum_saturation_fraction": args.maximum_saturation_fraction,
        },
    }


def _select_validation_checkpoint(
    report: dict[str, object], args: argparse.Namespace
) -> tuple[dict[str, object], dict[str, object], str]:
    environment = report.get("environment_config")
    history = report.get("learning_curve")
    if (
        not isinstance(environment, dict)
        or not isinstance(history, list)
        or not history
    ):
        raise ValueError(f"missing validation history: {report.get('plant_id')}")
    candidates: list[tuple[dict[str, object], dict[str, object]]] = []
    for point in history:
        if not isinstance(point, dict):
            raise ValueError(f"invalid validation point: {report.get('plant_id')}")
        quality = _validation_quality(point, environment, args)
        if quality["passed"]:
            candidates.append((point, quality))
    if candidates:
        point, quality = min(
            candidates,
            key=lambda item: float(item[0]["mean_episode_cost"]),
        )
        return point, quality, "minimum_cost_among_quality_eligible_checkpoints"

    fallback = report.get("best_validation")
    if not isinstance(fallback, dict):
        raise ValueError(f"missing best validation: {report.get('plant_id')}")
    return (
        fallback,
        _validation_quality(fallback, environment, args),
        "minimum_cost_fallback_no_quality_eligible_checkpoint",
    )


def _assemble_bank(
    args: argparse.Namespace,
    requests: list[dict[str, object]],
    destination: Path,
) -> dict[str, object]:
    actor_dir = destination / "actors"
    actor_dir.mkdir(parents=True, exist_ok=True)
    teachers: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    expected_contract: dict[str, object] | None = None
    for request in requests:
        report_path = Path(str(request["output"])) / "report.json"
        report = _read_json(report_path)
        selected_validation, quality, selection_method = _select_validation_checkpoint(
            report, args
        )
        checkpoint_dir = _resolve_artifact(
            report["artifacts"]["validation_checkpoints"]
        )
        selected_path = checkpoint_dir / (
            f"actor_step_{int(selected_validation['step']):08d}.pt"
        )
        if not selected_path.is_file():
            raise FileNotFoundError(selected_path)
        _, record, _, checkpoint = load_specialist_actor(selected_path, device="cpu")
        contract = checkpoint.get("actor_observation_contract")
        if not isinstance(contract, dict):
            raise ValueError(f"checkpoint has no observation contract: {selected_path}")
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError(
                "selected pure-reward Teachers have different observations"
            )

        entry = {
            "plant_id": record.plant_id,
            "split": record.split,
            "quality_region": record.quality_region,
            "status": "complete" if quality["passed"] else "quality_gate_failed",
            "quality_gate": quality,
            "selected_validation_step": int(selected_validation["step"]),
            "selection_method": selection_method,
            "source_report": os.path.relpath(report_path, destination),
            "source_report_sha256": sha256_file(report_path),
            "source_checkpoint": str(selected_path),
        }
        if not quality["passed"]:
            rejected.append(entry)
            continue
        actor_path = actor_dir / f"{record.plant_id}.pt"
        shutil.copy2(selected_path, actor_path)
        entry.update(
            {
                "actor_checkpoint": os.path.relpath(actor_path, destination),
                "actor_checkpoint_sha256": sha256_file(actor_path),
            }
        )
        teachers.append(entry)

    region_counts = Counter(str(entry["quality_region"]) for entry in teachers)
    bank = {
        "schema_version": "specialist_teacher_bank_v1",
        "status": "complete" if len(teachers) >= 2 else "quality_gate_failed",
        "algorithm": "pure_reward_td3_best_validation",
        "source": git_source_revision(),
        "library": {
            "path": str(args.library.resolve()),
            "sha256": sha256_file(args.library),
        },
        "attempted_teacher_count": len(requests),
        "teacher_count": len(teachers),
        "accepted_teacher_count": len(teachers),
        "rejected_teacher_count": len(rejected),
        "quality_region_counts": dict(region_counts),
        "actor_observation_contract": expected_contract,
        "selection_contract": {
            "checkpoint_metric": (
                "minimum fixed-validation mean episode cost among checkpoints "
                "passing the predeclared quality gate"
            ),
            "fallback": "minimum-cost checkpoint when no checkpoint passes",
            "quality_gate_uses_time_normalized_force_tv": True,
        },
        "teachers": teachers,
        "rejected_teachers": rejected,
    }
    _write_json(destination / "teacher_bank.json", bank)
    return bank


def main() -> None:
    args = _parse_args()
    if args.workers <= 0 or args.steps <= 1 or args.warmup_steps >= args.steps:
        raise ValueError("invalid Teacher Bank training dimensions")
    plant_ids = tuple(args.plant_id or DEFAULT_PLANT_IDS)
    if len(set(plant_ids)) != len(plant_ids):
        raise ValueError("Teacher Bank plant IDs must be unique")
    records = load_persisted_records(args.library, list(plant_ids))
    if {record.plant_id for record in records} != set(plant_ids):
        raise ValueError("not all requested Teacher aircraft exist in the library")

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    steps = 1_000 if args.smoke_test else args.steps
    warmup_steps = 100 if args.smoke_test else args.warmup_steps
    evaluation_interval = 1_000 if args.smoke_test else args.evaluation_interval_steps
    requests = [
        _training_request(
            library=args.library.resolve(),
            output=destination / "runs" / plant_id,
            plant_id=plant_id,
            seed=args.seed + index,
            steps=steps,
            warmup_steps=warmup_steps,
            evaluation_interval_steps=evaluation_interval,
            device=args.device,
        )
        for index, plant_id in enumerate(plant_ids)
    ]
    _write_json(
        destination / "request.json",
        {
            "status": "running",
            "smoke_test": args.smoke_test,
            "workers": args.workers,
            "requests": requests,
        },
    )
    context = mp.get_context("spawn")
    completed: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(_train_worker, request): request for request in requests}
        for future in as_completed(futures):
            result = future.result()
            completed.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

    if args.smoke_test:
        result = {
            "status": "smoke_test_complete",
            "completed": sorted(completed, key=lambda row: str(row["plant_id"])),
        }
    else:
        bank = _assemble_bank(args, requests, destination)
        result = {
            "status": bank["status"],
            "attempted_teacher_count": bank["attempted_teacher_count"],
            "accepted_teacher_count": bank["accepted_teacher_count"],
            "rejected_teacher_count": bank["rejected_teacher_count"],
            "teacher_bank": str(destination / "teacher_bank.json"),
        }
    _write_json(destination / "request.json", {**result, "requests": requests})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
