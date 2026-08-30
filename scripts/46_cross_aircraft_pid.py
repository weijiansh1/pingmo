"""Apply one aircraft's frozen PID gains to another aircraft without retuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import PIDGains, RollRatePIDPolicy  # noqa: E402
from src.envs.roll_rate_commands import (  # noqa: E402
    SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION,
    RollRateCommandProfile,
    specialist_evaluation_commands,
    specialist_independent_test_commands,
)
from src.experiments.exploratory_sac import load_persisted_records  # noqa: E402
from src.teacher.specialist.trainer import (  # noqa: E402
    SpecialistTrainingConfig,
    build_specialist_env,
    rollout_policy,
    tracking_metrics,
)
from src.utils.provenance import git_source_revision, sha256_file  # noqa: E402


COLORS = {
    "target_local": "#2468a2",
    "source_transfer": "#b3262e",
}
PID_CONTRACT_FIELDS = (
    "plant_dt_s",
    "policy_dt_s",
    "reference_natural_frequency_rad_s",
    "reference_damping_ratio",
    "reference_delay_mode",
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
    parser.add_argument("--source-pid-report", type=Path, required=True)
    parser.add_argument("--target-pid-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--evaluation-seed", type=int, default=20260902)
    parser.add_argument(
        "--command-suite",
        choices=("independent-test", "validation"),
        default="independent-test",
    )
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "complete":
        raise ValueError(f"PID report is not complete: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _pid_policy(gains: PIDGains, environment: object) -> RollRatePIDPolicy:
    return RollRatePIDPolicy(
        gains,
        policy_dt_s=environment.policy_dt_s,
        command_scale_rad_s=environment.command_scale_rad_s,
        integral_error_scale_rad=environment.integral_error_scale_rad,
        roll_acceleration_scale_rad_s2=environment.roll_acceleration_scale_rad_s2,
        force_limit_n=environment.force_limit_n,
    )


def _assert_pid_contract_matches(
    source: dict[str, object], target: dict[str, object]
) -> dict[str, object]:
    differences = {
        name: {"source": source.get(name), "target": target.get(name)}
        for name in PID_CONTRACT_FIELDS
        if source.get(name) != target.get(name)
    }
    if differences:
        raise ValueError(f"PID tuning contracts differ: {differences}")
    return {name: target[name] for name in PID_CONTRACT_FIELDS}


def _suite_fingerprint(profiles: tuple[RollRateCommandProfile, ...]) -> str:
    payload = json.dumps(
        [asdict(profile) for profile in profiles],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metrics(
    trace: dict[str, np.ndarray], force_limit_n: float, duration_s: float
) -> dict[str, float]:
    metrics = tracking_metrics(trace, force_limit_n)
    requested = np.asarray(trace["requested_f_as_n"], dtype=float)
    metrics.update(
        {
            "requested_force_total_variation_rate_n_s": float(
                metrics["requested_force_total_variation_n"] / duration_s
            ),
            "maximum_abs_requested_force_n": float(np.max(np.abs(requested))),
        }
    )
    return metrics


def _summary(rows: list[dict[str, object]], key: str) -> dict[str, float]:
    metrics = [row[key] for row in rows]
    costs = np.asarray([item["episode_cost"] for item in metrics], dtype=float)
    rmse = np.asarray([item["tracking_rmse_deg_s"] for item in metrics], dtype=float)
    tv_rate = np.asarray(
        [item["requested_force_total_variation_rate_n_s"] for item in metrics],
        dtype=float,
    )
    return {
        "pairs": len(metrics),
        "mean_episode_cost": float(np.mean(costs)),
        "median_episode_cost": float(np.median(costs)),
        "mean_tracking_rmse_deg_s": float(np.mean(rmse)),
        "median_tracking_rmse_deg_s": float(np.median(rmse)),
        "maximum_peak_error_deg_s": float(
            np.max([item["tracking_peak_error_deg_s"] for item in metrics])
        ),
        "mean_requested_force_total_variation_rate_n_s": float(np.mean(tv_rate)),
        "median_requested_force_total_variation_rate_n_s": float(np.median(tv_rate)),
        "maximum_abs_requested_force_n": float(
            np.max([item["maximum_abs_requested_force_n"] for item in metrics])
        ),
        "mean_force_rate_limit_active_fraction": float(
            np.mean([item["force_rate_limit_active_fraction"] for item in metrics])
        ),
        "mean_force_saturation_fraction": float(
            np.mean([item["force_saturation_fraction"] for item in metrics])
        ),
    }


def _save_plot(
    traces: list[tuple[str, dict[str, dict[str, np.ndarray]]]],
    path: Path,
    *,
    source_id: str,
    target_id: str,
    title_suffix: str,
) -> None:
    labels = {
        "target_local": f"{target_id} local PID",
        "source_transfer": f"{source_id} PID on {target_id}",
    }
    figure, axes = plt.subplots(
        len(traces),
        3,
        figsize=(17, 2.8 * len(traces)),
        squeeze=False,
        layout="constrained",
    )
    for row_index, (command_id, controllers) in enumerate(traces):
        response_axis, error_axis, force_axis = axes[row_index]
        reference_trace = controllers["target_local"]
        time_s = np.asarray(reference_trace["time_s"], dtype=float)
        reference_deg_s = np.rad2deg(reference_trace["p_reference_rad_s"])
        response_axis.plot(
            time_s,
            np.rad2deg(reference_trace["p_command_rad_s"]),
            color="#777777",
            linestyle=":",
            linewidth=1.0,
            label="p_c",
        )
        response_axis.plot(
            time_s,
            reference_deg_s,
            color="#111111",
            linestyle="--",
            linewidth=1.6,
            label="p_ref",
        )
        for key, trace in controllers.items():
            label = labels[key]
            response_deg_s = np.rad2deg(trace["p_rad_s"])
            response_axis.plot(
                trace["time_s"],
                response_deg_s,
                color=COLORS[key],
                linewidth=1.35,
                label=label,
            )
            error_axis.plot(
                trace["time_s"],
                response_deg_s - reference_deg_s,
                color=COLORS[key],
                linewidth=1.25,
                label=label,
            )
            force_axis.plot(
                trace["time_s"],
                trace["requested_f_as_n"],
                color=COLORS[key],
                linewidth=1.15,
                label=label,
            )
        response_axis.set_title(command_id)
        response_axis.set_ylabel("p (deg/s)")
        error_axis.set_ylabel("p - p_ref (deg/s)")
        force_axis.set_ylabel("Requested F_as (N)")
        error_axis.axhline(0.0, color="#777777", linewidth=0.7)
        force_axis.axhline(0.0, color="#777777", linewidth=0.7)
        for axis in (response_axis, error_axis, force_axis):
            axis.grid(alpha=0.22)
            axis.spines[["top", "right"]].set_visible(False)
        if row_index == 0:
            response_axis.legend(loc="best", ncol=2, fontsize=8)
            error_axis.legend(loc="best", fontsize=8)
            force_axis.legend(loc="best", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(f"Frozen PID transfer: {source_id} -> {target_id} ({title_suffix})")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    source_report = _load_report(args.source_pid_report)
    target_report = _load_report(args.target_pid_report)
    source_id = str(source_report["plant_id"])
    target_id = str(target_report["plant_id"])
    if source_id == target_id:
        raise ValueError("source and target aircraft must differ")
    pid_contract = _assert_pid_contract_matches(source_report, target_report)

    target_record = load_persisted_records(args.library, [target_id])[0]
    if target_report.get("plant_parameters") != asdict(target_record.parameters):
        raise ValueError("target PID report does not match the target plant record")
    source_gains = PIDGains(**source_report["gains"])
    target_gains = PIDGains(**target_report["gains"])
    config = SpecialistTrainingConfig(
        episode_duration_s=args.duration,
        plant_dt_s=float(target_report["plant_dt_s"]),
        policy_dt_s=float(target_report["policy_dt_s"]),
        reference_natural_frequency_rad_s=float(
            target_report["reference_natural_frequency_rad_s"]
        ),
        reference_damping_ratio=float(target_report["reference_damping_ratio"]),
        reference_delay_mode=str(target_report["reference_delay_mode"]),
        seed=args.evaluation_seed,
        device="cpu",
    )
    independent_test = args.command_suite == "independent-test"
    profiles = (
        specialist_independent_test_commands(args.duration)
        if independent_test
        else specialist_evaluation_commands(args.duration)
    )

    traces: list[tuple[str, dict[str, dict[str, np.ndarray]]]] = []
    rows: list[dict[str, object]] = []
    for index, profile in enumerate(profiles):
        seed = args.evaluation_seed + index
        local_environment = build_specialist_env(target_record, config, (profile,))
        transfer_environment = build_specialist_env(target_record, config, (profile,))
        controller_traces = {
            "target_local": rollout_policy(
                _pid_policy(target_gains, local_environment),
                local_environment,
                seed=seed,
            ),
            "source_transfer": rollout_policy(
                _pid_policy(source_gains, transfer_environment),
                transfer_environment,
                seed=seed,
            ),
        }
        for signal_name in ("time_s", "p_command_rad_s", "p_reference_rad_s"):
            if not np.array_equal(
                controller_traces["target_local"][signal_name],
                controller_traces["source_transfer"][signal_name],
            ):
                raise RuntimeError(f"PID runs received different {signal_name}")
        traces.append((profile.command_id, controller_traces))
        rows.append(
            {
                "command_id": profile.command_id,
                "command_kind": profile.kind,
                "target_local": _metrics(
                    controller_traces["target_local"],
                    config.force_limit_n,
                    profile.duration_s,
                ),
                "source_transfer": _metrics(
                    controller_traces["source_transfer"],
                    config.force_limit_n,
                    profile.duration_s,
                ),
            }
        )

    summaries = {
        "target_local": _summary(rows, "target_local"),
        "source_transfer": _summary(rows, "source_transfer"),
    }
    local_summary = summaries["target_local"]
    transfer_summary = summaries["source_transfer"]
    degradation = {
        "mean_rmse_ratio_transfer_over_local": float(
            transfer_summary["mean_tracking_rmse_deg_s"]
            / local_summary["mean_tracking_rmse_deg_s"]
        ),
        "mean_episode_cost_ratio_transfer_over_local": float(
            transfer_summary["mean_episode_cost"] / local_summary["mean_episode_cost"]
        ),
        "mean_tv_rate_ratio_transfer_over_local": float(
            transfer_summary["mean_requested_force_total_variation_rate_n_s"]
            / local_summary["mean_requested_force_total_variation_rate_n_s"]
        ),
        "commands_where_transfer_rmse_is_better": int(
            sum(
                row["source_transfer"]["tracking_rmse_deg_s"]
                < row["target_local"]["tracking_rmse_deg_s"]
                for row in rows
            )
        ),
    }

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    full_plot = destination / "cross_aircraft_pid_time_domain.png"
    step_plot = destination / "cross_aircraft_pid_step_focus.png"
    suite_title = "frozen independent test" if independent_test else "validation"
    _save_plot(
        traces,
        full_plot,
        source_id=source_id,
        target_id=target_id,
        title_suffix=suite_title,
    )
    _save_plot(
        traces[:3],
        step_plot,
        source_id=source_id,
        target_id=target_id,
        title_suffix=f"{suite_title}, held steps",
    )
    report = {
        "schema_version": "cross_aircraft_frozen_pid_v1",
        "status": "complete",
        "source": git_source_revision(),
        "experiment": {
            "source_aircraft": source_id,
            "target_aircraft": target_id,
            "no_target_retuning": True,
            "comparison_scope": "controller transfer on one fixed target aircraft",
        },
        "fairness": {
            "same_target_plant": True,
            "same_command_samples": True,
            "same_reference_samples": True,
            "same_initial_condition_seed": True,
            "same_force_and_rate_limits": True,
            "pid_tuning_contract": pid_contract,
            "evaluation_duration_s": args.duration,
            "evaluation_seed": args.evaluation_seed,
        },
        "target_plant": {
            "plant_id": target_record.plant_id,
            "parameters": asdict(target_record.parameters),
        },
        "command_suite": {
            "name": args.command_suite,
            "version": (
                SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION
                if independent_test
                else "specialist-validation-v1"
            ),
            "sha256": _suite_fingerprint(profiles),
            "profiles": [asdict(profile) for profile in profiles],
        },
        "controllers": {
            "target_local": {
                "plant_id": target_id,
                "gains": asdict(target_gains),
                "report_path": str(args.target_pid_report.resolve()),
                "report_sha256": sha256_file(args.target_pid_report),
            },
            "source_transfer": {
                "plant_id": source_id,
                "gains": asdict(source_gains),
                "report_path": str(args.source_pid_report.resolve()),
                "report_sha256": sha256_file(args.source_pid_report),
                "retuned_on_target": False,
            },
        },
        "summary": summaries,
        "transfer_degradation": degradation,
        "rows": rows,
        "artifacts": {
            "time_domain": str(full_plot),
            "step_focus": str(step_plot),
        },
    }
    report_path = destination / "comparison.json"
    _write_json(report_path, report)
    print(
        json.dumps(
            {"summary": summaries, "transfer_degradation": degradation},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
