"""Compare the original 0803 Teacher, refined Teacher, and tuned PID."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
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
from src.teacher.specialist.trainer import (  # noqa: E402
    SpecialistTrainingConfig,
    build_specialist_env,
    load_specialist_actor,
    rollout_policy,
    tracking_metrics,
)
from src.utils.provenance import git_source_revision, sha256_file  # noqa: E402


CONTROLLER_COLORS = {
    "Tuned PID": "#2468a2",
    "Original TD3": "#d47a1f",
    "Refined TD3": "#b3262e",
}
CONTROLLER_LABELS = ("Tuned PID", "Original TD3", "Refined TD3")
CURVE_METRICS = (
    ("mean_episode_cost", "Mean episode cost"),
    ("mean_tracking_rmse_deg_s", "Tracking RMSE (deg/s)"),
    ("maximum_peak_error_deg_s", "Peak error (deg/s)"),
    ("mean_requested_force_total_variation_n", "30 s requested-force TV (N)"),
)
RUNTIME_CONTRACT_FIELDS = (
    "episode_duration_s",
    "plant_dt_s",
    "policy_dt_s",
    "command_scale_deg_s",
    "force_limit_n",
    "force_rate_limit_n_s",
    "actuator_time_constant_s",
    "reference_natural_frequency_rad_s",
    "reference_damping_ratio",
    "reference_delay_mode",
    "tracking_error_weight",
    "force_energy_weight",
    "force_delta_weight",
    "reward_scale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--refined-checkpoint", type=Path, required=True)
    parser.add_argument("--pid-report", type=Path, required=True)
    parser.add_argument("--original-report", type=Path, required=True)
    parser.add_argument("--refined-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--command-suite",
        choices=("independent-test", "validation"),
        default="independent-test",
    )
    parser.add_argument("--evaluation-seed", type=int, default=20260901)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
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


def _runtime_contract(config: SpecialistTrainingConfig) -> dict[str, object]:
    return {name: getattr(config, name) for name in RUNTIME_CONTRACT_FIELDS}


def _assert_same_runtime_contract(
    original: SpecialistTrainingConfig,
    refined: SpecialistTrainingConfig,
) -> dict[str, object]:
    original_contract = _runtime_contract(original)
    refined_contract = _runtime_contract(refined)
    differences = {
        name: {
            "original": original_contract[name],
            "refined": refined_contract[name],
        }
        for name in RUNTIME_CONTRACT_FIELDS
        if original_contract[name] != refined_contract[name]
    }
    if differences:
        raise ValueError(
            "TD3 checkpoints have different evaluation runtime contracts: "
            f"{differences}"
        )
    return refined_contract


def _checkpoint_step(path: Path, payload: dict[str, object]) -> int:
    for key in ("step", "steps", "completed_steps"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    match = re.search(r"actor_step_(\d+)\.pt$", path.name)
    if match is not None:
        return int(match.group(1))
    raise ValueError(f"cannot determine checkpoint step from {path}")


def _suite_fingerprint(profiles: tuple[RollRateCommandProfile, ...]) -> str:
    serialized = json.dumps(
        [asdict(profile) for profile in profiles],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _comparison_metrics(
    trace: dict[str, np.ndarray],
    force_limit_n: float,
    duration_s: float,
) -> dict[str, float]:
    metrics = tracking_metrics(trace, force_limit_n)
    requested = np.asarray(trace["requested_f_as_n"], dtype=float)
    commanded = np.asarray(trace["commanded_f_as_n"], dtype=float)
    requested_increments = np.diff(requested)
    metrics.update(
        {
            "evaluation_duration_s": float(duration_s),
            "requested_force_total_variation_rate_n_s": float(
                metrics["requested_force_total_variation_n"] / duration_s
            ),
            "force_total_variation_rate_n_s": float(
                metrics["force_total_variation_n"] / duration_s
            ),
            "requested_force_mean_absolute_increment_n": float(
                np.mean(np.abs(requested_increments))
            ),
            "requested_force_rms_increment_n": float(
                np.sqrt(np.mean(np.square(requested_increments)))
            ),
            "maximum_abs_requested_force_n": float(np.max(np.abs(requested))),
            "maximum_abs_commanded_force_n": float(np.max(np.abs(commanded))),
        }
    )
    return metrics


def _summary(rows: list[dict[str, object]], label: str) -> dict[str, float]:
    metrics = [row[label] for row in rows]
    episode_cost = np.asarray([item["episode_cost"] for item in metrics], dtype=float)
    tracking_rmse = np.asarray(
        [item["tracking_rmse_deg_s"] for item in metrics], dtype=float
    )
    requested_variation = np.asarray(
        [item["requested_force_total_variation_n"] for item in metrics], dtype=float
    )
    requested_variation_rate = np.asarray(
        [item["requested_force_total_variation_rate_n_s"] for item in metrics],
        dtype=float,
    )
    return {
        "pairs": len(metrics),
        "mean_episode_cost": float(np.mean(episode_cost)),
        "median_episode_cost": float(np.median(episode_cost)),
        "mean_tracking_rmse_deg_s": float(np.mean(tracking_rmse)),
        "median_tracking_rmse_deg_s": float(np.median(tracking_rmse)),
        "maximum_peak_error_deg_s": float(
            np.max([item["tracking_peak_error_deg_s"] for item in metrics])
        ),
        "mean_requested_force_total_variation_n": float(np.mean(requested_variation)),
        "median_requested_force_total_variation_n": float(
            np.median(requested_variation)
        ),
        "maximum_requested_force_total_variation_n": float(np.max(requested_variation)),
        "mean_requested_force_total_variation_rate_n_s": float(
            np.mean(requested_variation_rate)
        ),
        "median_requested_force_total_variation_rate_n_s": float(
            np.median(requested_variation_rate)
        ),
        "mean_requested_force_rms_increment_n": float(
            np.mean([item["requested_force_rms_increment_n"] for item in metrics])
        ),
        "maximum_abs_requested_force_n": float(
            np.max([item["maximum_abs_requested_force_n"] for item in metrics])
        ),
        "mean_force_saturation_fraction": float(
            np.mean([item["force_saturation_fraction"] for item in metrics])
        ),
    }


def _save_time_domain_plot(
    traces: list[tuple[str, dict[str, dict[str, np.ndarray]]]],
    path: Path,
    *,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        len(traces),
        3,
        figsize=(17, 2.8 * len(traces)),
        squeeze=False,
        layout="constrained",
    )
    for row_index, (command_id, controllers) in enumerate(traces):
        response_axis, error_axis, force_axis = axes[row_index]
        reference_trace = controllers["Tuned PID"]
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
        for label, trace in controllers.items():
            controller_time = np.asarray(trace["time_s"], dtype=float)
            response_deg_s = np.rad2deg(trace["p_rad_s"])
            color = CONTROLLER_COLORS[label]
            response_axis.plot(
                controller_time,
                response_deg_s,
                color=color,
                linewidth=1.35,
                label=label,
            )
            error_axis.plot(
                controller_time,
                response_deg_s - reference_deg_s,
                color=color,
                linewidth=1.25,
                label=label,
            )
            force_axis.plot(
                controller_time,
                trace["requested_f_as_n"],
                color=color,
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
            response_axis.legend(loc="best", ncol=3, fontsize=8)
            error_axis.legend(loc="best", fontsize=8)
            force_axis.legend(loc="best", fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("Time (s)")
    figure.suptitle(title)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _pid_validation_summary(
    gains: PIDGains,
    record: object,
    config: SpecialistTrainingConfig,
    *,
    seed: int,
) -> dict[str, float]:
    rows: list[dict[str, object]] = []
    for index, profile in enumerate(
        specialist_evaluation_commands(config.episode_duration_s)
    ):
        environment = build_specialist_env(record, config, (profile,))
        trace = rollout_policy(
            _pid_policy(gains, environment),
            environment,
            seed=seed + index,
        )
        rows.append(
            {
                "Tuned PID": _comparison_metrics(
                    trace,
                    config.force_limit_n,
                    profile.duration_s,
                )
            }
        )
    return _summary(rows, "Tuned PID")


def _save_convergence_plot(
    original_report: dict[str, object],
    refined_report: dict[str, object],
    pid_summary: dict[str, float],
    path: Path,
) -> None:
    original_points = original_report["learning_curve"]
    refined_points = refined_report["learning_curve"]
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.4), layout="constrained")
    for axis, (metric, title) in zip(axes, CURVE_METRICS, strict=True):
        for label, points in (
            ("Original TD3", original_points),
            ("Refined TD3", refined_points),
        ):
            steps = np.asarray([point["step"] for point in points], dtype=float)
            values = np.asarray([point[metric] for point in points], dtype=float)
            axis.plot(
                steps,
                values,
                color=CONTROLLER_COLORS[label],
                marker="o",
                markersize=3.5,
                linewidth=1.7,
                label=label,
            )
        axis.axhline(
            float(pid_summary[metric]),
            color=CONTROLLER_COLORS["Tuned PID"],
            linestyle="--",
            linewidth=1.4,
            label="Tuned PID",
        )
        axis.set_title(title)
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="best")
    figure.suptitle("Aircraft 0803: fixed 30 s validation during training")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    original, original_record, original_config, original_payload = (
        load_specialist_actor(args.original_checkpoint, device=args.device)
    )
    refined, refined_record, refined_config, refined_payload = load_specialist_actor(
        args.refined_checkpoint, device=args.device
    )
    if original_record != refined_record:
        raise ValueError("original and refined checkpoints refer to different aircraft")
    runtime_contract = _assert_same_runtime_contract(original_config, refined_config)
    original_step = _checkpoint_step(args.original_checkpoint, original_payload)
    refined_step = _checkpoint_step(args.refined_checkpoint, refined_payload)
    if original_step != refined_step:
        raise ValueError(
            "fair TD3 comparison requires checkpoints from the same environment step; "
            f"got original={original_step}, refined={refined_step}"
        )

    pid_payload = _load_json(args.pid_report)
    if pid_payload["plant_id"] != refined_record.plant_id:
        raise ValueError("PID report and checkpoints refer to different aircraft")
    if pid_payload.get("plant_parameters") != asdict(refined_record.parameters):
        raise ValueError(
            "PID report and checkpoints contain different plant parameters"
        )
    gains = PIDGains(**pid_payload["gains"])
    traces: list[tuple[str, dict[str, dict[str, np.ndarray]]]] = []
    rows: list[dict[str, object]] = []
    independent_test = args.command_suite == "independent-test"
    profiles = (
        specialist_independent_test_commands(refined_config.episode_duration_s)
        if independent_test
        else specialist_evaluation_commands(refined_config.episode_duration_s)
    )
    for index, profile in enumerate(profiles):
        seed = args.evaluation_seed + index
        pid_env = build_specialist_env(refined_record, refined_config, (profile,))
        original_env = build_specialist_env(
            original_record, original_config, (profile,)
        )
        refined_env = build_specialist_env(refined_record, refined_config, (profile,))
        controller_traces = {
            "Tuned PID": rollout_policy(
                _pid_policy(gains, pid_env), pid_env, seed=seed
            ),
            "Original TD3": rollout_policy(original, original_env, seed=seed),
            "Refined TD3": rollout_policy(refined, refined_env, seed=seed),
        }
        for signal_name in ("time_s", "p_command_rad_s", "p_reference_rad_s"):
            reference_values = controller_traces["Tuned PID"][signal_name]
            if any(
                not np.array_equal(reference_values, trace[signal_name])
                for trace in controller_traces.values()
            ):
                raise RuntimeError(
                    f"controllers did not receive an identical {signal_name} trace"
                )
        traces.append((profile.command_id, controller_traces))
        rows.append(
            {
                "command_id": profile.command_id,
                "command_kind": profile.kind,
                "duration_s": profile.duration_s,
                **{
                    label: _comparison_metrics(
                        trace,
                        refined_config.force_limit_n,
                        profile.duration_s,
                    )
                    for label, trace in controller_traces.items()
                },
            }
        )

    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    summaries = {label: _summary(rows, label) for label in CONTROLLER_LABELS}
    suite_slug = args.command_suite.replace("-", "_")
    time_domain_path = destination / f"0803_{suite_slug}_time_domain.png"
    step_focus_path = destination / f"0803_{suite_slug}_step_focus.png"
    convergence_path = destination / "0803_validation_convergence.png"
    suite_title = "frozen independent test" if independent_test else "validation"
    _save_time_domain_plot(
        traces,
        time_domain_path,
        title=f"Aircraft 0803: PID vs TD3 on {suite_title} commands",
    )
    _save_time_domain_plot(
        traces[:3],
        step_focus_path,
        title=f"Aircraft 0803: {suite_title} held-step response focus",
    )
    pid_validation_summary = _pid_validation_summary(
        gains,
        refined_record,
        refined_config,
        seed=args.evaluation_seed,
    )
    _save_convergence_plot(
        _load_json(args.original_report),
        _load_json(args.refined_report),
        pid_validation_summary,
        convergence_path,
    )
    original_observation_contract = original_payload.get(
        "actor_observation_contract",
        build_specialist_env(
            original_record, original_config, profiles
        ).actor_observation_contract(),
    )
    refined_observation_contract = refined_payload.get(
        "actor_observation_contract",
        build_specialist_env(
            refined_record, refined_config, profiles
        ).actor_observation_contract(),
    )
    original_training_report = _load_json(args.original_report)
    refined_training_report = _load_json(args.refined_report)
    original_td3_config = original_training_report.get("td3_config", {})
    refined_td3_config = refined_training_report.get("td3_config", {})
    all_training_keys = sorted(set(original_td3_config) | set(refined_td3_config))
    training_differences = {
        key: {
            "original": original_td3_config.get(key),
            "refined": refined_td3_config.get(key),
        }
        for key in all_training_keys
        if original_td3_config.get(key) != refined_td3_config.get(key)
    }
    report = {
        "schema_version": "pure_reward_td3_0803_fair_comparison_v2",
        "status": "complete",
        "source": git_source_revision(),
        "plant_id": refined_record.plant_id,
        "plant_parameters": asdict(refined_record.parameters),
        "pid_gains": asdict(gains),
        "command_suite": {
            "name": args.command_suite,
            "version": (
                SPECIALIST_INDEPENDENT_TEST_SUITE_VERSION
                if independent_test
                else "specialist-validation-v1"
            ),
            "sha256": _suite_fingerprint(profiles),
            "independent_of_training": independent_test,
            "independent_of_checkpoint_selection": independent_test,
            "profiles": [asdict(profile) for profile in profiles],
        },
        "fairness": {
            "claim_scope": (
                "paired deterministic deployment performance; not training efficiency"
            ),
            "same_plant_and_parameters": True,
            "same_runtime_contract": True,
            "runtime_contract": runtime_contract,
            "same_command_samples": True,
            "same_reference_samples": True,
            "same_initial_condition_seed": True,
            "evaluation_seed": args.evaluation_seed,
            "same_td3_checkpoint_environment_steps": True,
            "td3_checkpoint_environment_steps": original_step,
            "same_td3_training_seed": original_config.seed == refined_config.seed,
            "test_commands_used_for_training": False if independent_test else None,
            "test_commands_used_for_checkpoint_selection": (
                False if independent_test else True
            ),
            "td3_actor_observation_contracts_identical": (
                original_observation_contract == refined_observation_contract
            ),
            "original_actor_observation_contract": original_observation_contract,
            "refined_actor_observation_contract": refined_observation_contract,
            "td3_training_configuration_differences": training_differences,
            "pid_training_budget_matched_to_td3": False,
            "training_efficiency_claim_supported": False,
            "single_change_ablation_claim_supported": False,
            "multi_seed_statistical_claim_supported": False,
        },
        "summary": summaries,
        "pid_validation_summary_for_convergence_plot": pid_validation_summary,
        "rows": rows,
        "checkpoints": {
            "original": {
                "path": str(args.original_checkpoint.resolve()),
                "sha256": sha256_file(args.original_checkpoint),
                "environment_step": original_step,
            },
            "refined": {
                "path": str(args.refined_checkpoint.resolve()),
                "sha256": sha256_file(args.refined_checkpoint),
                "environment_step": refined_step,
            },
        },
        "artifacts": {
            "time_domain_comparison": str(time_domain_path),
            "step_response_focus": str(step_focus_path),
            "training_convergence": str(convergence_path),
        },
    }
    report_path = destination / "comparison.json"
    _write_json(report_path, report)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
