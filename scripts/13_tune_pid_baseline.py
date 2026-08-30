"""Tune and evaluate a direct P-channel PID under the specialist contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.pid import (  # noqa: E402
    PIDGains,
    PIDTuningConfig,
    RollRatePIDPolicy,
    tune_pid_gains,
)
from src.experiments.exploratory_sac import load_persisted_records  # noqa: E402
from src.teacher.specialist.manager import select_specialist_records  # noqa: E402
from src.teacher.specialist.trainer import (  # noqa: E402
    SpecialistTrainingConfig,
    build_specialist_env,
    evaluate_specialist,
    rollout_policy,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library",
        type=Path,
        default=(
            ROOT
            / "data"
            / "aircraft"
            / "generated"
            / "p_channel_library_iv_a_manual_v1"
            / "plants.jsonl"
        ),
    )
    parser.add_argument("--plant-id")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "specialist_pid_gate"
    )
    parser.add_argument("--episode-duration", type=float, default=5.0)
    parser.add_argument("--plant-dt", type=float, default=0.001)
    parser.add_argument("--policy-dt", type=float, default=0.020)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260828)
    return parser.parse_args()


def _pid_policy(
    gains: PIDGains,
    environment,
) -> RollRatePIDPolicy:
    return RollRatePIDPolicy(
        gains,
        policy_dt_s=environment.policy_dt_s,
        command_scale_rad_s=environment.command_scale_rad_s,
        integral_error_scale_rad=environment.integral_error_scale_rad,
        roll_acceleration_scale_rad_s2=environment.roll_acceleration_scale_rad_s2,
        force_limit_n=environment.force_limit_n,
    )


def main() -> None:
    args = _parse_args()
    if args.plant_id:
        record = load_persisted_records(args.library, [args.plant_id])[0]
    else:
        record = select_specialist_records(
            args.library, 1, seed=args.seed
        )[0]
    config = SpecialistTrainingConfig(
        episode_duration_s=args.episode_duration,
        plant_dt_s=args.plant_dt,
        policy_dt_s=args.policy_dt,
        seed=args.seed,
        device="cpu",
    )
    tuning_profiles = tuple(
        profile
        for profile in build_specialist_env(record, config).command_profiles
        if profile.amplitude_deg_s > 0
    )
    template_environment = build_specialist_env(
        record, config, (tuning_profiles[0],)
    )

    def objective(gains: PIDGains) -> float:
        costs = []
        for index, profile in enumerate(tuning_profiles):
            environment = build_specialist_env(record, config, (profile,))
            trace = rollout_policy(
                _pid_policy(gains, environment),
                environment,
                seed=args.seed + index,
            )
            error = (
                trace["p_rad_s"] - trace["p_reference_rad_s"]
            ) / environment.command_scale_rad_s
            tail = error[int(0.8 * len(error)) :]
            costs.append(
                -float(np.sum(trace["reward"]))
                + 4.0 * float(np.mean(np.square(tail)))
            )
        return float(np.mean(costs))

    started = time.perf_counter()
    gains, tuning_cost = tune_pid_gains(
        objective,
        PIDTuningConfig(
            max_iterations=args.max_iterations,
            population_size=args.population_size,
            seed=args.seed,
        ),
    )
    tuning_elapsed_s = time.perf_counter() - started
    evaluation = evaluate_specialist(
        _pid_policy(gains, template_environment),
        record,
        config,
        output_dir=args.output,
        controller_label="PID",
    )
    report = {
        "status": "complete",
        "controller": "direct_reference_tracking_pid",
        "plant_id": record.plant_id,
        "plant_parameters": asdict(record.parameters),
        "gains": asdict(gains),
        "tuning_cost": tuning_cost,
        "tuning_elapsed_s": tuning_elapsed_s,
        "tuning_command_ids": [profile.command_id for profile in tuning_profiles],
        "episode_duration_s": config.episode_duration_s,
        "plant_dt_s": config.plant_dt_s,
        "policy_dt_s": config.policy_dt_s,
        "reference_natural_frequency_rad_s": (
            config.reference_natural_frequency_rad_s
        ),
        "reference_damping_ratio": config.reference_damping_ratio,
        "reference_delay_mode": config.reference_delay_mode,
        "evaluation": evaluation,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "pid_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report={report_path}")
    print(f"plot={args.output / 'response_comparison.png'}")


if __name__ == "__main__":
    main()
