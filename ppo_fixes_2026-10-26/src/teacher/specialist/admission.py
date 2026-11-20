"""Single contract for pure-reward Teacher Bank admission.

Two different evaluations exist and must not be conflated:

* ``trainer_final_actor_v1`` -- the training-time gate computed on the **final
  actor** (total requested-force TV in N, thresholds 1.0/3.0/50 N).  It is
  written into ``report.json`` and is NOT the bank admission decision.
* ``bank_selected_checkpoint_v1`` -- the archived bank admission: for each
  validation point, the gate is evaluated on that point (TV converted to N/s
  by dividing by the episode duration), then the cheapest quality-eligible
  checkpoint is exported.  If no checkpoint qualifies, the fallback is still
  admitted as failed with a different selection method.

``select_bank_checkpoint`` is the one place that decides the bank step and its
gate; ``scripts/47_train_pure_reward_teacher_bank.py`` and any future caller
must use it instead of re-implementing the thresholds.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

BANK_GATE_NAME = "bank_selected_checkpoint_v1"
BANK_GATE_SCOPE = "selected_checkpoint_validation"
FINAL_GATE_NAME = "trainer_final_actor_v1"
FINAL_GATE_SCOPE = "final_actor_evaluation"

ELIGIBLE_SELECTION = "minimum_cost_among_quality_eligible_checkpoints"
FALLBACK_SELECTION = "minimum_cost_fallback_no_quality_eligible_checkpoint"


@dataclass(frozen=True)
class BankAdmissionThresholds:
    minimum_tracking_improvement_rate: float = 1.0
    maximum_mean_tracking_rmse_deg_s: float = 1.5
    maximum_peak_error_deg_s: float = 5.0
    maximum_mean_requested_force_total_variation_rate_n_s: float = 12.0
    maximum_saturation_fraction: float = 0.01

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def evaluate_bank_admission(
    validation: Mapping[str, Any],
    episode_duration_s: float,
    thresholds: BankAdmissionThresholds = BankAdmissionThresholds(),
) -> dict[str, Any]:
    """Evaluate one checkpoint's fixed-validation metrics under the bank gate.

    ``validation`` is a validation point (``learning_curve`` entry or
    ``best_validation``) with the same keys the trainer writes.  TV is a total
    over the episode, so the rate is ``tv_total / episode_duration_s``; the
    duration is required and must be positive because a silent 1 s default
    would make the gate 30x too strict.
    """

    duration = float(episode_duration_s)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("episode_duration_s must be a positive finite number")

    tv_total = validation["mean_requested_force_total_variation_n"]
    observed = {
        "tracking_improvement_rate": validation["tracking_improvement_rate"],
        "mean_tracking_rmse_deg_s": validation["mean_tracking_rmse_deg_s"],
        "maximum_peak_error_deg_s": validation["maximum_peak_error_deg_s"],
        "mean_requested_force_total_variation_n": tv_total,
        "mean_requested_force_total_variation_rate_n_s": float(tv_total) / duration,
        "mean_force_saturation_fraction": validation["mean_force_saturation_fraction"],
        "step": int(validation["step"]),
    }
    finite = all(_finite(v) for v in observed.values())
    checks = {
        "tracking_improvement_rate": (
            finite
            and float(observed["tracking_improvement_rate"])
            >= thresholds.minimum_tracking_improvement_rate
        ),
        "mean_tracking_rmse": (
            finite
            and float(observed["mean_tracking_rmse_deg_s"])
            <= thresholds.maximum_mean_tracking_rmse_deg_s
        ),
        "peak_tracking_error": (
            finite
            and float(observed["maximum_peak_error_deg_s"])
            <= thresholds.maximum_peak_error_deg_s
        ),
        "requested_force_tv_rate": (
            finite
            and float(observed["mean_requested_force_total_variation_rate_n_s"])
            <= thresholds.maximum_mean_requested_force_total_variation_rate_n_s
        ),
        "force_saturation": (
            finite
            and float(observed["mean_force_saturation_fraction"])
            <= thresholds.maximum_saturation_fraction
        ),
    }
    return {
        "gate": BANK_GATE_NAME,
        "scope": BANK_GATE_SCOPE,
        "checks": checks,
        "observed": observed,
        "thresholds": {
            "minimum_tracking_improvement_rate": thresholds.minimum_tracking_improvement_rate,
            "maximum_mean_tracking_rmse_deg_s": thresholds.maximum_mean_tracking_rmse_deg_s,
            "maximum_peak_error_deg_s": thresholds.maximum_peak_error_deg_s,
            "maximum_mean_requested_force_total_variation_rate_n_s": (
                thresholds.maximum_mean_requested_force_total_variation_rate_n_s
            ),
            "maximum_saturation_fraction": thresholds.maximum_saturation_fraction,
            "episode_duration_s": duration,
        },
        "episode_duration_s": duration,
        "nonfinite": not finite,
        "passed": bool(finite and all(checks.values())),
    }


_POINT_KEYS = (
    "step",
    "mean_tracking_rmse_deg_s",
    "maximum_peak_error_deg_s",
    "mean_requested_force_total_variation_n",
    "mean_force_saturation_fraction",
    "tracking_improvement_rate",
    "mean_episode_cost",
)


def _same_history_point(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    for key in _POINT_KEYS:
        if key == "step":
            if int(a[key]) != int(b[key]):
                return False
            continue
        try:
            if float(a[key]) != float(b[key]):
                return False
        except (TypeError, ValueError):
            return False
    return True


def select_bank_checkpoint(
    history: Sequence[Mapping[str, Any]],
    episode_duration_s: float,
    thresholds: BankAdmissionThresholds = BankAdmissionThresholds(),
    *,
    fallback: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Select the exported bank checkpoint and its gate.

    Eligible checkpoints are ranked by ``mean_episode_cost`` (the same rule as
    the archived bank); the cost must be finite and the step must be parseable.
    With no eligible checkpoint the fallback must be one of the **current**
    history points (same step and metrics); a fallback that is absent from the
    history, disagrees with it, or itself passes the gate is a contradiction and
    raises ``ValueError`` instead of being admitted.  The returned fallback gate
    is always failed.
    """

    if not history:
        raise ValueError("validation history is empty")
    candidates: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for point in history:
        # A non-finite cost cannot rank a candidate; a malformed step should
        # fail loudly rather than being silently skipped or admitted.
        int(point["step"])
        if not _finite(point.get("mean_episode_cost")):
            continue
        quality = evaluate_bank_admission(point, episode_duration_s, thresholds)
        if quality["passed"]:
            candidates.append((point, quality))
    if candidates:
        point, quality = min(
            candidates, key=lambda item: float(item[0]["mean_episode_cost"])
        )
        return point, quality, ELIGIBLE_SELECTION
    if fallback is None:
        raise ValueError("no quality-eligible checkpoint and no fallback")
    try:
        fallback_step = int(fallback["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("fallback has no parseable step") from exc
    matches = [
        point
        for point in history
        if int(point["step"]) == fallback_step and _same_history_point(point, fallback)
    ]
    if len(matches) != 1:
        raise ValueError(
            "fallback must exactly match one point in the current validation history"
        )
    point = matches[0]
    if not _finite(point.get("mean_episode_cost")):
        raise ValueError("fallback cost must be finite")
    quality = evaluate_bank_admission(point, episode_duration_s, thresholds)
    if quality["passed"]:
        raise ValueError(
            "fallback passes the admission gate but no eligible checkpoint was found; "
            "inconsistent inputs must not be admitted"
        )
    return point, quality, FALLBACK_SELECTION


def actor_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe_final_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Tag a trainer report gate so it cannot be read as bank admission."""

    described = dict(gate)
    described["gate"] = FINAL_GATE_NAME
    described["scope"] = FINAL_GATE_SCOPE
    described["bank_admission"] = False
    return described
