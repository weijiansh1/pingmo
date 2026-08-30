import numpy as np

from src.aircraft.sampler import generate_plant_library
from src.envs.commands import CommandProfile
from src.experiments.policy_evaluation import evaluate_policy_pairs


class ZeroPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(1, dtype=np.float32), None


class BatchZeroPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        self.calls += 1
        return np.zeros((observation.shape[0], 1), dtype=np.float32)


def test_pair_evaluation_reports_response_and_constraint_metrics() -> None:
    record = generate_plant_library(31, {"validation": 1})[0]
    profile = CommandProfile(
        "step-1.00",
        "step",
        amplitude=1.0,
        duration_s=0.01,
        onset_s=0.0,
    )

    rows = evaluate_policy_pairs(ZeroPolicy(), [record], [profile], seed=31)

    assert len(rows) == 1
    row = rows[0]
    assert row["plant_id"] == record.plant_id
    assert row["split"] == "validation"
    assert row["command_id"] == "step-1.00"
    assert row["policy_steps"] == 10
    assert row["plant_samples"] == 10
    assert row["controlled_action_saturation_fraction"] == 0.0
    assert row["controlled_action_total_variation_n"] == 0.0
    assert row["raw_roll_peak_abs_rad_s"] >= 0.0
    assert row["controlled_roll_peak_abs_rad_s"] >= 0.0


def test_pair_evaluation_batches_policy_inference_across_pairs() -> None:
    records = generate_plant_library(37, {"validation": 2})
    profiles = (
        CommandProfile("step-pos", "step", amplitude=0.5, duration_s=0.003),
        CommandProfile("step-neg", "step", amplitude=-0.5, duration_s=0.003),
    )
    policy = BatchZeroPolicy()

    rows = evaluate_policy_pairs(policy, records, profiles, inference_batch_size=4, seed=37)

    assert len(rows) == 4
    assert policy.calls == 3
