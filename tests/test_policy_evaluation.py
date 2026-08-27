import numpy as np

from src.aircraft.sampler import generate_plant_library
from src.envs.commands import CommandProfile
from src.experiments.policy_evaluation import evaluate_policy_pairs


class ZeroPolicy:
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        return np.zeros(1, dtype=np.float32), None


def test_pair_evaluation_reports_tracking_and_constraint_metrics() -> None:
    record = generate_plant_library(31, {"validation": 1})[0]
    profile = CommandProfile("step-1.00", "step", amplitude=1.0)

    rows = evaluate_policy_pairs(ZeroPolicy(), [record], [profile], horizon_steps=4, seed=31)

    assert len(rows) == 1
    assert rows[0]["plant_id"] == record.plant_id
    assert rows[0]["split"] == "validation"
    assert rows[0]["command_id"] == "step-1.00"
    assert rows[0]["steps"] == 4
    assert rows[0]["tracking_rmse"] >= 0.0
    assert rows[0]["saturation_fraction"] == 0.0
    assert rows[0]["command_total_variation_n"] == 0.0
