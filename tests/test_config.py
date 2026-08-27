from pathlib import Path

import pytest

from src.utils.config import load_yaml
from src.gjb.profile import load_roll_profile


def test_load_yaml_returns_mapping_and_rejects_sequence(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("dt: 0.005\nname: smoke\n", encoding="utf-8")
    assert load_yaml(mapping) == {"dt": 0.005, "name": "smoke"}

    sequence = tmp_path / "sequence.yaml"
    sequence.write_text("- invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(sequence)


def test_iv_a_profile_exposes_force_scale_and_rejects_unknown_profile() -> None:
    root = Path(__file__).parents[1]
    profile = load_roll_profile(root / "data/gjb_roll_spec.yaml", "IV-A")

    assert profile["pilot_force_scale_n"] == 22.0
    assert profile["action_authority_ratios"] == [0.1, 0.2, 0.3, 0.5]
    assert profile["model_version"] == "GJB_s1_corrected"
    with pytest.raises(ValueError, match="unknown roll profile"):
        load_roll_profile(root / "data/gjb_roll_spec.yaml", "unknown")
