"""Read frozen roll-axis experiment profiles without inventing GJB values."""

from pathlib import Path
from typing import Any

from src.utils.config import load_yaml


def load_roll_profile(path: str | Path, profile_name: str) -> dict[str, Any]:
    document = load_yaml(path)
    profiles = document.get("profiles")
    if not isinstance(profiles, dict) or profile_name not in profiles:
        raise ValueError(f"unknown roll profile: {profile_name}")
    profile = profiles[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"roll profile {profile_name} must be a mapping")
    return {"name": profile_name, "model_version": document["model_version"], **profile}
