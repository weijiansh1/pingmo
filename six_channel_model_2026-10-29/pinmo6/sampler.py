"""Joint longitudinal-lateral library built on the frozen 5,000-object roll library.

Each source roll object keeps its parameters, split and IV-A component levels
unchanged.  It gains one longitudinal parameter set whose GJB level does not
exceed the roll level, so the joint (worst-component) level equals the source
level and the 1/2/3/OOD quotas are preserved.  Within a longitudinal level,
one component is pushed to that level in rotation and the others are drawn
from level 1 up to it, mirroring the roll library's bucket rule.
Pedal channels and inertia ratios are drawn from research envelopes.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from pinmo6 import PINGMO
from pinmo6.aircraft import InertiaRatios, SixChannelParameters
from pinmo6.lateral import DIRECTIONAL_UNIT_DIM, DirectionalParameters, directional_from_unit
from pinmo6.longitudinal import (COMPONENTS, LONGITUDINAL_UNIT_DIM, LongitudinalParameters,
                                 classify_longitudinal, longitudinal_from_unit)
from src.aircraft.parameters import PChannelParameters

SOURCE_LIBRARY = PINGMO / "FQControl_Model/data/aircraft/generated/p_channel_library_iv_a_5000_v1/plants.jsonl"
ROLL_FIELDS = ("l_fa", "lambda_s", "t_r", "zeta_d", "omega_d", "r_omega", "r_zeta", "tau_p")
UNIT_DIM = LONGITUDINAL_UNIT_DIM + DIRECTIONAL_UNIT_DIM + 2


def roll_parameters(row: dict) -> PChannelParameters:
    return PChannelParameters(**{name: row["parameters"][name] for name in ROLL_FIELDS})


def _component_levels(level: int, pushed: str, rng: np.random.Generator) -> dict[str, int]:
    if level == 1:
        return dict.fromkeys(COMPONENTS, 1)
    return {name: level if name == pushed else int(rng.integers(1, level + 1)) for name in COMPONENTS}


def _inertia_from_unit(unit: np.ndarray) -> InertiaRatios:
    return InertiaRatios(k_zx=0.8 + 0.2 * float(unit[0]), k_xz=0.05 * float(unit[1]))


def build_joint_library(output_dir: str | Path, seed: int = 20261029,
                        source: Path = SOURCE_LIBRARY) -> Path:
    destination = Path(output_dir)
    if (destination / "plants.jsonl").exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    source_bytes = source.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode("utf-8").splitlines() if line.strip()]
    unit = qmc.Sobol(d=UNIT_DIM, scramble=True, seed=seed).random_base2(math.ceil(math.log2(len(rows))))
    rng = np.random.default_rng(seed)
    rotation = {2: 0, 3: 0}
    records = []
    for index, row in enumerate(rows):
        roll_level = row["gjb_level"] if row["sampling_bucket"] != "OOD" else None
        lon_level = int(rng.integers(1, (roll_level or 3) + 1))
        pushed = COMPONENTS[rotation.get(lon_level, 0) % len(COMPONENTS)]
        if lon_level in rotation:
            rotation[lon_level] += 1
        levels = _component_levels(lon_level, pushed, rng)
        latent = unit[index]
        longitudinal = longitudinal_from_unit(latent[:LONGITUDINAL_UNIT_DIM], levels)
        directional = directional_from_unit(latent[LONGITUDINAL_UNIT_DIM:LONGITUDINAL_UNIT_DIM + DIRECTIONAL_UNIT_DIM])
        inertia = _inertia_from_unit(latent[-2:])
        grade = classify_longitudinal(longitudinal)
        if grade["component_levels"] != levels:
            raise AssertionError(f"{row['plant_id']}: designed {levels}, classified {grade['component_levels']}")
        records.append({
            "plant_id": row["plant_id"],
            "split": row["split"],
            "sampling_bucket": row["sampling_bucket"],
            "joint_level": None if roll_level is None else max(roll_level, grade["level"]),
            "roll": {"parameters": {name: row["parameters"][name] for name in ROLL_FIELDS},
                     "gjb_level": row["gjb_level"],
                     "gjb_component_levels": row["gjb_component_levels"],
                     "ood_reasons": row["ood_reasons"]},
            "longitudinal": {"parameters": asdict(longitudinal),
                             "derived": {"t_theta2": longitudinal.t_theta2, "cap": longitudinal.cap,
                                         "k_theta": longitudinal.k_theta, "k_nz": longitudinal.k_nz},
                             "gjb_level": grade["level"],
                             "gjb_component_levels": grade["component_levels"],
                             "pushed_component": None if lon_level == 1 else pushed},
            "directional": asdict(directional),
            "inertia": asdict(inertia),
        })
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "plants.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n",
        encoding="utf-8")
    manifest = {
        "schema_version": "six-channel-joint-v1",
        "seed": seed,
        "total_plants": len(records),
        "source_library": str(source.relative_to(PINGMO)),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "profile": "IV-A, centre stick, n_L = 7",
        "longitudinal_sources": {
            "frequency": "GJB 2874-97 Fig. A3(a), PDF 78",
            "damping": "Table A6, PDF 77",
            "delay": "Table A7, PDF 80",
            "stick_force": "Table A10(a), PDF 154",
        },
        "research_envelopes": {
            "v0_m_s": [60.0, 250.0], "inv_t_theta2_1_s": [0.4, 2.5],
            "sideslip_per_force_deg_per_n": [0.02, 0.2], "r_omega_r": [0.1, 0.5],
            "zeta_r": [0.05, 0.6], "tau_rp_s": [0.001, 0.2],
            "k_zx": [0.8, 1.0], "k_xz": [0.0, 0.05],
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return destination


def load_joint_library(path: str | Path) -> list[tuple[dict, SixChannelParameters]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        out.append((record, SixChannelParameters(
            longitudinal=LongitudinalParameters(**record["longitudinal"]["parameters"]),
            roll=PChannelParameters(**record["roll"]["parameters"]),
            directional=DirectionalParameters(**record["directional"]),
            inertia=InertiaRatios(**record["inertia"]),
        )))
    return out
