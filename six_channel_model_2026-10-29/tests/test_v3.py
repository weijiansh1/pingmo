"""v3 library: quotas, exact modes, real-aircraft envelopes and reproducible grades."""
from collections import Counter
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from pinmo6.criteria import (dutch_roll_eigenvector, dutch_roll_level, psi_beta_deg,  # noqa: E402
                             sideslip_excursion, steady_sideslip)
from pinmo6.library_v2 import load_envelopes  # noqa: E402
from pinmo6.longitudinal import classify_longitudinal  # noqa: E402
from pinmo6.plausibility import lateral_ratios, violation  # noqa: E402
from pinmo6.sampler_v3 import OMEGA_SP_MAX, SPLIT_COUNTS, ZETA_SP_MAX, load_v3  # noqa: E402
from src.aircraft.gain_calibration import sensitivity_1s_deg_per_n  # noqa: E402
from src.aircraft.p_channel import p_channel_polynomials  # noqa: E402
from src.aircraft.sampler import classify_iv_a_quality  # noqa: E402

LIBRARY = HERE / "library/six_channel_5000_v3/plants.jsonl"
ENTRIES = load_v3(LIBRARY) if LIBRARY.exists() else []


@unittest.skipUnless(ENTRIES, "v3 library not built")
class QuotaTests(unittest.TestCase):
    def test_splits_and_joint_levels(self):
        self.assertEqual(Counter(r["split"] for r, _ in ENTRIES), Counter(SPLIT_COUNTS))
        self.assertEqual(Counter(r["sampling_bucket"] for r, _ in ENTRIES),
                         Counter({"L1": 1500, "L2": 1500, "L3": 1500, "OOD": 500}))
        for record, _ in ENTRIES:
            levels = record["component_levels"].values()
            if record["sampling_bucket"] == "OOD":
                self.assertTrue(record["ood_reasons"])
            else:
                self.assertEqual(record["joint_level"], max(levels))
                self.assertEqual(f"L{record['joint_level']}", record["sampling_bucket"])


@unittest.skipUnless(ENTRIES, "v3 library not built")
class ExactnessTests(unittest.TestCase):
    def test_derivatives_reproduce_roll_channel_and_short_period(self):
        for record, parts in ENTRIES[::25]:
            a, b = parts["lateral"].matrices()
            num, den = signal.ss2tf(a, b[:, [0]], np.array([[0.0, 1.0, 0.0, 0.0]]), np.zeros((1, 1)))
            ref_num, ref_den = p_channel_polynomials(parts["roll"])
            np.testing.assert_allclose(den, ref_den, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(num[0][-4:], ref_num, rtol=1e-8, atol=1e-8 * np.max(np.abs(ref_num)))
            a, b = parts["longitudinal"].matrices()
            _, den = signal.ss2tf(a, b[:, [0]], np.array([[0.0, 0.0, 1.0, 0.0]]), np.zeros((1, 1)))
            loes = parts["loes"]
            _, remainder = np.polydiv(den, [1.0, 2 * loes.zeta_sp * loes.omega_sp, loes.omega_sp**2])
            self.assertLess(np.max(np.abs(remainder)), 1e-8 * np.max(np.abs(den)))

    def test_sampling_caps_and_sensitivity(self):
        for record, parts in ENTRIES:
            self.assertLessEqual(parts["loes"].zeta_sp, ZETA_SP_MAX + 1e-12)
            self.assertLessEqual(parts["loes"].omega_sp, OMEGA_SP_MAX + 1e-9)
        for record, parts in ENTRIES[::100]:
            measured = sensitivity_1s_deg_per_n(parts["roll"], dt=0.001)
            self.assertAlmostEqual(measured / record["roll"]["sensitivity_1s_deg_per_n"], 1.0, places=9)


@unittest.skipUnless(ENTRIES, "v3 library not built")
class RealismAndGradeTests(unittest.TestCase):
    def test_every_object_lies_inside_the_real_aircraft_envelope(self):
        ratios = {k: tuple(v) for k, v in load_envelopes()["lateral_ratios"].items()}
        for record, parts in ENTRIES:
            score, names = violation(lateral_ratios(parts["lateral"], parts["roll"]), ratios)
            self.assertEqual(score, 0.0, (record["plant_id"], names))

    def test_grades_recompute_from_the_stored_objects(self):
        for record, parts in ENTRIES[::40]:
            roll, lateral = parts["roll"], parts["lateral"]
            levels = record["component_levels"]
            roll_grade = classify_iv_a_quality(roll, record["roll"]["sensitivity_1s_deg_per_n"])
            for name, value in roll_grade["gjb_component_levels"].items():
                if name != "dutch_roll":            # stored with the |phi/beta| correction, checked below
                    self.assertEqual(levels[f"roll_{name}"], value)
            shape = dutch_roll_eigenvector(lateral, roll)
            self.assertEqual(levels["roll_dutch_roll"], dutch_roll_level(roll, shape["phi_over_beta"])["level"])
            psi = psi_beta_deg(lateral, roll)
            self.assertEqual(levels["sideslip_excursion"],
                             sideslip_excursion(lateral, roll, psi, speed_range=record["envelope"]["speed_range"])["level"])
            steady = steady_sideslip(lateral)
            self.assertEqual(levels["steady_sideslip_sense"],
                             1 if steady["dihedral_sense_ok"] and steady["pedal_sense_ok"] else 3)
            for name, value in classify_longitudinal(parts["loes"])["component_levels"].items():
                self.assertEqual(levels[f"pitch_{name}"], value)

    def test_objects_simulate_from_trim(self):
        for record, parts in ENTRIES[::500]:
            aircraft = parts["aircraft"]()
            state = None
            for _ in range(2000):
                state = aircraft.step(0.0, 0.0, 0.0)
            self.assertTrue(np.all(np.isfinite(state)))
            self.assertAlmostEqual(state[0], parts["loes"].v0, places=9)


if __name__ == "__main__":
    unittest.main()
