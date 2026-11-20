"""Coupled model: exact inversions, small-perturbation equivalence and large-motion physics."""
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import signal

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from pinmo6 import G  # noqa: E402
from pinmo6.coupled import CoupledAircraft, InertiaMoments, simulate_coupled  # noqa: E402
from pinmo6.derivatives import solve_lateral, solve_longitudinal  # noqa: E402
from pinmo6.sampler import load_joint_library  # noqa: E402
from pinmo6.turbulence import DrydenGust, DrydenScales, dryden_filters, filter_variance  # noqa: E402
from src.aircraft.delay import FractionalDelay  # noqa: E402
from src.aircraft.p_channel import PChannel, p_channel_polynomials  # noqa: E402

DT = 0.001
ENTRIES = load_joint_library(HERE / "library/joint_six_channel_5000_v1/plants.jsonl")


def _aircraft(index: int = 0, y_b: float = -0.12, l_r: float = 0.3, inertia=(3.0, 3.8, 0.05)):
    _, base = ENTRIES[index]
    lon = base.longitudinal
    n_rp = math.radians(base.directional.sideslip_per_force_deg_per_n) * base.roll.omega_d**2
    lateral = solve_lateral(base.roll, lon.v0, y_b, l_r, n_rp=n_rp, l_rp_over_n_rp=-0.1, y_rp_over_n_rp=0.002)
    longitudinal = solve_longitudinal(lon, x_u=-2 * G / (lon.v0 * 12.0), x_a_aero=-0.4 * G)
    return CoupledAircraft(lon, base.roll, base.directional.tau_rp, lateral, longitudinal,
                           InertiaMoments(*inertia), dt=DT), base


def _linear_response(a, b, column, delay_s, amplitude, steps):
    ad, bd, *_ = signal.cont2discrete((a, b[:, [column]], np.eye(len(a)), np.zeros((len(a), 1))), DT, "zoh")
    delay, x, out = FractionalDelay(DT, delay_s), np.zeros(len(a)), []
    for _ in range(steps):
        out.append(x.copy())
        x = ad @ x + bd[:, 0] * delay.push(amplitude)
    return np.array(out)


class InversionTests(unittest.TestCase):
    def test_lateral_inversion_keeps_frozen_roll_channel(self):
        for index in range(0, 5000, 250):
            _, base = ENTRIES[index]
            d = solve_lateral(base.roll, base.longitudinal.v0, -0.12, 0.3)
            a, b = d.matrices()
            num, den = signal.ss2tf(a, b[:, [0]], np.array([[0.0, 1.0, 0.0, 0.0]]), np.zeros((1, 1)))
            ref_num, ref_den = p_channel_polynomials(base.roll)
            np.testing.assert_allclose(den, ref_den, rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(num[0][-4:], ref_num, rtol=1e-9, atol=1e-9 * np.max(np.abs(ref_num)))

    def test_longitudinal_inversion_keeps_short_period(self):
        for index in range(0, 5000, 250):
            _, base = ENTRIES[index]
            lon = base.longitudinal
            d = solve_longitudinal(lon, x_u=-2 * G / (lon.v0 * 12.0), x_a_aero=-0.4 * G)
            a, b = d.matrices()
            num, den = signal.ss2tf(a, b[:, [0]], np.array([[0.0, 0.0, 1.0, 0.0]]), np.zeros((1, 1)))
            short = [1.0, 2 * lon.zeta_sp * lon.omega_sp, lon.omega_sp**2]
            quotient, remainder = np.polydiv(den, short)
            self.assertLess(np.max(np.abs(remainder)), 1e-9 * np.max(np.abs(den)))
            np.testing.assert_allclose(quotient, [1.0, *d.long_period], rtol=1e-9, atol=1e-12)
            self.assertAlmostEqual(np.polyval(num[0], -1.0 / lon.t_theta2) / lon.k_theta, 0.0, places=9)
            self.assertAlmostEqual(num[0][-4] / lon.k_theta, 1.0, places=10)


class SmallPerturbationTests(unittest.TestCase):
    def test_roll_rate_matches_frozen_channel(self):
        aircraft, base = _aircraft()
        inputs = np.zeros((5000, 3))
        inputs[:, 1] = 0.05
        p = simulate_coupled(aircraft, inputs)["states"][:, 5]
        plant = PChannel(base.roll, dt=DT)
        reference = np.array([plant.step(0.05)[0] for _ in range(5000)])
        self.assertLess(np.max(np.abs(p - reference)) / np.max(np.abs(reference)), 1e-3)

    def test_cross_channels_match_linear_model(self):
        aircraft, base = _aircraft()
        a, b = aircraft.lateral.matrices()
        for column, amplitude, delay in ((1, 0.05, base.roll.tau_p), (2, 0.2, base.directional.tau_rp)):
            inputs = np.zeros((5000, 3))
            inputs[:, column] = amplitude
            states = simulate_coupled(aircraft, inputs)["states"]
            linear = _linear_response(a, b, column - 1, delay, amplitude, 5000)
            for state_index, linear_index in ((4, 0), (5, 1), (6, 2)):
                scale = np.max(np.abs(linear[:, linear_index]))
                self.assertLess(np.max(np.abs(states[:, state_index] - linear[:, linear_index])) / scale, 2e-3)

    def test_pitch_channel_matches_linear_model(self):
        aircraft, base = _aircraft()
        lon = base.longitudinal
        a, b = aircraft.longitudinal.matrices()
        amplitude = 0.01 * lon.fs_per_g
        inputs = np.zeros((6000, 3))
        inputs[:, 0] = amplitude
        states = simulate_coupled(aircraft, inputs)["states"]
        linear = _linear_response(a, b, 0, lon.tau_theta, amplitude, 6000)
        pairs = ((0, states[:, 0] - lon.v0), (1, states[:, 1] - lon.alpha_trim), (2, states[:, 2]))
        for index, simulated in pairs:
            scale = np.max(np.abs(linear[:, index]))
            self.assertLess(np.max(np.abs(simulated - linear[:, index])) / scale, 2e-3)


class LargeMotionTests(unittest.TestCase):
    def test_trim_is_an_equilibrium(self):
        aircraft, base = _aircraft()
        states = simulate_coupled(aircraft, np.zeros((3000, 3)))["states"]
        expected = np.array([base.longitudinal.v0, base.longitudinal.alpha_trim, 0.0,
                             base.longitudinal.alpha_trim, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(states[-1, :9], expected, atol=1e-12)

    def test_bank_without_pull_descends_and_turns_toward_the_bank(self):
        aircraft, base = _aircraft()
        aircraft.reset()
        aircraft._dcm = aircraft._dcm @ np.array([[1, 0, 0], [0, math.cos(0.5), -math.sin(0.5)],
                                                  [0, math.sin(0.5), math.cos(0.5)]])
        aircraft._euler[0] = 0.5
        for _ in range(2000):
            state = aircraft.step(0.0, 0.0, 0.0)
        self.assertLess(aircraft.gamma, 0.0)        # lost vertical lift
        self.assertGreater(state[8], 0.0)           # heading moves right in a right bank

    def test_gyroscopic_terms_solve_euler_equations(self):
        inertia = InertiaMoments(2.5, 3.1, 0.12)
        p, q, r = 0.7, -0.3, 0.4
        dp, dq, dr = inertia.gyroscopic(p, q, r)
        tensor = np.array([[1.0, 0, -inertia.ixz], [0, inertia.iy, 0], [-inertia.ixz, 0, inertia.iz]])
        omega = np.array([p, q, r])
        np.testing.assert_allclose(tensor @ np.array([dp, dq, dr]), -np.cross(omega, tensor @ omega), atol=1e-14)

    def test_torque_free_tumble_conserves_angular_momentum(self):
        # no aerodynamic moments: the inertial angular momentum C_bn J omega_b must stay fixed while
        # alpha and beta wander far from trim (rates about a frame turning with alpha would not)
        aircraft, _ = _aircraft()
        lat, lon = aircraft.lateral, aircraft.longitudinal
        aircraft = CoupledAircraft(aircraft.loes, aircraft.roll, 0.0,
                                   type(lat)(lat.g_over_v, *[0.0] * 12),
                                   type(lon)(G, 0.0, G, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0),
                                   InertiaMoments(3.0, 3.8, 0.4), dt=DT)
        aircraft._x[2], aircraft._x[4], aircraft._x[5] = 0.6, 2.5, -0.3
        tensor = np.diag([1.0, 3.0, 3.8])
        tensor[0, 2] = tensor[2, 0] = -0.4
        trim = aircraft.loes.alpha_trim

        def momentum():
            p_b = aircraft._x[4] * math.cos(trim) - aircraft._x[5] * math.sin(trim)
            r_b = aircraft._x[4] * math.sin(trim) + aircraft._x[5] * math.cos(trim)
            return aircraft._dcm @ tensor @ np.array([p_b, aircraft._x[2], r_b])

        start = momentum()
        alpha_swing = 0.0
        for _ in range(2000):
            state = aircraft.step(0.0, 0.0, 0.0)
            alpha_swing = max(alpha_swing, abs(state[1] - trim))
        self.assertGreater(alpha_swing, 0.3)
        self.assertLess(np.linalg.norm(momentum() - start) / np.linalg.norm(start), 5e-3)

    def test_downward_gust_unloads_the_wing_and_bends_the_path_down(self):
        aircraft, base = _aircraft()
        gust = np.zeros(6)
        gust[1] = 2.0 / base.longitudinal.v0       # w_g = +2 m/s (body z down), alpha_g = w_g / V
        aircraft.reset()
        aircraft.step(0.0, 0.0, 0.0, gust=gust)
        self.assertLess(aircraft.nz_g, 0.0)
        for _ in range(1500):
            aircraft.step(0.0, 0.0, 0.0, gust=gust)
        self.assertLess(aircraft.gamma, 0.0)


class TurbulenceTests(unittest.TestCase):
    def test_linear_gust_variance_equals_sigma_squared(self):
        scales = DrydenScales(533.4, 533.4, 266.7, 1.5, 1.2, 0.9, 10.0)
        filters = dryden_filters(scales, 100.0)
        for name, sigma in (("u", 1.5), ("v", 1.2), ("w", 0.9)):
            self.assertAlmostEqual(filter_variance(*filters[name]) / sigma**2, 1.0, delta=2e-3)

    def test_simulated_gusts_reach_the_specified_variance(self):
        scales = DrydenScales(200.0, 200.0, 100.0, 2.0, 2.0, 2.0, 10.0)
        generator = DrydenGust(scales, 100.0, 0.01, seed=3)
        samples = np.array([generator.step() for _ in range(200000)])
        self.assertAlmostEqual(samples[:, 0].var() / 4.0, 1.0, delta=0.08)
        self.assertAlmostEqual((samples[:, 1] * 100.0).var() / 4.0, 1.0, delta=0.08)


    def test_rotary_gust_signs_follow_the_air_relative_convention(self):
        scales = DrydenScales(200.0, 200.0, 100.0, 2.0, 2.0, 2.0, 2.0)
        generator = DrydenGust(scales, 100.0, 0.01, seed=5)
        samples = np.array([generator.step() for _ in range(50000)])
        alpha_rate, beta_rate = np.diff(samples[:, 1]) / 0.01, np.diff(samples[:, 3]) / 0.01
        # q_g = -dw_g/dx = -alpha_g_dot, r_g = +dv_g/dx = +beta_g_dot (frozen field)
        self.assertGreater(np.corrcoef(samples[1:, 2], -alpha_rate)[0, 1], 0.5)
        self.assertGreater(np.corrcoef(samples[1:, 5], beta_rate)[0, 1], 0.5)


class CriteriaTests(unittest.TestCase):
    def test_dutch_roll_correction_raises_the_damping_floor(self):
        from pinmo6.criteria import dutch_roll_level
        from src.aircraft.parameters import PChannelParameters
        roll = PChannelParameters(0.2, -0.05, 0.5, 0.2, 3.0, 1.0, 1.0, 0.05)   # zeta*omega = 0.6
        self.assertEqual(dutch_roll_level(roll, 1.0)["level"], 1)           # 9 < 20, no correction
        # omega_d^2 |phi/beta| = 9 * 4 = 36: level-1 floor 0.35 + 0.014 * 16 = 0.574 < 0.6
        self.assertEqual(dutch_roll_level(roll, 4.0)["level"], 1)
        # 9 * 6 = 54: floor 0.35 + 0.014 * 34 = 0.826 > 0.6; level 2 floor 0.05 + 0.009 * 34 = 0.356
        self.assertEqual(dutch_roll_level(roll, 6.0)["level"], 2)

    def test_figure_boundaries_interpolate_between_breakpoints(self):
        from pinmo6.criteria import ROLL_OSCILLATION, SIDESLIP_EXCURSION, _boundary
        self.assertAlmostEqual(_boundary(ROLL_OSCILLATION[1], -165.0), 0.15)
        self.assertAlmostEqual(_boundary(ROLL_OSCILLATION[2], -300.0), 0.60 - 0.5 * 30 / 80)
        self.assertAlmostEqual(_boundary(SIDESLIP_EXCURSION[1], -240.0), 6.0)
        self.assertAlmostEqual(_boundary(SIDESLIP_EXCURSION[1], -20.0), 2.0)


V2_LIBRARY = HERE / "library/coupled_six_channel_5000_v2/plants.jsonl"


@unittest.skipUnless(V2_LIBRARY.exists(), "coupled v2 library not built")
class CoupledLibraryTests(unittest.TestCase):
    def test_every_object_keeps_its_roll_channel_and_short_period(self):
        import json
        from pinmo6.derivatives import LateralDerivatives, LongitudinalDerivatives
        extras = {json.loads(line)["plant_id"]: json.loads(line)
                  for line in V2_LIBRARY.read_text(encoding="utf-8").splitlines()}
        self.assertEqual(len(extras), len(ENTRIES))
        for record, base in ENTRIES[::10]:
            extra = extras[record["plant_id"]]
            lateral = LateralDerivatives(**extra["lateral"])
            a, b = lateral.matrices()
            num, den = signal.ss2tf(a, b[:, [0]], np.array([[0.0, 1.0, 0.0, 0.0]]), np.zeros((1, 1)))
            ref_num, ref_den = p_channel_polynomials(base.roll)
            np.testing.assert_allclose(den, ref_den, rtol=1e-8, atol=1e-8)
            np.testing.assert_allclose(num[0][-4:], ref_num, rtol=1e-8, atol=1e-8 * np.max(np.abs(ref_num)))
            fields = {k: v for k, v in extra["longitudinal"].items() if k in LongitudinalDerivatives.__dataclass_fields__}
            fields["long_period"] = tuple(fields["long_period"])
            a, b = LongitudinalDerivatives(**fields).matrices()
            _, den = signal.ss2tf(a, b[:, [0]], np.array([[0.0, 0.0, 1.0, 0.0]]), np.zeros((1, 1)))
            lon = base.longitudinal
            _, remainder = np.polydiv(den, [1.0, 2 * lon.zeta_sp * lon.omega_sp, lon.omega_sp**2])
            self.assertLess(np.max(np.abs(remainder)), 1e-8 * np.max(np.abs(den)))
            self.assertIn(extra["lateral_selection"],
                          {"within_envelope", "within_envelope_no_downgrade", "closest_to_envelope"})


if __name__ == "__main__":
    unittest.main()
