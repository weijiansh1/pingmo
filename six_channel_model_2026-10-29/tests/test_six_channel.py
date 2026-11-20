"""CPU checks: transfer functions, physical relations, kinematics, grading and library."""
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy import integrate, signal

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from pinmo6 import G  # noqa: E402
from pinmo6.aircraft import SixChannelParameters, simulate  # noqa: E402
from pinmo6.kinematics import body_to_ned, body_velocity, euler_rates  # noqa: E402
from pinmo6.kinematics import stability_to_body as _scalar_stability_to_body  # noqa: E402
from pinmo6.lateral import directional_polynomials  # noqa: E402
from pinmo6.longitudinal import (classify_longitudinal, frequency_level, longitudinal_polynomials,  # noqa: E402
                                 stick_force_bounds, stick_force_level)
from pinmo6.sampler import SOURCE_LIBRARY, load_joint_library  # noqa: E402
from src.aircraft.p_channel import PChannel, p_channel_polynomials  # noqa: E402

LIBRARY = HERE / "library/joint_six_channel_5000_v1/plants.jsonl"
DT = 0.001
ENTRIES = load_joint_library(LIBRARY)


def _sample(count: int = 9) -> list[SixChannelParameters]:
    step = len(ENTRIES) // count
    return [ENTRIES[index * step][1] for index in range(count)]


def _grid_delays(parameters: SixChannelParameters) -> SixChannelParameters:
    """Round delays onto the plant grid so ZOH equals the continuous response exactly."""
    lon, roll, directional = parameters.longitudinal, parameters.roll, parameters.directional
    snap = lambda tau: round(tau / DT) * DT  # noqa: E731
    return replace(parameters,
                   longitudinal=replace(lon, tau_theta=snap(lon.tau_theta)),
                   roll=replace(roll, tau_p=snap(roll.tau_p)),
                   directional=replace(directional, tau_rp=snap(directional.tau_rp)))


def _delayed_step(numerator, denominator, time, tau):
    _, response = signal.step((numerator, denominator), T=time)
    shift = round(tau / DT)
    return np.concatenate((np.zeros(shift), response[:len(time) - shift]))


def _step_inputs(channel: int, steps: int, amplitude: float = 1.0) -> np.ndarray:
    inputs = np.zeros((steps, 3))
    inputs[:, channel] = amplitude
    return inputs


class TransferFunctionTests(unittest.TestCase):
    def test_each_output_matches_its_transfer_function(self):
        time = np.arange(6000) * DT
        for parameters in map(_grid_delays, _sample()):
            lon, roll, directional = parameters.longitudinal, parameters.roll, parameters.directional
            pitch = simulate(parameters, _step_inputs(0, len(time)), inertial_coupling=False)
            lateral = simulate(parameters, _step_inputs(1, len(time)), inertial_coupling=False)
            pedal = simulate(parameters, _step_inputs(2, len(time)), inertial_coupling=False)
            lon_tf = longitudinal_polynomials(lon)
            dir_tf = directional_polynomials(roll, directional)
            cases = [
                ("q", pitch["states"][:, 2], lon_tf["q"], lon.tau_theta),
                ("alpha", pitch["states"][:, 1] - lon.alpha_trim, lon_tf["alpha"], lon.tau_theta),
                ("nz", pitch["nz_g"], lon_tf["nz"], lon.tau_theta),
                ("p", lateral["states"][:, 5], p_channel_polynomials(roll), roll.tau_p),
                ("beta", pedal["states"][:, 4], dir_tf["beta"], directional.tau_rp),
                ("r", pedal["states"][:, 6], dir_tf["r"], directional.tau_rp),
            ]
            for name, simulated, (num, den), tau in cases:
                reference = _delayed_step(num, den, time, tau)
                error = np.max(np.abs(simulated - reference)) / max(np.max(np.abs(reference)), 1e-12)
                self.assertLess(error, 1e-6, name)

    def test_roll_channel_is_the_frozen_p_channel(self):
        rng = np.random.default_rng(0)
        for parameters in _sample(5):
            inputs = rng.normal(size=(3000, 3)) * np.array([20.0, 10.0, 50.0])
            states = simulate(parameters, inputs)["states"]
            plant = PChannel(parameters.roll, dt=DT)
            frozen = np.array([plant.step(force)[0] for force in inputs[:, 1]])
            np.testing.assert_array_equal(states[:, 5], frozen)


def stability_to_body(p, r, alpha):
    return np.vectorize(_scalar_stability_to_body)(p, r, alpha)


class PhysicalRelationTests(unittest.TestCase):
    def test_gamma_follows_theta_through_first_order_lag(self):
        time = np.arange(8000) * DT
        for parameters in _sample(5):
            lon = parameters.longitudinal
            run = simulate(parameters, _step_inputs(0, len(time), 10.0), inertial_coupling=False)
            theta = run["states"][:, 3] - lon.alpha_trim
            _, gamma_ref, _ = signal.lsim(([1.0], [lon.t_theta2, 1.0]), theta, time)
            error = np.max(np.abs(run["gamma"] - gamma_ref)) / np.max(np.abs(gamma_ref))
            self.assertLess(error, 5e-3)
            np.testing.assert_allclose(run["states"][:, 1], run["states"][:, 3] - run["gamma"], atol=1e-12)

    def test_cap_is_initial_pitch_acceleration_over_steady_load_factor(self):
        for parameters in map(_grid_delays, _sample(5)):
            lon = parameters.longitudinal
            steady_nz = lon.k_nz / lon.omega_sp**2
            self.assertAlmostEqual(steady_nz * lon.fs_per_g, 1.0, places=12)
            q = simulate(parameters, _step_inputs(0, 600), inertial_coupling=False)["states"][:, 2]
            first = int(np.argmax(np.abs(q) > 0))
            pitch_acceleration = (4 * q[first] - q[first + 1]) / (2 * DT)  # slope of a t + b t^2 at 0+
            self.assertAlmostEqual(pitch_acceleration / steady_nz / lon.cap, 1.0, delta=0.005)

    def test_inputs_are_decoupled_without_inertial_coupling(self):
        rng = np.random.default_rng(1)
        parameters = _sample(1)[0]
        for channel, silent in ((0, (4, 5, 6)), (1, (1, 2, 4, 6)), (2, (1, 2, 5))):
            inputs = np.zeros((2000, 3))
            inputs[:, channel] = rng.normal(size=2000) * 10
            states = simulate(parameters, inputs, inertial_coupling=False)["states"]
            states[:, 1] -= parameters.longitudinal.alpha_trim
            self.assertTrue(np.all(states[:, list(silent)] == 0.0), channel)

    def test_inertial_coupling_drives_pitch_from_roll_and_yaw(self):
        parameters = _sample(3)[1]
        lon, inertia = parameters.longitudinal, parameters.inertia
        steps = 4000
        inputs = np.zeros((steps, 3))
        inputs[:, 1], inputs[:, 2] = 15.0, 80.0
        run = simulate(parameters, inputs)
        alpha, q = run["states"][:, 1], run["states"][:, 2]
        p, r = stability_to_body(run["states"][:, 5], run["states"][:, 6], alpha)
        self.assertGreater(np.max(np.abs(q)), 0.0)
        a = np.array([[-1 / lon.t_theta2, 1.0], [lon.m_alpha, lon.m_q]])

        def rhs(t, x):
            k = min(int(t / DT), steps - 1)
            return a @ x + np.array([0.0, inertia.pitch_acceleration(p[k], r[k])])

        solution = integrate.solve_ivp(rhs, (0, (steps - 1) * DT), [0.0, 0.0],
                                       t_eval=np.arange(steps) * DT, max_step=DT, rtol=1e-9, atol=1e-12)
        error = np.max(np.abs(solution.y[1] - q)) / np.max(np.abs(q))
        self.assertLess(error, 2e-3)


class KinematicsTests(unittest.TestCase):
    def test_coordinated_level_turn(self):
        v, phi = 120.0, math.radians(30)
        turn = G * math.tan(phi) / v
        q, r = turn * math.sin(phi), turn * math.cos(phi)
        phi_dot, theta_dot, psi_dot = euler_rates(phi, 0.0, 0.0, q, r)
        self.assertAlmostEqual(phi_dot, 0.0, places=14)
        self.assertAlmostEqual(theta_dot, 0.0, places=14)
        self.assertAlmostEqual(psi_dot, turn, places=14)

    def test_velocity_transform_preserves_speed(self):
        rotation = body_to_ned(0.3, -0.2, 1.1)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-14)
        velocity = rotation @ body_velocity(150.0, 0.08, -0.03)
        self.assertAlmostEqual(np.linalg.norm(velocity), 150.0, places=10)


    def test_stability_axis_roll_keeps_velocity_direction(self):
        parameters = _sample(1)[0]
        inputs = np.zeros((3000, 3))
        inputs[:1000, 1] = 10.0
        run = simulate(parameters, inputs, inertial_coupling=False)
        position = run["states"][:, 9:12]
        # rolling about body x instead gave ~40 m of sideways drift here
        self.assertLess(np.max(np.abs(position[:, 1])), 1e-9 * np.max(np.abs(position[:, 0])))
        self.assertLess(np.max(np.abs(run["gamma"])), 1e-12)

    def test_propagated_attitude_obeys_euler_rate_equations(self):
        parameters = _sample(3)[2]
        rng = np.random.default_rng(2)
        inputs = np.repeat(rng.normal(size=(40, 3)) * np.array([30.0, 8.0, 60.0]), 100, axis=0)
        states = simulate(parameters, inputs)["states"]
        alpha, q, phi, theta, psi = (states[:, i] for i in (1, 2, 7, 3, 8))
        p_body, r_body = stability_to_body(states[:, 5], states[:, 6], alpha)
        angles = np.column_stack((phi, theta, psi))
        middle = 0.5 * (angles[1:] + angles[:-1])
        # body rates are held over each step, so evaluate them at k and the angles mid-step
        rates = np.array([euler_rates(a[0], a[1], pb, qb, rb)
                          for a, pb, qb, rb in zip(middle, p_body[:-1], q[:-1], r_body[:-1])])
        finite = np.diff(angles, axis=0) / DT
        self.assertLess(np.max(np.abs(finite - rates)), 1e-6 * np.max(np.abs(rates)))


class GradingTests(unittest.TestCase):
    def test_frequency_boundaries(self):
        self.assertEqual(frequency_level(math.sqrt(0.28 * 10), 10), 1)
        self.assertEqual(frequency_level(math.sqrt(0.27 * 10), 10), 2)
        self.assertEqual(frequency_level(0.99, 1.5), 2)   # CAP 0.65 but omega_sp below 1.0
        self.assertEqual(frequency_level(math.sqrt(11 * 10), 10), 3)

    def test_stick_force_table_a10(self):
        bounds = stick_force_bounds(10.0)
        self.assertAlmostEqual(bounds[1][1], 109.0)
        self.assertAlmostEqual(bounds[1][0], 100.0 / 6)
        self.assertAlmostEqual(bounds[2][1], 163.0)
        self.assertEqual(stick_force_bounds(1.0)[1][1], 125.0)
        self.assertEqual(stick_force_level(260.0, 10.0), None)


class LibraryTests(unittest.TestCase):
    def test_roll_objects_and_quotas_are_preserved(self):
        source = [json.loads(line) for line in SOURCE_LIBRARY.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(source), len(ENTRIES))
        for row, (record, _) in zip(source, ENTRIES):
            self.assertEqual(row["plant_id"], record["plant_id"])
            for name, value in record["roll"]["parameters"].items():
                self.assertEqual(row["parameters"][name], value)
            if row["sampling_bucket"] != "OOD":
                self.assertEqual(record["joint_level"], row["gjb_level"])
                self.assertLessEqual(record["longitudinal"]["gjb_level"], row["gjb_level"])

    def test_stored_levels_reclassify(self):
        for record, parameters in ENTRIES:
            grade = classify_longitudinal(parameters.longitudinal)
            self.assertEqual(grade["component_levels"], record["longitudinal"]["gjb_component_levels"])


if __name__ == "__main__":
    unittest.main()
