"""Physical roll-loop boundaries; the learned central control law stays frozen."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch


def test_surface_actuator_limits_actual_deflection_and_slew() -> None:
    # A request is a position; clipping only the network action cannot enforce slew.
    from src.aircraft.actuator import SurfaceActuator, SurfaceActuatorConfig

    actuator = SurfaceActuator(SurfaceActuatorConfig(
        dt_s=0.02, deflection_limit_rad=math.radians(10),
        rate_limit_rad_s=math.radians(40)))
    first = actuator.step(math.radians(20))
    assert math.degrees(first.requested_rad) == pytest.approx(20)
    assert math.degrees(first.applied_rad) == pytest.approx(0.8)
    assert first.amplitude_limited and first.rate_limited
    for _ in range(30):
        last = actuator.step(math.radians(20))
    assert math.degrees(last.applied_rad) == pytest.approx(10)
    reverse = actuator.step(math.radians(-10))
    assert math.degrees(reverse.applied_rad) == pytest.approx(9.2)
    actuator.reset()
    assert actuator.state.applied_rad == 0
    with pytest.raises(ValueError):
        actuator.step(float("nan"))


def test_bank_outer_loop_uses_estimated_bank_and_body_rate_kinematics() -> None:
    # Using p = phi_dot when q/r or pitch are nonzero gives a wrong body-rate demand.
    from src.controllers.roll_autopilot import RollOuterLoop, RollStateEstimate

    loop = RollOuterLoop()
    level = RollStateEstimate(0.0, roll_rad=math.radians(5), p_rad_s=0.0)
    result = loop.rate_setpoint(math.radians(25), level)
    assert math.degrees(result) == pytest.approx(10)  # (25-5)/2 s
    coupled = RollStateEstimate(0.0, roll_rad=math.radians(30), p_rad_s=0.0,
                               pitch_rad=math.radians(45), q_rad_s=0.2,
                               r_rad_s=0.4)
    result = loop.rate_setpoint(math.radians(30), coupled)
    assert result == pytest.approx(-0.4464101615137755)
    # A 60 degree request is first limited to the configured 30 degree bank.
    assert math.degrees(loop.rate_setpoint(math.radians(60), level)) == pytest.approx(12.5)
    far = RollStateEstimate(0.0, roll_rad=math.radians(-80), p_rad_s=0.0)
    assert math.degrees(loop.rate_setpoint(math.radians(30), far)) == pytest.approx(30)


def test_autopilot_command_modes_polarity_and_reset() -> None:
    from src.controllers.roll_autopilot import PingmoRateController, RollAutopilot, RollStateEstimate

    # A full positive virtual action must request a negative DHC-6 aileron angle.
    central = PingmoRateController(lambda obs: np.array([1.0]))
    autopilot = RollAutopilot(central)
    state = RollStateEstimate(0.0, 0.0, 0.0)
    output = autopilot.step(state, roll_setpoint_rad=math.radians(20))
    assert math.degrees(output.p_setpoint_rad_s) == pytest.approx(10)
    assert math.degrees(output.inner.requested_aileron_rad) == pytest.approx(-10)
    assert math.degrees(output.surface.applied_rad) == pytest.approx(-0.8)
    with pytest.raises(ValueError):
        autopilot.step(state)
    with pytest.raises(ValueError):
        autopilot.step(state, roll_setpoint_rad=0.1, p_setpoint_rad_s=0.1)
    autopilot.reset()
    output = autopilot.step(state, p_setpoint_rad_s=math.radians(80))
    assert math.degrees(output.p_setpoint_rad_s) == pytest.approx(30)
    assert math.degrees(output.surface.applied_rad) == pytest.approx(-0.8)


def test_nonfinite_feedback_and_irregular_clock_do_not_reach_the_policy() -> None:
    from src.controllers.roll_autopilot import PingmoRateController, RollAutopilot, RollStateEstimate

    with pytest.raises(ValueError):
        RollStateEstimate(0, float("nan"), 0)
    with pytest.raises(ValueError):
        PingmoRateController(lambda obs: np.array([0]), dt_s=0.01)
    autopilot = RollAutopilot(PingmoRateController(lambda obs: np.array([0])))
    autopilot.step(RollStateEstimate(0, 0, 0), p_setpoint_rad_s=0.1)
    with pytest.raises(ValueError, match="20 ms|clock|time"):
        autopilot.step(RollStateEstimate(0.04, 0, 0), p_setpoint_rad_s=0.1)
    autopilot.reset()
    with pytest.raises(ValueError):
        autopilot.step(RollStateEstimate(0, 0, 0), p_setpoint_rad_s=float("inf"))


def test_physical_rate_loop_preserves_the_original_network_trajectory() -> None:
    # This catches reference-index, integral, history, sign and unit regressions
    # together, using an actual conditional network and two independent loops.
    from src.controllers.roll_autopilot import PingmoRateController, RollAutopilot, RollStateEstimate
    from src.student.dense.network import DenseConditionalStudent
    from src.transfer import fine_tune as ft

    torch.manual_seed(19)
    model = DenseConditionalStudent(35, 8, 1, width=16, residual_blocks=1,
                                    residual_scale=0.1, enforce_odd_policy=True).eval()
    theta = torch.zeros(8)
    policy = ft.make_controller(model, theta)
    profile = ft.RollRateCommandProfile("regression", "doublet", 12.0, onset_s=0.2,
                                        segment_duration_s=0.4, duration_s=2.0)
    batch = ft.command_batch_50hz((profile,))
    # Stable physical model: p[k+1] = .8 p[k] - .3 delta_a[k] (deg units).
    old, _ = ft.arx_rollout_numpy(policy, batch, ft.ArxPlantNumpy([-0.8], [-0.3], 1))
    plant = ft.ArxPlantNumpy([-0.8], [-0.3], 1)
    autopilot = RollAutopilot(PingmoRateController(policy))
    p, bank = 0.0, 0.0
    actual_p, actual_ref, actual_applied = [], [], []
    for k, command in enumerate(batch["command"][0]):
        output = autopilot.step(RollStateEstimate(k * 0.02, bank, p),
                                p_setpoint_rad_s=float(command))
        p_next = math.radians(plant.push(math.degrees(output.surface.applied_rad)))
        bank += 0.01 * (p + p_next)
        p = p_next
        actual_p.append(p)
        actual_ref.append(output.inner.next_reference_rad_s)
        actual_applied.append(output.surface.applied_rad)
    np.testing.assert_allclose(actual_ref, old["ref"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(actual_p, old["p"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(actual_applied, np.radians(-10 / 22 * old["applied_n"]),
                               atol=1e-6, rtol=0)


def test_deployment_loader_accepts_the_actual_dhc6_checkpoint_schema(tmp_path) -> None:
    # The offline script's private loader must not be required for deployment.
    from src.student.controller import load_student_controller
    from src.student.dense.network import DenseConditionalStudent

    torch.manual_seed(17)
    model = DenseConditionalStudent(35, 8, 1, width=16, residual_blocks=1).eval()
    payload = dict(schema_version="dhc6_unified_student_v1",
                   student_schema="dense_conditional_student_v1",
                   observation_dim=35, aircraft_parameter_dim=8, action_dim=1,
                   network_width=16, residual_blocks=1, residual_scale=0.1,
                   enforce_odd_policy=True, model=model.state_dict())
    path = tmp_path / "full.pt"
    torch.save(payload, path)
    theta, observation = np.ones(8, dtype=np.float32), np.ones(35, dtype=np.float32)
    policy, loaded = load_student_controller(path, theta)
    with torch.no_grad():
        expected = model(torch.tensor(observation)[None], torch.tensor(theta)[None]).numpy()[0]
    np.testing.assert_array_equal(policy.predict(observation), expected)
    assert loaded["schema_version"] == "dhc6_unified_student_v1"

