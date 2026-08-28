"""Online response costs evaluated at the plant integration rate."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ResponseCostSample:
    wrong_way: float = 0.0
    added_delay: float = 0.0
    sensitivity: float = 0.0
    oscillation: float = 0.0
    spiral_recovery: float = 0.0


class OnlineResponseCostTracker:
    """Compare action-dependent response timing with the matched raw plant.

    The sampled transport delay itself is never penalized.  Only delay added by
    the augmentation, wrong-way motion, and response-level sensitivity excess
    can change the Actor's return.
    """

    def __init__(
        self,
        *,
        plant_dt_s: float,
        force_scale_n: float,
        roll_rate_scale_rad_s: float,
        transport_delay_s: float,
        command_kind: str,
        onset_fraction: float = 0.01,
        delay_scale_s: float = 0.10,
        sensitivity_limit_deg_per_n: float = 3.38,
        edge_fraction: float = 0.02,
    ) -> None:
        if min(plant_dt_s, force_scale_n, roll_rate_scale_rad_s, delay_scale_s, sensitivity_limit_deg_per_n) <= 0:
            raise ValueError("response-cost scales and time steps must be positive")
        if not 0 < onset_fraction < 1 or not 0 < edge_fraction < 1 or transport_delay_s < 0:
            raise ValueError("invalid onset fraction or transport delay")
        self.plant_dt_s = plant_dt_s
        self.force_scale_n = force_scale_n
        self.roll_rate_scale_rad_s = roll_rate_scale_rad_s
        self.transport_delay_s = transport_delay_s
        self.command_kind = command_kind
        self.edge_threshold_n = edge_fraction * force_scale_n
        self._edge_metrics_enabled = command_kind in {
            "step", "pulse", "doublet", "square", "staircase", "piecewise", "legacy-step",
        }
        self.onset_threshold = max(onset_fraction * roll_rate_scale_rad_s, 1e-10)
        self.delay_scale_s = delay_scale_s
        self.sensitivity_limit_deg_per_n = sensitivity_limit_deg_per_n
        self.max_added_onset_delay_s = 0.0
        self.current_added_onset_delay_s = 0.0
        self.current_response_wait_s = 0.0
        self.latest_sensitivity_deg_per_n: float | None = None
        self._previous_force = 0.0
        self._edge_time_s: float | None = None
        self._edge_force_before = 0.0
        self._edge_force_delta = 0.0
        self._edge_roll_rate = 0.0
        self._edge_raw_roll_rate = 0.0
        self._edge_bank_angle = 0.0
        self._raw_onset_time_s: float | None = None
        self._controlled_onset_time_s: float | None = None
        self._sensitivity_recorded = False
        self._release_time_s: float | None = None
        self._edge_direction = 0.0
        self._previous_signed_response = 0.0
        self._previous_signed_slope = 0.0
        self._extrema: list[float] = []

    def reset(self, *, initial_force_n: float = 0.0, initial_roll_rate_rad_s: float = 0.0, initial_bank_angle_rad: float = 0.0) -> None:
        self.max_added_onset_delay_s = 0.0
        self.current_added_onset_delay_s = 0.0
        self.current_response_wait_s = 0.0
        self.latest_sensitivity_deg_per_n = None
        self._previous_force = initial_force_n
        self._edge_time_s = None
        self._edge_force_before = initial_force_n
        self._edge_force_delta = 0.0
        self._edge_roll_rate = initial_roll_rate_rad_s
        self._edge_raw_roll_rate = initial_roll_rate_rad_s
        self._edge_bank_angle = initial_bank_angle_rad
        self._raw_onset_time_s = None
        self._controlled_onset_time_s = None
        self._sensitivity_recorded = False
        self._release_time_s = None
        self._edge_direction = 0.0
        self._previous_signed_response = 0.0
        self._previous_signed_slope = 0.0
        self._extrema = []

    def update(
        self,
        *,
        time_s: float,
        force_n: float,
        roll_rate_rad_s: float,
        raw_roll_rate_rad_s: float,
        bank_angle_rad: float,
    ) -> ResponseCostSample:
        force_change = force_n - self._previous_force
        if self._edge_metrics_enabled and abs(force_change) >= self.edge_threshold_n:
            self._start_edge(time_s, force_n, roll_rate_rad_s, raw_roll_rate_rad_s, bank_angle_rad)

        added_delay_cost = 0.0
        sensitivity_cost = 0.0
        if self._edge_time_s is not None:
            self.current_response_wait_s = (
                max(0.0, time_s - self._edge_time_s)
                if self._controlled_onset_time_s is None
                else 0.0
            )
            if self._raw_onset_time_s is None and abs(raw_roll_rate_rad_s - self._edge_raw_roll_rate) >= self.onset_threshold:
                self._raw_onset_time_s = time_s
            if self._controlled_onset_time_s is None and abs(roll_rate_rad_s - self._edge_roll_rate) >= self.onset_threshold:
                self._controlled_onset_time_s = time_s
                if self._raw_onset_time_s is not None:
                    added = max(0.0, self._controlled_onset_time_s - self._raw_onset_time_s)
                    self.max_added_onset_delay_s = max(self.max_added_onset_delay_s, added)
                self.current_added_onset_delay_s = 0.0
                self.current_response_wait_s = 0.0
            if self._raw_onset_time_s is not None and self._controlled_onset_time_s is None:
                self.current_added_onset_delay_s = max(0.0, time_s - self._raw_onset_time_s)
                added_delay_cost = self.plant_dt_s / self.delay_scale_s

            elapsed = time_s - self._edge_time_s
            held_from_zero = abs(self._edge_force_before) <= 0.01 * self.force_scale_n
            still_held = abs(force_n - (self._edge_force_before + self._edge_force_delta)) <= 1e-12
            if not self._sensitivity_recorded and held_from_zero and still_held and elapsed >= 1.0:
                sensitivity = abs(math.degrees(bank_angle_rad - self._edge_bank_angle) / self._edge_force_delta)
                self.latest_sensitivity_deg_per_n = sensitivity
                excess = max(0.0, (sensitivity - self.sensitivity_limit_deg_per_n) / self.sensitivity_limit_deg_per_n)
                sensitivity_cost = excess * excess
                self._sensitivity_recorded = True

        wrong_way_cost = 0.0
        if abs(force_n) > 0.01 * self.force_scale_n:
            grace_complete = self._edge_time_s is None or time_s - self._edge_time_s >= self.transport_delay_s
            if grace_complete:
                signed_rate = roll_rate_rad_s if force_n > 0 else -roll_rate_rad_s
                wrong_way = max(0.0, -signed_rate / self.roll_rate_scale_rad_s)
                wrong_way_cost = wrong_way * wrong_way * self.plant_dt_s

        oscillation_cost = self._oscillation_cost(force_n, roll_rate_rad_s)
        spiral_recovery_cost = 0.0
        if self._release_time_s is not None and time_s - self._release_time_s >= self.transport_delay_s:
            normalized_rate = roll_rate_rad_s / self.roll_rate_scale_rad_s
            spiral_recovery_cost = normalized_rate * normalized_rate * self.plant_dt_s

        self._previous_force = force_n
        return ResponseCostSample(
            wrong_way_cost,
            added_delay_cost,
            sensitivity_cost,
            oscillation_cost,
            spiral_recovery_cost,
        )

    def _start_edge(self, time_s: float, force_n: float, roll_rate_rad_s: float, raw_roll_rate_rad_s: float, bank_angle_rad: float) -> None:
        self._edge_time_s = time_s
        self._edge_force_before = self._previous_force
        self._edge_force_delta = force_n - self._previous_force
        self._edge_roll_rate = roll_rate_rad_s
        self._edge_raw_roll_rate = raw_roll_rate_rad_s
        self._edge_bank_angle = bank_angle_rad
        self._raw_onset_time_s = None
        self._controlled_onset_time_s = None
        self.current_added_onset_delay_s = 0.0
        self.current_response_wait_s = 0.0
        self._sensitivity_recorded = False
        if abs(force_n) <= 0.01 * self.force_scale_n:
            self._release_time_s = time_s
            self._edge_direction = 0.0
            self._extrema = []
        else:
            self._release_time_s = None
            self._edge_direction = math.copysign(1.0, self._edge_force_delta)
            self._previous_signed_response = 0.0
            self._previous_signed_slope = 0.0
            self._extrema = []

    def _oscillation_cost(self, force_n: float, roll_rate_rad_s: float) -> float:
        """Emit the A120-shaped response ratio once per completed oscillation.

        This is a dense-training proxy, not a formal Figure A116 grade.  It is
        active only while a discrete command level is held and never tries to
        reinterpret sine/chirp command reversals as a modal oscillation.
        """

        if self._edge_direction == 0.0 or abs(force_n) <= 0.01 * self.force_scale_n:
            return 0.0
        signed_response = self._edge_direction * (roll_rate_rad_s - self._edge_roll_rate)
        signed_slope = signed_response - self._previous_signed_response
        emitted_cost = 0.0
        if self._previous_signed_slope > 0.0 >= signed_slope:
            self._extrema.append(self._previous_signed_response)
        elif self._previous_signed_slope < 0.0 <= signed_slope:
            self._extrema.append(self._previous_signed_response)

        if len(self._extrema) >= 3:
            p1, p2, p3 = self._extrema[-3:]
            denominator = p1 + p3 + 2.0 * p2
            if p1 > p2 < p3 and denominator > 1e-12:
                ratio = max(0.0, (p1 + p3 - 2.0 * p2) / denominator)
                emitted_cost = ratio * ratio
            self._extrema = self._extrema[-2:]

        self._previous_signed_response = signed_response
        self._previous_signed_slope = signed_slope
        return emitted_cost
