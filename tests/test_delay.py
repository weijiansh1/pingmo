import pytest

from src.aircraft.delay import FractionalDelay


def test_fractional_delay_interpolates_between_samples() -> None:
    delay = FractionalDelay(dt=0.005, delay_s=0.0125)
    outputs = [delay.push(float(value)) for value in range(5)]
    assert outputs[:3] == [0.0, 0.0, 0.0]
    assert outputs[3] == pytest.approx(0.5)
    assert outputs[4] == pytest.approx(1.5)
