"""Causal FIFO delay with linear interpolation for fractional samples."""

from collections import deque
import math

import numpy as np


class FractionalDelay:
    def __init__(self, dt: float, delay_s: float) -> None:
        if dt <= 0 or delay_s < 0:
            raise ValueError("dt must be positive and delay_s non-negative")
        self.dt, self.delay_s = dt, delay_s
        self._whole = int(math.floor(delay_s / dt))
        self._fraction = delay_s / dt - self._whole
        self._history: deque[float] = deque(maxlen=self._whole + 3)
        self.reset()

    def reset(self) -> None:
        self._history.clear()
        self._history.extend([0.0] * self._history.maxlen)

    def push(self, value: float) -> float:
        self._history.append(float(value))
        newer = self._history[-(self._whole + 1)]
        if self._fraction == 0:
            return newer
        older = self._history[-(self._whole + 2)]
        return (1.0 - self._fraction) * newer + self._fraction * older

    def state(self, width: int) -> np.ndarray:
        """Return the causal FIFO, left-padded to a batch-stable width."""
        if width < len(self._history):
            raise ValueError("width cannot truncate the delay history")
        return np.pad(np.asarray(self._history, dtype=np.float32), (width - len(self._history), 0))
