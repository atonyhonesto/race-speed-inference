"""Out-of-distribution detection.

If the car is seeing conditions the model never trained on (a heat spike, a
new track configuration), its confidence scores can be confidently wrong.
This detector flags any raw feature that sits too many standard deviations
from what the model saw in training, so the pipeline can route around the
model instead of trusting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .telemetry import FEATURES, Frame


@dataclass
class DriftDetector:
    means: dict[str, float]
    stds: dict[str, float]
    z_limit: float = 4.0

    @classmethod
    def fit(cls, frames: list[Frame], z_limit: float = 4.0) -> "DriftDetector":
        means, stds = {}, {}
        for name in FEATURES:
            values = [getattr(f, name) for f in frames if getattr(f, name) is not None]
            m = sum(values) / len(values)
            means[name] = m
            stds[name] = max(1e-6, math.sqrt(sum((v - m) ** 2 for v in values) / len(values)))
        return cls(means, stds, z_limit)

    def check(self, frame: Frame) -> list[str]:
        """Names of features outside the training envelope (empty list = in range)."""
        drifted = []
        for name in FEATURES:
            value = getattr(frame, name)
            if value is None:
                continue
            if abs(value - self.means[name]) / self.stds[name] > self.z_limit:
                drifted.append(name)
        return drifted
