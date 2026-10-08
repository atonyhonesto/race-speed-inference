"""The primary strategy model: a small logistic regression in pure Python.

The point of this repo is the system around the model, not the model itself,
so it is intentionally tiny and dependency-free. In production this slot would
hold an INT8-quantized network served by ONNX Runtime or TensorRT; the
pipeline treats it as a black box that returns a logit.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .telemetry import Frame


def engineer(frame: Frame) -> list[float]:
    """Turn raw telemetry into model inputs. Assumes no missing values."""
    tire = frame.tire_age_laps or 0.0
    fuel = frame.fuel_laps_remaining or 0.0
    caution = frame.caution or 0.0
    return [
        tire,
        fuel,
        max(0.0, 6.0 - fuel),          # fuel urgency ramps up near empty
        caution,
        caution * tire,                 # cheap stop under yellow on old tires
        frame.lap_time_delta_s or 0.0,
        frame.gap_behind_s or 0.0,
        frame.track_temp_c or 0.0,
    ]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


@dataclass
class LogisticModel:
    weights: list[float] = field(default_factory=list)
    bias: float = 0.0
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)
    temperature: float = 1.0  # set by calibration.fit_temperature

    def fit(self, frames: list[Frame], epochs: int = 300, lr: float = 0.3,
            l2: float = 1e-3, seed: int = 7) -> "LogisticModel":
        xs = [engineer(f) for f in frames]
        ys = [1.0 if f.should_pit else 0.0 for f in frames]
        n_features = len(xs[0])
        self.means = [sum(col) / len(col) for col in zip(*xs)]
        self.stds = [
            max(1e-6, math.sqrt(sum((v - m) ** 2 for v in col) / len(col)))
            for col, m in zip(zip(*xs), self.means)
        ]
        xs = [self._scale(x) for x in xs]

        rng = random.Random(seed)
        self.weights = [rng.gauss(0, 0.01) for _ in range(n_features)]
        self.bias = 0.0
        n = len(xs)
        # Full-batch gradient descent: slow to write about, fast enough to run.
        for _ in range(epochs):
            grad_w = [0.0] * n_features
            grad_b = 0.0
            for x, y in zip(xs, ys):
                err = sigmoid(self._linear(x)) - y
                for i, xi in enumerate(x):
                    grad_w[i] += err * xi
                grad_b += err
            for i in range(n_features):
                self.weights[i] -= lr * (grad_w[i] / n + l2 * self.weights[i])
            self.bias -= lr * grad_b / n
        return self

    def logit(self, frame: Frame) -> float:
        """Raw score before temperature scaling."""
        return self._linear(self._scale(engineer(frame)))

    def predict_proba(self, frame: Frame) -> float:
        """Calibrated probability that pitting this lap is the right call."""
        return sigmoid(self.logit(frame) / self.temperature)

    def _scale(self, x: list[float]) -> list[float]:
        return [(v - m) / s for v, m, s in zip(x, self.means, self.stds)]

    def _linear(self, x: list[float]) -> float:
        return sum(w * v for w, v in zip(self.weights, x)) + self.bias
