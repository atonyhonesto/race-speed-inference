"""Calibration: making "91% confident" actually mean right about 91% of the time.

The article's second "thought to ponder" is that confidence without
calibration is noise. Temperature scaling is the simplest fix: divide the
logit by a single learned constant T, chosen on held-out data to minimise
negative log-likelihood. Expected Calibration Error (ECE) measures the gap.
"""

from __future__ import annotations

import math

from .model import LogisticModel, sigmoid
from .telemetry import Frame


def confidence(p_pit: float) -> float:
    """Confidence in whichever call the model is making (pit or stay out)."""
    return max(p_pit, 1.0 - p_pit)


def expected_calibration_error(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    """Weighted average |accuracy - confidence| across confidence bins."""
    total = len(probs)
    if total == 0:
        return 0.0
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for p, y in zip(probs, labels):
        conf = confidence(p)
        correct = (p >= 0.5) == y
        # confidence lives in [0.5, 1.0]; spread the bins across that range
        idx = min(bins - 1, int((conf - 0.5) / 0.5 * bins))
        buckets[idx].append((conf, correct))
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, ok in bucket if ok) / len(bucket)
        ece += len(bucket) / total * abs(accuracy - avg_conf)
    return ece


def _nll(logits: list[float], labels: list[bool], temperature: float) -> float:
    eps = 1e-12
    loss = 0.0
    for z, y in zip(logits, labels):
        p = sigmoid(z / temperature)
        loss -= math.log(p + eps) if y else math.log(1 - p + eps)
    return loss / len(logits)


def fit_temperature(model: LogisticModel, frames: list[Frame]) -> float:
    """Grid-search the temperature on a held-out calibration set and apply it."""
    logits = [model.logit(f) for f in frames]
    labels = [f.should_pit for f in frames]
    candidates = [0.25 + 0.05 * i for i in range(96)]  # 0.25 .. 5.0
    best = min(candidates, key=lambda t: _nll(logits, labels, t))
    model.temperature = best
    return best
