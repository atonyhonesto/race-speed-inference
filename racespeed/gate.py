"""The dual-threshold confidence gate.

    confidence >= upper            -> PRIMARY   (model's call goes to the wall)
    lower <= confidence < upper    -> ADVISORY  (fallback's call, model shown alongside)
    confidence < lower             -> REJECT    (fallback only, model hidden)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateResult(str, Enum):
    PRIMARY = "PRIMARY"
    ADVISORY = "ADVISORY"
    REJECT = "REJECT"


@dataclass(frozen=True)
class ConfidenceGate:
    lower: float = 0.70
    upper: float = 0.85

    def __post_init__(self) -> None:
        if not 0.5 <= self.lower < self.upper <= 1.0:
            raise ValueError("thresholds must satisfy 0.5 <= lower < upper <= 1.0")

    def classify(self, confidence: float) -> GateResult:
        if confidence >= self.upper:
            return GateResult.PRIMARY
        if confidence >= self.lower:
            return GateResult.ADVISORY
        return GateResult.REJECT
