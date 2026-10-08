"""racespeed: a reference implementation of low-latency inference with
confidence gating and hierarchical fallback, framed around a race-day pit call.

Companion code for the LinkedIn article "AI/ML at Race Speed".
"""

from .calibration import expected_calibration_error, fit_temperature
from .drift import DriftDetector
from .fallback import FallbackChain
from .gate import ConfidenceGate, GateResult
from .model import LogisticModel
from .pipeline import Decision, InferencePipeline
from .telemetry import FaultPlan, Frame, RaceSimulator, simulate_races

__all__ = [
    "ConfidenceGate", "Decision", "DriftDetector", "FallbackChain", "FaultPlan",
    "Frame", "GateResult", "InferencePipeline", "LogisticModel", "RaceSimulator",
    "expected_calibration_error", "fit_temperature", "simulate_races",
]
