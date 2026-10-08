"""The race-day inference pipeline: sensor to pit wall inside 25 ms.

Stage budgets are cumulative deadlines measured from the moment a frame
arrives, matching the article:

    sensor 0 ms -> edge inference 10 ms -> confidence gate 12 ms
    -> strategy output 18 ms -> pit wall dashboard 25 ms

Inference runs on a worker thread with a hard timeout. If the model hasn't
answered by its deadline the pipeline stops waiting and goes to the fallback
chain, so a slow model can never make the whole call late.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from typing import Callable

from .calibration import confidence as conf_of
from .drift import DriftDetector
from .fallback import PIT, STAY_OUT, FallbackChain
from .gate import ConfidenceGate, GateResult
from .model import LogisticModel
from .telemetry import Frame

STAGE_DEADLINES_MS = {"inference": 10.0, "gate": 12.0, "strategy": 18.0, "dashboard": 25.0}
HANDOFF_MARGIN_MS = 1.0


@dataclass
class Decision:
    """What reaches the pit wall, plus everything needed to audit it later."""

    lap: int
    action: str                      # PIT / STAY_OUT / CREW_CHIEF_CALL
    source: str                      # PRIMARY_MODEL / ADVISORY / RULES / CACHED / HUMAN
    confidence: float | None
    failure_mode: str | None         # why the model was bypassed, if it was
    model_advisory: str | None       # model's call shown next to a fallback in the middle zone
    reasons: list[str] = field(default_factory=list)
    stage_ms: dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    deadline_met: bool = True
    truth: bool | None = None        # ground truth, logged for post-race evaluation
    injected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class InferencePipeline:
    def __init__(self, model: LogisticModel, drift: DriftDetector,
                 gate: ConfidenceGate | None = None,
                 fallback: FallbackChain | None = None,
                 latency_injector: Callable[[Frame], float] | None = None,
                 deadlines_ms: dict[str, float] | None = None):
        self.model = model
        self.drift = drift
        self.gate = gate or ConfidenceGate()
        self.fallback = fallback or FallbackChain()
        self.latency_injector = latency_injector
        self.deadlines = deadlines_ms or STAGE_DEADLINES_MS
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="edge-infer")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "InferencePipeline":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ #

    def process(self, frame: Frame) -> Decision:
        t0 = time.perf_counter()
        elapsed = lambda: (time.perf_counter() - t0) * 1000.0  # noqa: E731
        stage_ms: dict[str, float] = {}
        context: dict = {"lap": frame.lap}

        failure: str | None = None
        p_pit: float | None = None

        # Failure mode 1: feature data unavailable (sensor dropout, telemetry gap)
        if frame.has_missing_features():
            failure = "FEATURE_UNAVAILABLE"
            context["missing"] = [k for k, v in zip(
                ("tire", "fuel", "gap_ahead", "gap_behind", "caution", "temp", "delta"),
                frame.features()) if v is None]

        # Failure mode 2: out-of-distribution input
        if failure is None:
            drifted = self.drift.check(frame)
            if drifted:
                failure = "OUT_OF_DISTRIBUTION"
                context["drifted"] = drifted

        # Edge inference under a hard deadline (failure mode 3: model timeout)
        if failure is None:
            # Stop waiting slightly before the deadline so the hand-off to the
            # fallback chain still lands inside the inference budget.
            budget_s = max(0.0, self.deadlines["inference"] - HANDOFF_MARGIN_MS - elapsed()) / 1000.0
            future = self._pool.submit(self._infer, frame)
            try:
                p_pit = future.result(timeout=budget_s)
            except FutureTimeout:
                failure = "MODEL_TIMEOUT"
                context["timeout_ms"] = self.deadlines["inference"]
        stage_ms["inference"] = round(elapsed(), 3)

        # Confidence gate (failure mode 4: low confidence)
        decision: Decision
        if p_pit is not None:
            conf = conf_of(p_pit)
            model_action = PIT if p_pit >= 0.5 else STAY_OUT
            verdict = self.gate.classify(conf)
            stage_ms["gate"] = round(elapsed(), 3)

            if verdict is GateResult.PRIMARY:
                self.fallback.cache.remember(frame.lap, model_action, conf)
                decision = Decision(frame.lap, model_action, "PRIMARY_MODEL", round(conf, 3),
                                    None, None, [f"model confidence {conf:.2f} >= {self.gate.upper}"])
            else:
                context["model"] = f"{model_action}@{conf:.2f}"
                fb = self.fallback.decide(frame, context)
                if verdict is GateResult.ADVISORY:
                    decision = Decision(frame.lap, fb.action, "ADVISORY", round(conf, 3),
                                        "LOW_CONFIDENCE", model_action,
                                        [f"model in middle zone ({conf:.2f}); showing {fb.tier} call", fb.reason])
                else:
                    decision = Decision(frame.lap, fb.action, fb.tier, fb.confidence,
                                        "LOW_CONFIDENCE", None,
                                        [f"model confidence {conf:.2f} < {self.gate.lower}", fb.reason])
        else:
            stage_ms["gate"] = round(elapsed(), 3)
            fb = self.fallback.decide(frame, context)
            decision = Decision(frame.lap, fb.action, fb.tier, fb.confidence, failure, None, [fb.reason])

        stage_ms["strategy"] = round(elapsed(), 3)
        # (A real system would render to the dashboard here.)
        stage_ms["dashboard"] = round(elapsed(), 3)

        decision.stage_ms = stage_ms
        decision.total_ms = stage_ms["dashboard"]
        decision.deadline_met = all(stage_ms[s] <= self.deadlines[s] for s in self.deadlines)
        decision.truth = frame.should_pit
        decision.injected = list(frame.injected)
        return decision

    def _infer(self, frame: Frame) -> float:
        if self.latency_injector:
            delay = self.latency_injector(frame)
            if delay > 0:
                time.sleep(delay)
        return self.model.predict_proba(frame)
