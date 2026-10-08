"""Hierarchical fallback: what the system does when the model can't be trusted.

Tier 1 - rules: deterministic, microsecond-fast, cover the common scenarios.
         Returns None when a situation isn't covered.
Tier 2 - cached call: the last high-confidence model output, decayed lap by
         lap. Stale knowledge is still knowledge, until it isn't.
Tier 3 - human: escalate to the crew chief with every piece of context the
         system has. This tier always answers, so a call always reaches the wall.
"""

from __future__ import annotations

from dataclasses import dataclass

from .telemetry import Frame

PIT, STAY_OUT, CREW_CHIEF = "PIT", "STAY_OUT", "CREW_CHIEF_CALL"


@dataclass
class FallbackCall:
    action: str
    tier: str
    confidence: float | None
    reason: str


class RuleBook:
    """Tier 1: the distilled judgement of experienced engineers, as code."""

    def decide(self, f: Frame) -> FallbackCall | None:
        fuel, tires, caution = f.fuel_laps_remaining, f.tire_age_laps, f.caution
        if fuel is not None and fuel <= 3:
            return FallbackCall(PIT, "RULES", None, "fuel window closing (<=3 laps)")
        if caution and tires is not None and tires >= 20:
            return FallbackCall(PIT, "RULES", None, "caution with 20+ lap tires")
        if not caution and tires is not None and fuel is not None and tires < 10 and fuel > 10:
            return FallbackCall(STAY_OUT, "RULES", None, "fresh tires, fuel to spare, green flag")
        return None  # not a scenario the rules were written for


class CachedCall:
    """Tier 2: re-use the last trusted model call with confidence decay."""

    def __init__(self, decay_per_lap: float = 0.03, floor: float = 0.70):
        self.decay = decay_per_lap
        self.floor = floor
        self._last: tuple[int, str, float] | None = None  # (lap, action, confidence)

    def remember(self, lap: int, action: str, confidence: float) -> None:
        self._last = (lap, action, confidence)

    def decide(self, f: Frame) -> FallbackCall | None:
        if self._last is None:
            return None
        lap, action, conf = self._last
        decayed = conf - self.decay * (f.lap - lap)
        if decayed < self.floor:
            return None
        return FallbackCall(action, "CACHED", round(decayed, 3),
                            f"last trusted call from lap {lap}, decayed")


class CrewChief:
    """Tier 3: hand the decision to a human, with context surfaced."""

    def decide(self, f: Frame, context: dict) -> FallbackCall:
        summary = ", ".join(f"{k}={v}" for k, v in context.items())
        return FallbackCall(CREW_CHIEF, "HUMAN", None, f"escalated: {summary}")


class FallbackChain:
    def __init__(self, rules: RuleBook | None = None, cache: CachedCall | None = None,
                 human: CrewChief | None = None):
        self.rules = rules or RuleBook()
        self.cache = cache or CachedCall()
        self.human = human or CrewChief()

    def decide(self, f: Frame, context: dict) -> FallbackCall:
        return self.rules.decide(f) or self.cache.decide(f) or self.human.decide(f, context)
