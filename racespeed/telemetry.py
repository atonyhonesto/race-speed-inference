"""Synthetic race telemetry.

Real ECU feeds are proprietary, so this module generates a believable stand-in:
one frame per lap with tire age, fuel window, gaps, caution flags and tire
fall-off. It can also inject the failure modes the article describes
(sensor dropout and out-of-distribution conditions) so the fallback paths get
exercised.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

FEATURES = (
    "tire_age_laps",
    "fuel_laps_remaining",
    "gap_ahead_s",
    "gap_behind_s",
    "caution",
    "track_temp_c",
    "lap_time_delta_s",
)


@dataclass
class Frame:
    """One lap's worth of telemetry as seen by the pit wall."""

    lap: int
    tire_age_laps: float | None
    fuel_laps_remaining: float | None
    gap_ahead_s: float | None
    gap_behind_s: float | None
    caution: float | None  # 1.0 under yellow, 0.0 under green
    track_temp_c: float | None
    lap_time_delta_s: float | None  # seconds slower than the stint's best lap
    should_pit: bool = False  # ground truth, only used for training/evaluation
    injected: list[str] = field(default_factory=list)

    def features(self) -> list[float | None]:
        return [getattr(self, name) for name in FEATURES]

    def has_missing_features(self) -> bool:
        return any(value is None for value in self.features())


@dataclass
class FaultPlan:
    """Which failures to inject while simulating a race."""

    sensor_dropout_rate: float = 0.0
    heat_spike_laps: range | None = None  # laps where track temp leaves the training range


class RaceSimulator:
    """Generates a race lap by lap.

    The hidden ground-truth strategy is deliberately fuzzy (noise on the
    labels) so the model is genuinely unsure on some laps. That uncertainty is
    what makes the confidence gate and fallback tiers worth demonstrating.
    """

    def __init__(self, seed: int, laps: int = 200, fuel_window_laps: int = 55,
                 faults: FaultPlan | None = None, label_noise: float = 0.08):
        self.rng = random.Random(seed)
        self.laps = laps
        self.fuel_window = fuel_window_laps
        self.faults = faults or FaultPlan()
        self.label_noise = label_noise

    def run(self) -> Iterator[Frame]:
        rng = self.rng
        tire_age = 0.0
        fuel = float(self.fuel_window)
        caution_laps_left = 0
        base_temp = rng.uniform(28.0, 42.0)

        for lap in range(1, self.laps + 1):
            # Cautions arrive randomly and last a few laps.
            if caution_laps_left == 0 and rng.random() < 0.035:
                caution_laps_left = rng.randint(3, 6)
            caution = 1.0 if caution_laps_left > 0 else 0.0
            caution_laps_left = max(0, caution_laps_left - 1)

            track_temp = base_temp + rng.gauss(0, 1.0)
            # Tire fall-off grows with age and heat.
            delta = 0.018 * tire_age + 0.004 * max(0.0, track_temp - 30) * tire_age / 10
            delta += rng.gauss(0, 0.15)
            gap_ahead = abs(rng.gauss(1.8, 1.2))
            gap_behind = abs(rng.gauss(1.8, 1.2))

            frame = Frame(
                lap=lap,
                tire_age_laps=tire_age,
                fuel_laps_remaining=fuel,
                gap_ahead_s=round(gap_ahead, 3),
                gap_behind_s=round(gap_behind, 3),
                caution=caution,
                track_temp_c=round(track_temp, 2),
                lap_time_delta_s=round(max(0.0, delta), 3),
            )
            clean_call = self._true_strategy(frame)
            # Labels carry some noise: strategists don't always agree on the
            # "right" call. The car itself follows the clean call.
            frame.should_pit = (not clean_call) if self.rng.random() < self.label_noise else clean_call
            self._inject_faults(frame)

            if clean_call:
                tire_age, fuel = 0.0, float(self.fuel_window)
            else:
                tire_age += 1.0
                fuel -= 0.6 if caution else 1.0  # cautions save fuel
            yield frame

    def _true_strategy(self, f: Frame) -> bool:
        """The 'right' call, as an experienced crew chief would make it."""
        assert f.fuel_laps_remaining is not None and f.tire_age_laps is not None
        if f.fuel_laps_remaining <= 3:
            decision = True
        elif f.caution and f.tire_age_laps >= 22:
            decision = True
        elif not f.caution and f.lap_time_delta_s and f.lap_time_delta_s > 1.15 and f.gap_behind_s and f.gap_behind_s > 2.5:
            decision = True
        else:
            decision = False
        return decision

    def _inject_faults(self, f: Frame) -> None:
        plan = self.faults
        if plan.sensor_dropout_rate and self.rng.random() < plan.sensor_dropout_rate:
            f.lap_time_delta_s = None
            f.injected.append("sensor_dropout")
        if plan.heat_spike_laps and f.lap in plan.heat_spike_laps:
            f.track_temp_c = (f.track_temp_c or 35.0) + 30.0
            f.injected.append("heat_spike")


def simulate_races(count: int, start_seed: int = 0, **kwargs) -> list[Frame]:
    """Concatenate several clean races, used as training/calibration data."""
    frames: list[Frame] = []
    for seed in range(start_seed, start_seed + count):
        frames.extend(RaceSimulator(seed=seed, **kwargs).run())
    return frames
