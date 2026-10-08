import time
import unittest

from racespeed import (ConfidenceGate, DriftDetector, FallbackChain, Frame, GateResult,
                       InferencePipeline, LogisticModel, RaceSimulator,
                       expected_calibration_error, fit_temperature, simulate_races)
from racespeed.fallback import CREW_CHIEF, PIT, STAY_OUT, CachedCall, RuleBook


def make_frame(**overrides) -> Frame:
    base = dict(lap=50, tire_age_laps=15.0, fuel_laps_remaining=20.0, gap_ahead_s=1.5,
                gap_behind_s=1.5, caution=0.0, track_temp_c=35.0, lap_time_delta_s=0.4)
    base.update(overrides)
    return Frame(**base)


class StubModel:
    """A model whose output and speed the test controls."""

    def __init__(self, p_pit: float, delay_s: float = 0.0):
        self.p_pit, self.delay_s = p_pit, delay_s

    def predict_proba(self, frame):
        if self.delay_s:
            time.sleep(self.delay_s)
        return self.p_pit


class Fixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = simulate_races(8, start_seed=500)
        cls.drift = DriftDetector.fit(cls.train)

    def pipeline(self, model, **kw):
        pipe = InferencePipeline(model, self.drift, **kw)
        self.addCleanup(pipe.close)
        return pipe


class GateTests(unittest.TestCase):
    def test_zones(self):
        gate = ConfidenceGate(0.70, 0.85)
        self.assertIs(gate.classify(0.95), GateResult.PRIMARY)
        self.assertIs(gate.classify(0.85), GateResult.PRIMARY)   # boundary is inclusive
        self.assertIs(gate.classify(0.80), GateResult.ADVISORY)
        self.assertIs(gate.classify(0.70), GateResult.ADVISORY)
        self.assertIs(gate.classify(0.69), GateResult.REJECT)

    def test_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            ConfidenceGate(0.9, 0.8)


class FallbackTests(unittest.TestCase):
    def test_rules_cover_fuel_emergency(self):
        call = RuleBook().decide(make_frame(fuel_laps_remaining=2.0))
        self.assertEqual(call.action, PIT)

    def test_rules_return_none_when_uncovered(self):
        self.assertIsNone(RuleBook().decide(make_frame()))

    def test_cache_decays_then_expires(self):
        cache = CachedCall(decay_per_lap=0.05, floor=0.70)
        cache.remember(lap=50, action=STAY_OUT, confidence=0.90)
        self.assertEqual(cache.decide(make_frame(lap=52)).action, STAY_OUT)
        self.assertIsNone(cache.decide(make_frame(lap=56)))  # 0.90 - 0.30 < 0.70

    def test_chain_always_answers(self):
        call = FallbackChain().decide(make_frame(), context={"lap": 50})
        self.assertEqual(call.action, CREW_CHIEF)
        self.assertEqual(call.tier, "HUMAN")


class PipelineTests(Fixture):
    def test_confident_model_goes_straight_to_the_wall(self):
        d = self.pipeline(StubModel(0.97)).process(make_frame())
        self.assertEqual((d.source, d.action, d.failure_mode), ("PRIMARY_MODEL", PIT, None))

    def test_middle_zone_shows_model_as_advisory(self):
        d = self.pipeline(StubModel(0.78)).process(make_frame(fuel_laps_remaining=2.0))
        self.assertEqual(d.source, "ADVISORY")
        self.assertEqual(d.model_advisory, PIT)
        self.assertEqual(d.action, PIT)  # the rules' call is what's acted on

    def test_low_confidence_falls_back(self):
        d = self.pipeline(StubModel(0.55)).process(make_frame(tire_age_laps=5.0))
        self.assertEqual((d.source, d.failure_mode), ("RULES", "LOW_CONFIDENCE"))

    def test_sensor_dropout_bypasses_model(self):
        d = self.pipeline(StubModel(0.99)).process(make_frame(lap_time_delta_s=None))
        self.assertEqual(d.failure_mode, "FEATURE_UNAVAILABLE")
        self.assertNotEqual(d.source, "PRIMARY_MODEL")

    def test_out_of_distribution_bypasses_model(self):
        d = self.pipeline(StubModel(0.99)).process(make_frame(track_temp_c=95.0))
        self.assertEqual(d.failure_mode, "OUT_OF_DISTRIBUTION")

    def test_slow_model_times_out_inside_budget(self):
        d = self.pipeline(StubModel(0.99, delay_s=0.05)).process(make_frame())
        self.assertEqual(d.failure_mode, "MODEL_TIMEOUT")
        self.assertLess(d.total_ms, 25.0)
        self.assertTrue(d.deadline_met)

    def test_primary_call_feeds_the_cache(self):
        pipe = self.pipeline(StubModel(0.95))
        pipe.process(make_frame(lap=10))  # p_pit=0.95 -> confident PIT, remembered
        pipe.model = StubModel(0.60)      # model goes shaky next lap
        d = pipe.process(make_frame(lap=11))
        self.assertEqual((d.source, d.action), ("CACHED", PIT))


class CalibrationTests(unittest.TestCase):
    def test_temperature_scaling_fixes_overconfidence(self):
        train = simulate_races(10, start_seed=10)
        calib = simulate_races(5, start_seed=50)
        model = LogisticModel().fit(train, epochs=150)
        model.weights = [w * 3 for w in model.weights]
        model.bias *= 3
        labels = [f.should_pit for f in calib]
        before = expected_calibration_error([model.predict_proba(f) for f in calib], labels)
        t = fit_temperature(model, calib)
        after = expected_calibration_error([model.predict_proba(f) for f in calib], labels)
        self.assertGreater(t, 1.5)
        self.assertLess(after, before)


class EndToEndTests(Fixture):
    def test_full_race_every_lap_gets_a_call(self):
        model = LogisticModel().fit(self.train, epochs=150)
        pipe = self.pipeline(model)
        decisions = [pipe.process(f) for f in RaceSimulator(seed=3).run()]
        self.assertEqual(len(decisions), 200)
        self.assertTrue(all(d.action for d in decisions))
        self.assertTrue(all(d.total_ms < 25.0 for d in decisions))


if __name__ == "__main__":
    unittest.main()
