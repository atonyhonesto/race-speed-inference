"""Run a simulated 200-lap race through the pipeline and print a race report.

    python demo.py                 # default race with injected faults
    python demo.py --seed 42       # different race
    python demo.py --log out.jsonl # where to write the per-lap audit log
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter

from racespeed import (ConfidenceGate, DriftDetector, FaultPlan, InferencePipeline,
                       LogisticModel, RaceSimulator, expected_calibration_error,
                       fit_temperature, simulate_races)


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log", default="race_decisions.jsonl")
    parser.add_argument("--slow-rate", type=float, default=0.03,
                        help="share of laps where inference is artificially slowed past its budget")
    args = parser.parse_args()

    print("Training on 30 simulated races, calibrating on 10 more ...")
    train = simulate_races(30, start_seed=1000)
    calib = simulate_races(10, start_seed=2000)
    model = LogisticModel().fit(train)
    drift = DriftDetector.fit(train)

    labels = [f.should_pit for f in calib]
    ece_trained = expected_calibration_error([model.predict_proba(f) for f in calib], labels)

    # Deep networks tend to be overconfident. Simulate one by sharpening the
    # logits 3x, then show temperature scaling pulling it back into line.
    sharp = copy.deepcopy(model)
    sharp.weights = [w * 3 for w in sharp.weights]
    sharp.bias *= 3
    ece_sharp = expected_calibration_error([sharp.predict_proba(f) for f in calib], labels)
    t_sharp = fit_temperature(sharp, calib)
    ece_sharp_fixed = expected_calibration_error([sharp.predict_proba(f) for f in calib], labels)

    temperature = fit_temperature(model, calib)

    faults = FaultPlan(sensor_dropout_rate=0.03, heat_spike_laps=range(150, 158))
    race = RaceSimulator(seed=args.seed, faults=faults)
    jitter = random.Random(args.seed)
    slow = lambda frame: 0.030 if jitter.random() < args.slow_rate else 0.0  # noqa: E731

    decisions = []
    with InferencePipeline(model, drift, ConfidenceGate(0.70, 0.85), latency_injector=slow) as pipe:
        for frame in race.run():
            decisions.append(pipe.process(frame))

    with open(args.log, "w", encoding="utf-8") as fh:
        for d in decisions:
            fh.write(json.dumps(d.to_dict()) + "\n")

    sources = Counter(d.source for d in decisions)
    failures = Counter(d.failure_mode for d in decisions if d.failure_mode)
    totals = [d.total_ms for d in decisions]
    primary = [d for d in decisions if d.source == "PRIMARY_MODEL"]
    primary_acc = sum((d.action == "PIT") == d.truth for d in primary) / max(1, len(primary))

    print()
    print("Calibration (expected calibration error, lower is better)")
    print(f"  model as trained              ECE {ece_trained:.3f}   (fitted T={temperature:.2f})")
    print(f"  simulated overconfident net   ECE {ece_sharp:.3f}")
    print(f"  ...after temperature scaling  ECE {ece_sharp_fixed:.3f}   (fitted T={t_sharp:.2f})")
    print()
    print(f"Race report: {len(decisions)} laps, seed {args.seed}")
    print("-" * 52)
    print(f"{'Who made the call':<26}{'Laps':>8}{'Share':>10}")
    for src in ("PRIMARY_MODEL", "ADVISORY", "RULES", "CACHED", "HUMAN"):
        n = sources.get(src, 0)
        print(f"{src:<26}{n:>8}{n / len(decisions):>10.1%}")
    print("-" * 52)
    print(f"{'Why the model was bypassed':<26}{'Laps':>8}")
    for mode, n in failures.most_common():
        print(f"{mode:<26}{n:>8}")
    print("-" * 52)
    print(f"Primary-model accuracy on laps it owned: {primary_acc:.1%}")
    print(f"Latency  p50 {percentile(totals, 50):.2f} ms   p99 {percentile(totals, 99):.2f} ms   "
          f"max {max(totals):.2f} ms   (budget 25 ms)")
    print(f"Every call inside its stage deadlines: {all(d.deadline_met for d in decisions)}")
    print(f"Audit log: {args.log}")


if __name__ == "__main__":
    main()
