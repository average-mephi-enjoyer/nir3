from dataclasses import replace
from datetime import timedelta
import unittest
import numpy as np
from benchmark.drift_generator import DriftInjector
from benchmark.baseline_comparison import BaselineBenchmarkRunner
from benchmark.metrics import BenchmarkEvaluator


class Experiments(unittest.TestCase):
    def setUp(self):
        self.args = DriftInjector().generate_feature_scenario("step", duration_days=60, seed=21)

    def test_identical_disabled_mechanisms_match(self):
        overrides = {name: {"update": "static", "anchor": False, "freeze": False} for name in BaselineBenchmarkRunner.METHODS}
        rows, traces = BaselineBenchmarkRunner.evaluate_stream(*self.args, mechanism_overrides=overrides)
        for trace in traces.values():
            np.testing.assert_array_equal(trace.scores, traces["Static"].scores)
            self.assertEqual(trace.calibration_scores, traces["Static"].calibration_scores)
        self.assertEqual(len({row.threshold for row in rows}), 1)

    def test_future_and_labels_do_not_change_detection(self):
        _, full = BaselineBenchmarkRunner.evaluate_stream(*self.args)
        truncated = list(self.args)
        truncated[2] = truncated[2][:40]
        _, prefix = BaselineBenchmarkRunner.evaluate_stream(*truncated)
        flipped = list(self.args)
        flipped[2] = [replace(o, label=-1) for o in flipped[2]]
        _, unknown = BaselineBenchmarkRunner.evaluate_stream(*flipped)
        for method in full:
            np.testing.assert_allclose(full[method].scores[:40], prefix[method].scores, equal_nan=True)
            np.testing.assert_allclose(full[method].scores, unknown[method].scores, equal_nan=True)
            self.assertEqual(full[method].baselines, unknown[method].baselines)

    def test_calendar_exposure_includes_gaps(self):
        start = self.args[2][0].timestamp
        self.assertEqual(BenchmarkEvaluator.calendar_normal_exposure(start, start + timedelta(days=10), start + timedelta(days=5)), 5)

    def test_future_length_cannot_change_synthetic_role_prior(self):
        short = DriftInjector().generate_feature_scenario("ema_proxy_drift", duration_days=60, seed=21)
        long = DriftInjector().generate_feature_scenario("ema_proxy_drift", duration_days=90, seed=21)
        self.assertEqual(short[3], long[3])
        np.testing.assert_array_equal([o.markers.to_numpy() for o in short[2]],
                                      [o.markers.to_numpy() for o in long[2][:60]])

    def test_false_alert_review_and_later_attack(self):
        args = DriftInjector().generate_feature_scenario("false_alert_then_attack", duration_days=60, intensity=6, seed=21)
        schedule = [{"timestamp": args[2][15].timestamp, "training": [o.markers for o in args[0]],
                     "calibration_scores": [1.] * 20, "approved": True, "source": "scripted independent exercise review"}]
        rows, traces = BaselineBenchmarkRunner.evaluate_stream(*args, review_schedule=schedule)
        p = next(r for r in rows if r.method_name == "Protected")
        self.assertGreater(p.false_freezes, 0)
        self.assertTrue(traces["Protected"].reviews)
        self.assertEqual(p.detected_scenario, 1)
        self.assertGreater(p.frozen_days, 0)

    def test_ablation_is_separate_and_oracle_is_measured(self):
        rows, _ = BaselineBenchmarkRunner.evaluate_stream(*self.args, experiment="B")
        self.assertTrue(all(r.experiment == "B" for r in rows))
        self.assertTrue(all(np.isfinite(r.oracle_mean_error) for r in rows))
