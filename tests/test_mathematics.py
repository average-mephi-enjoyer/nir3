import unittest
from datetime import datetime, timedelta, timezone
import numpy as np
from core.models import MarkerVector
from profiling.profile import ContaminationResistantProfile
from profiling.sequential_decision import ConformalMartingaleDetector
from benchmark.metrics import BenchmarkEvaluator


class Mathematics(unittest.TestCase):
    def test_ema_mean_and_variance_hand_calculation(self):
        p = ContaminationResistantProfile(alpha_ema=.5, protect_anchor=False)
        p.fit([MarkerVector(pronoun_marker=0), MarkerVector(pronoun_marker=2)])
        p._update_trusted_baseline({"pronoun_marker": 3})
        self.assertEqual(p.trusted_means["pronoun_marker"], 2)
        self.assertEqual(p.trusted_variances["pronoun_marker"], 2)

    def test_rank_p_and_log_product(self):
        detector = ConformalMartingaleDetector(betting_power=.5)
        for value in range(10):
            detector.calibrate(value)
        p, value, _ = detector.update(10)
        self.assertAlmostEqual(p, 1/11)
        self.assertAlmostEqual(value, .5 * np.sqrt(11))
        self.assertEqual(len(detector.calibration_scores), 10)

    def test_metrics_unknowns_and_union_exposure(self):
        ap, recall = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr([1, 0, -1], [1, 0, 2])
        self.assertEqual((ap, recall), (1, 1))
        self.assertEqual(BenchmarkEvaluator.calculate_contamination_rate(4, 1), 25)
        t = datetime(2020, 1, 1, tzinfo=timezone.utc)
        days = BenchmarkEvaluator.calendar_normal_exposure(t, t + timedelta(days=10),
            intervals=[(t + timedelta(days=2), t + timedelta(days=6)), (t + timedelta(days=4), t + timedelta(days=8))])
        self.assertEqual(days, 4)

    def test_separate_variance_anchor_blocks_before_update(self):
        p = ContaminationResistantProfile(anchor_limit=100, anchor_variance_ratio=1, alpha_ema=.5)
        p.fit([MarkerVector(pronoun_marker=0), MarkerVector(pronoun_marker=2)])
        before = dict(p.trusted_means)
        self.assertFalse(p._update_trusted_baseline({"pronoun_marker": 10}))
        self.assertEqual(p.trusted_means, before)
