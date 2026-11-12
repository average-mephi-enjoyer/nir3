import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
from benchmark.baseline_comparison import BaselineBenchmarkRunner
from benchmark.drift_generator import DriftInjector
from benchmark.metrics import BenchmarkEvaluator
from core.models import CommunicationMessage, EmployeeContext, MarkerVector, RiskSignal, TelemetryEvent
from extractors.graph_topology import DynamicCommunicationGraph
from parsers.cert_ground_truth import load_cert_ground_truth
from profiling.profile import ContaminationResistantProfile
from profiling.quarantine import QuarantineBuffer
from profiling.sequential_decision import SequentialDecisionManager
from profiling.sliding_window import SlidingTimeWindow
from profiling.workload_filter import TeamWorkloadContextTracker
from verification.log_correlation import TargetedRetrospectiveVerifier


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.origin = datetime(2020, 1, 1, tzinfo=timezone.utc)

    def vector(self, value):
        return MarkerVector(**{marker: value for marker in MarkerVector.feature_names()})

    def test_quarantine_uses_calendar_and_whole_horizon(self):
        buffer = QuarantineBuffer()
        vector = self.vector(0.1)
        for i in range(20):
            buffer.enqueue(self.origin + timedelta(minutes=i), vector, {}, 0.5)
        self.assertEqual(buffer.extract_mature_clean_observations(self.origin + timedelta(days=1)), ([], []))
        buffer.enqueue(self.origin + timedelta(days=7), vector, {}, 3.0)
        approved, rejected = buffer.extract_mature_clean_observations(self.origin + timedelta(days=14, hours=1))
        self.assertEqual(len(approved), 0)
        self.assertEqual(len(rejected), 20)
        self.assertTrue(all(o.reason == "risk_in_quarantine_horizon" for o in rejected))

    def test_clean_quarantine_admission(self):
        buffer = QuarantineBuffer()
        buffer.enqueue(self.origin, self.vector(0.1), {}, 0.5)
        approved, rejected = buffer.extract_mature_clean_observations(self.origin + timedelta(days=14))
        self.assertEqual(len(approved), 1)
        self.assertFalse(rejected)

    def test_frozen_profile_and_explicit_resume(self):
        profile = ContaminationResistantProfile(quarantine_delay_days=1)
        profile.fit([self.vector(0.1), self.vector(0.2), self.vector(0.3)])
        means = dict(profile.trusted_means)
        profile.freeze_on_alert(self.origin)
        for day in range(40):
            profile.register_observation(self.vector(0.21), self.origin + timedelta(days=day))
        self.assertEqual(profile.trusted_means, means)
        self.assertEqual(profile.state, "FROZEN")
        with self.assertRaises(ValueError):
            profile.resume_after_review([self.vector(0.1), self.vector(0.2)])
        profile.resume_after_review([self.vector(0.1), self.vector(0.2)], approved=True)
        self.assertEqual(profile.state, "ACTIVE")

    def test_freeze_rolls_back_recent_updates(self):
        profile = ContaminationResistantProfile(quarantine_delay_days=1)
        profile.fit([self.vector(0.1), self.vector(0.2), self.vector(0.3)])
        original = dict(profile.trusted_means)
        profile.register_observation(self.vector(0.22), self.origin)
        profile.register_observation(self.vector(0.22), self.origin + timedelta(days=1))
        self.assertNotEqual(profile.trusted_means, original)
        profile.freeze_on_alert(self.origin + timedelta(days=1))
        self.assertEqual(profile.trusted_means, original)
        self.assertEqual(profile.admission_log[0].reason, "revoked_after_alert")

    def test_anchor_limits_accumulated_shift(self):
        profile = ContaminationResistantProfile(anchor_limit=0.5)
        profile.fit([self.vector(0.1), self.vector(0.2), self.vector(0.3)])
        for _ in range(1000):
            values = {m: value + 0.02 for m, value in profile.trusted_means.items()}
            profile._update_trusted_baseline(values)
        for marker in profile.TRACKED_MARKERS:
            self.assertLessEqual(abs(profile.trusted_means[marker] - profile.anchor_means[marker]), 0.050001)

    def test_profile_defers_freeze_until_sequential_confirmation(self):
        profile = ContaminationResistantProfile()
        profile.fit([self.vector(0.1), self.vector(0.2), self.vector(0.3)])
        _, _, alert = profile.register_observation(self.vector(2), self.origin, defer_freeze=True)
        self.assertTrue(alert)
        self.assertEqual(profile.state, "ACTIVE")
        profile.freeze_on_alert(self.origin)
        self.assertEqual(profile.state, "FROZEN")

    def test_role_mixture_variance(self):
        priors = {marker: {"mean": 1.0, "variance": 4.0} for marker in MarkerVector.feature_names()}
        profile = ContaminationResistantProfile(role_priors=priors, shrinkage_tau=2)
        profile.fit([self.vector(0), self.vector(2)])
        mean, std = profile.get_mixed_parameters("pronoun_marker")
        self.assertAlmostEqual(mean, 1)
        self.assertAlmostEqual(std ** 2, 3)

    def test_current_score_cannot_calibrate_itself(self):
        manager = SequentialDecisionManager()
        with self.assertRaises(ValueError):
            manager.process_observation(5.0, True, is_clean_mature=True)
        self.assertEqual(len(manager.martingale_detector.calibration_scores), 0)

    def test_cold_start_requires_independent_evidence(self):
        with self.assertRaises(ValueError):
            ContaminationResistantProfile().register_observation(self.vector(0.2), self.origin)

    def test_quarantine_preserves_exact_event_identity(self):
        buffer = QuarantineBuffer()
        for identifier in ("event_a", "event_b"):
            buffer.enqueue(self.origin, self.vector(0.1), {}, 0.5, identifier)
        approved, _ = buffer.extract_mature_clean_observations(self.origin + timedelta(days=14))
        self.assertEqual([observation.observation_id for observation in approved], ["event_a", "event_b"])

    def test_repeated_alarm_edges_and_freeze(self):
        manager = SequentialDecisionManager(k=1, n=1, use_conformal=False)
        self.assertTrue(manager.process_observation(5, True)[0])
        self.assertFalse(manager.process_observation(5, True)[0])
        manager.process_observation(0, False)
        self.assertTrue(manager.process_observation(5, True)[0])
        manager.freeze()
        self.assertFalse(manager.process_observation(5, True)[0])
        manager.resume_after_review()
        self.assertTrue(manager.process_observation(5, True)[0])

    def test_reply_latency_uses_incoming_message(self):
        window = SlidingTimeWindow("u", email="u@dtaa.com")
        window.add_message(CommunicationMessage("peer@dtaa.com", ["u@dtaa.com"], self.origin, subject="Task", body="hello"))
        window.add_message(CommunicationMessage("u@dtaa.com", ["peer@dtaa.com"], self.origin + timedelta(hours=2),
                                               sender_user_id="u", subject="Re: Task", body="reply"))
        self.assertEqual(window.get_reply_statistics()[0], 2)
        window.add_message(CommunicationMessage("u@dtaa.com", ["peer@dtaa.com"], self.origin + timedelta(hours=3),
                                               sender_user_id="u", body="unrelated"))
        self.assertEqual(window.get_reply_statistics()[0], 2)

    def test_graph_does_not_overwrite_betweenness(self):
        graph = DynamicCommunicationGraph()
        graph.add_communication(CommunicationMessage("u", ["external@example.org"], self.origin, body="hello"))
        context = graph.update_context(EmployeeContext("u"))
        self.assertEqual(context.betweenness, 0)
        self.assertEqual(context.external_ratio, 1)

    def test_workload_excludes_target_and_future(self):
        tracker = TeamWorkloadContextTracker()
        for day in range(30):
            for user in ("a", "b", "target"):
                tracker.register_role_activity("role", self.origin + timedelta(days=day),
                                               intensity=100 if user == "target" and day >= 23 else 1, employee_id=user)
        factor = tracker.get_role_workload_factor("role", self.origin + timedelta(days=30), exclude_user="target")
        self.assertAlmostEqual(factor, 0)
        self.assertEqual(tracker.get_role_workload_factor("role", self.origin - timedelta(days=1)), 0)

    def test_telemetry_retains_history_and_excludes_future(self):
        verifier = TargetedRetrospectiveVerifier()
        past = TelemetryEvent("past", "u", self.origin, "device_connect", "pc")
        future = TelemetryEvent("future", "u", self.origin + timedelta(days=90), "device_connect", "pc")
        verifier.register_telemetry_event(past)
        verifier.register_telemetry_event(future)
        captured = {}
        class Capture:
            def evaluate_mismatch(self, **kwargs):
                captured.update(kwargs)
                return None
        verifier.mismatch_detector = Capture()
        verifier.verify_alert_signal(RiskSignal("u", self.origin + timedelta(days=1), self.vector(0), {}))
        self.assertEqual([event.event_id for event in captured["telemetry_events"]], ["past"])

    def test_metrics_handle_ties_missing_values_and_alert_episodes(self):
        area, recall = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr(np.array([0, 1]), np.array([1, 1]))
        self.assertAlmostEqual(area, 0.5)
        self.assertEqual(recall, 0)
        area, recall = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr(np.array([0, 0]), np.array([0, 1]))
        self.assertTrue(np.isnan(area))
        self.assertTrue(np.isnan(recall))
        self.assertTrue(np.isnan(BenchmarkEvaluator.calculate_alarm_lead_time(None, self.origin)))
        rate = BenchmarkEvaluator.calculate_false_alarms_per_1000_days(np.zeros(4), np.array([2, 2, 0, 2]), 1, 2)
        self.assertEqual(rate, 1000)

    def test_ground_truth_uses_event_ids_and_release(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "r4.2-1").mkdir()
            (root / "insiders.csv").write_text("dataset,scenario,details,user,start,end\n4.2,1,u.csv,U,01/01/2020 00:00:00,01/03/2020 00:00:00\n", encoding="utf-8")
            (root / "r4.2-1/u.csv").write_text("email,e1,01/01/2020 00:00:00,U\nfile,f1,01/02/2020 00:00:00,U\n", encoding="utf-8")
            truth = load_cert_ground_truth(root / "insiders.csv")["u"]
            self.assertEqual(truth.first_action, self.origin + timedelta(days=1))
            message = CommunicationMessage("u", ["v"], self.origin, body="hello", message_id="other")
            self.assertEqual(truth.label(message), -1)
            message.message_id = "e1"
            self.assertEqual(truth.label(message), 1)

    def test_benchmark_five_methods_and_independent_splits(self):
        arguments = DriftInjector().generate_feature_scenario("step", duration_days=60, seed=21)
        rows, traces = BaselineBenchmarkRunner.evaluate_stream(*arguments)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0].admitted_total, 0)
        self.assertEqual(rows[1].admitted_total, len(arguments[2]))
        self.assertEqual(len(traces["Protected"].scores), len(arguments[2]))
        with self.assertRaises(ValueError):
            BaselineBenchmarkRunner.evaluate_stream(arguments[0], arguments[0], arguments[2])

    def test_synthetic_generation_is_reproducible(self):
        first = DriftInjector().generate_feature_scenario("ramp", seed=21)[2]
        second = DriftInjector().generate_feature_scenario("ramp", seed=21)[2]
        np.testing.assert_array_equal([o.markers.to_numpy() for o in first], [o.markers.to_numpy() for o in second])


if __name__ == "__main__":
    unittest.main()
