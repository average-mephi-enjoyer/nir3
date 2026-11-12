from datetime import datetime, timedelta, timezone
import unittest
from core.models import MarkerVector
from profiling.quarantine import QuarantineBuffer
from profiling.profile import ContaminationResistantProfile
from profiling.sequential_decision import SequentialDecisionManager


def vector(value):
    return MarkerVector(**dict.fromkeys(MarkerVector.feature_names(), value))


class ProfileLifecycle(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_quarantine_duplicate_future_and_threshold_snapshot(self):
        q = QuarantineBuffer(1, 2)
        q.enqueue(self.t, vector(0), {}, 1, "a")
        with self.assertRaises(ValueError):
            q.enqueue(self.t, vector(0), {}, 0, "a")
        q.h_quarantine = 100
        approved, _ = q.extract_mature_clean_observations(self.t + timedelta(days=1))
        self.assertEqual(approved[0].threshold, 2)
        self.assertEqual(approved[0].normalized_risk, .5)
        self.assertEqual(approved[0].status, "ADMITTED")
        self.assertFalse(approved[0].included_in_reference)
        q.enqueue(self.t + timedelta(days=2), vector(0), {}, 0, "b")
        with self.assertRaises(ValueError):
            q.extract_mature_clean_observations(self.t)

    def test_review_restarts_both_without_changing_anchor(self):
        p = ContaminationResistantProfile(quarantine_delay_days=1)
        training = [vector(x) for x in (0, 1, 2)]
        p.fit(training)
        anchor = dict(p.anchor_means)
        detector = SequentialDecisionManager(k=1, n=1)
        self.assertTrue(detector.process_observation(10, True)[0])
        detector.freeze_profile(p, self.t)
        self.assertFalse(detector.process_observation(10, True)[0])
        detector.review_profile(p, training, [1.] * 10, self.t + timedelta(days=1), approved=True)
        self.assertEqual(p.state, "ACTIVE")
        self.assertEqual(p.anchor_means, anchor)
        self.assertTrue(detector.process_observation(10, True)[0])
        self.assertEqual([item["state"] for item in p.state_history], ["ALERT_PENDING_REVIEW", "FROZEN", "RESUMING", "ACTIVE"])

    def test_anchor_changes_updates_but_not_scoring_rule(self):
        profiles = [ContaminationResistantProfile(protect_anchor=flag, anchor_limit=.01) for flag in (True, False)]
        for p in profiles:
            p.fit([vector(0), vector(2)])
        self.assertEqual(profiles[0].compute_deviations(vector(3)), profiles[1].compute_deviations(vector(3)))
        self.assertFalse(profiles[0]._update_trusted_baseline(dict.fromkeys(MarkerVector.feature_names(), 10)))
        self.assertTrue(profiles[1]._update_trusted_baseline(dict.fromkeys(MarkerVector.feature_names(), 10)))

    def test_deadline_never_raises_concealment_threshold(self):
        p = ContaminationResistantProfile()
        p.fit([vector(0), vector(2)])
        marker = MarkerVector(llm_concealment=10)
        self.assertEqual(p.compute_deviations(marker, 0)[0]["llm_concealment"], p.compute_deviations(marker, 1)[0]["llm_concealment"])
        self.assertEqual(p.compute_dynamic_thresholds(0), p.compute_dynamic_thresholds(1))

    def test_role_switch_requires_confirmation(self):
        p = ContaminationResistantProfile()
        prior = {"pronoun_marker": {"mean": 1, "variance": 2}}
        with self.assertRaises(ValueError):
            p.switch_role_prior("engineer", prior, self.t)
        p.switch_role_prior("engineer", prior, self.t, confirmed=True)
        self.assertEqual(p.role, "engineer")

    def test_review_revokes_old_reference_and_is_transactional(self):
        p = ContaminationResistantProfile(quarantine_delay_days=1)
        training = [vector(x) for x in (0, 1, 2)]
        p.fit(training)
        p.register_observation(vector(1), self.t, observation_id="old")
        p.register_observation(vector(1), self.t + timedelta(days=1), observation_id="next")
        p.freeze_on_alert(self.t + timedelta(days=20))
        self.assertEqual(p.admission_log[0].status, "ADMITTED")
        before = dict(p.trusted_means)
        missing = MarkerVector(unavailable=set(MarkerVector.feature_names()))
        with self.assertRaises(ValueError):
            p.resume_after_review([missing, missing], approved=True)
        self.assertEqual(p.state, "FROZEN")
        self.assertEqual(p.trusted_means, before)
        p.resume_after_review(training, approved=True, timestamp=self.t + timedelta(days=21))
        self.assertEqual(p.admission_log[0].status, "REVOKED")
        self.assertFalse(p.admission_log[0].included_in_reference)

    def test_prior_only_cold_start_never_reanchors_on_unreviewed_candidates(self):
        priors = {m: {"mean": 0., "variance": 1.} for m in MarkerVector.feature_names()}
        p = ContaminationResistantProfile(role_priors=priors, quarantine_delay_days=0)
        original = dict(p.anchor_means)
        for day in range(40):
            p.register_observation(vector(.1), self.t + timedelta(days=day))
        self.assertEqual(p.state, "COLD")
        self.assertEqual(p.anchor_means, original)
        self.assertFalse(p.admission_log)
