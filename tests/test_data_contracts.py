from datetime import datetime, timedelta, timezone
import unittest
from core.models import CommunicationMessage, MarkerVector
from extractors.graph_topology import DynamicCommunicationGraph
from extractors.spacy_statistical import SpacyStatisticalExtractor
from profiling.sliding_window import SlidingTimeWindow
from profiling.profile import ContaminationResistantProfile
from parsers.cert_ground_truth import IncidentTruth, UserIncidents


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.t = datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_exact_identity_and_alias_change(self):
        graph = DynamicCommunicationGraph()
        graph.register_alias("user", "old@dtaa.com")
        graph.register_alias("user", "new@dtaa.com")
        for sender in ("old@dtaa.com", "new@dtaa.com", "superuser"):
            graph.add_communication(CommunicationMessage(sender, ["peer"], self.t, body="hello"))
        self.assertEqual(graph.compute_advanced_topology("user")[1], 2)
        self.assertEqual(graph._resolve_candidate_nodes("use"), [])

    def test_missing_history_and_response_are_unavailable(self):
        window = SlidingTimeWindow("u")
        msg = CommunicationMessage("u", ["v"], self.t, body="hello")
        window.add_message(msg)
        context = window.export_context()
        self.assertFalse(context.activity_available["quarterly"])
        vector = SpacyStatisticalExtractor().compute_markers(msg, context)
        self.assertIn("response_marker", vector.unavailable)
        self.assertIn("activity_marker", vector.unavailable)

    def test_mask_does_not_dilute_risk_or_update_missing_feature(self):
        missing = set(MarkerVector.feature_names()) - {"pronoun_marker"}
        p = ContaminationResistantProfile(protect_anchor=False)
        p.fit([MarkerVector(pronoun_marker=x, unavailable=missing) for x in (0, 2)])
        z, risk, _ = p.compute_deviations(MarkerVector(pronoun_marker=3, unavailable=missing))
        self.assertEqual(len(z), 1)
        self.assertAlmostEqual(risk, 2 ** .5)
        p._update_trusted_baseline({"pronoun_marker": 1})
        self.assertEqual(p.trusted_means["response_marker"], 0)

    def test_multiple_scenarios_preserve_unknown_labels(self):
        truth = UserIncidents([IncidentTruth("u", str(i), self.t + timedelta(days=i), self.t + timedelta(days=i+1), email_ids={str(i)}) for i in (0, 5)])
        message = CommunicationMessage("u", ["v"], self.t, message_id="other")
        self.assertEqual(truth.label(message), -1)
        message.message_id = "5"
        self.assertEqual(truth.label(message), 1)
        with self.assertRaises(ValueError):
            _ = truth.start
