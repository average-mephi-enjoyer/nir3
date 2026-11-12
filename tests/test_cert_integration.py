import os
import unittest
from pathlib import Path
from parsers.cert_ground_truth import load_cert_ground_truth

ARCHIVE = Path(os.environ.get("CERT_ARCHIVE", "archive"))


@unittest.skipUnless((ARCHIVE / "answers/insiders.csv").exists(), "CERT not installed; synthetic suite remains runnable")
class CertIntegration(unittest.TestCase):
    def test_real_answer_key_structure_and_intervals(self):
        truth = load_cert_ground_truth(ARCHIVE / "answers/insiders.csv")
        self.assertTrue(truth)
        for user, incidents in truth.items():
            for incident in incidents:
                self.assertEqual(user, incident.user)
                self.assertLessEqual(incident.start, incident.end)
                self.assertTrue(incident.details)
                if incident.first_action:
                    self.assertTrue(incident.first_action_id)
                    self.assertIn(incident.first_action_type, {"file", "device", "http", "logon"})
