import unittest
from tempfile import TemporaryDirectory
from benchmark.cert_pipeline import stratified_users, validate_archive, independent_prior
from core.models import MarkerVector


class CertPipelineTests(unittest.TestCase):
    def test_missing_archive_is_explicit(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "CERT r4.2 unavailable"):
                validate_archive(directory)

    def test_reference_selection_reproducible_and_role_stratified(self):
        roster = {str(i): {"role": str(i % 3)} for i in range(30)}
        first = stratified_users(roster, 2, 42)
        self.assertEqual(first, stratified_users(dict(reversed(list(roster.items()))), 2, 42))
        self.assertEqual([len(v) for v in first.values()], [2, 2, 2])

    def test_prior_excludes_target_and_missing_markers(self):
        vectors = [MarkerVector(pronoun_marker=v, unavailable={"response_marker"}) for v in (0, 1)]
        peers = {("r", "target"): [MarkerVector(pronoun_marker=100)] * 2,
                 ("r", "a"): vectors, ("r", "b"): vectors}
        prior, info = independent_prior(peers, "r", "target")
        self.assertEqual(prior["pronoun_marker"]["mean"], .5)
        self.assertNotIn("response_marker", prior)
        self.assertEqual(info["users"], ["a", "b"])
        self.assertEqual(independent_prior(peers, "unknown", "target")[0], {})
