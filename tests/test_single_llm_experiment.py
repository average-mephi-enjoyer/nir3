import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from benchmark.baseline_comparison import FeatureObservation
from core.models import CommunicationMessage, MarkerVector
from core.schemas import LLMSemanticAnalysisResponse
from pydantic import ValidationError
from scripts.run_single_llm_experiment import SEMANTIC, cached_inference, merge_semantics, coverage


class SingleLLMExperimentTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.message = CommunicationMessage("u", ["v"], self.now, message_id="m", body="hide tracks")

    def test_merge_preserves_common_features_and_mask(self):
        base = MarkerVector(pronoun_marker=.3, social_marker=.4, unavailable=SEMANTIC | {"response_marker"})
        semantic = MarkerVector(llm_concealment=.7, semantic_metadata={"mode": "local_llm"})
        merged = merge_semantics(base, semantic)
        self.assertEqual(merged.pronoun_marker, .3)
        self.assertEqual(merged.social_marker, .4)
        self.assertEqual(merged.llm_concealment, .7)
        self.assertEqual(merged.unavailable, {"response_marker"})
        self.assertEqual(base.unavailable, SEMANTIC | {"response_marker"})
        missing = merge_semantics(base, MarkerVector(unavailable=SEMANTIC))
        self.assertTrue(SEMANTIC <= missing.unavailable)

    def test_cache_is_bound_to_text_message_and_protocol(self):
        with TemporaryDirectory() as directory:
            extractor = Mock()
            extractor.compute_markers.return_value = MarkerVector(llm_concealment=.7)
            cache = Path(directory)
            vector, cached = cached_inference(extractor, self.message, cache, "v1")
            self.assertFalse(cached)
            self.assertEqual(vector.llm_concealment, .7)
            self.assertTrue(cached_inference(extractor, self.message, cache, "v1")[1])
            self.assertFalse(cached_inference(extractor, replace(self.message, body="new text"), cache, "v1")[1])
            self.assertFalse(cached_inference(extractor, replace(self.message, message_id="m2"), cache, "v1")[1])
            self.assertFalse(cached_inference(extractor, self.message, cache, "v2")[1])
            self.assertEqual(extractor.compute_markers.call_count, 4)
            for path in cache.glob("*.json"):
                path.write_text(path.read_text().replace('"llm_concealment": 0.7', '"llm_concealment": 0.9'))
            with self.assertRaisesRegex(ValueError, "integrity"):
                cached_inference(extractor, self.message, cache, "v1")

    def test_invalid_output_is_missing_but_infrastructure_error_stops(self):
        def invalid(*args):
            try:
                LLMSemanticAnalysisResponse.model_validate_json("invalid")
            except ValidationError as cause:
                raise RuntimeError("invalid output") from cause
        with TemporaryDirectory() as directory:
            extractor = Mock(last_call={"failure": "ValidationError", "mode": "local_llm"})
            extractor.compute_markers.side_effect = invalid
            vector, _ = cached_inference(extractor, self.message, Path(directory), "v1")
            self.assertTrue(SEMANTIC <= vector.unavailable)
            self.assertEqual(vector.semantic_metadata["mode"], "local_llm")
            extractor.compute_markers.side_effect = RuntimeError("server offline")
            with self.assertRaisesRegex(RuntimeError, "offline"):
                cached_inference(extractor, self.message, Path(directory), "v2")

    def test_coverage_separates_gap_and_missing(self):
        observations = [FeatureObservation(self.now + timedelta(days=i), MarkerVector(unavailable=SEMANTIC if i == 2 else set())) for i in range(4)]
        boundaries = tuple(self.now + timedelta(days=i) for i in (1, 2, 3, 4))
        result = coverage(observations, boundaries)
        self.assertEqual([v["messages"] for v in result.values()], [1, 1, 1, 1])
        self.assertEqual(result["gap"]["semantic_available"], 0)
        self.assertEqual(result["test"]["semantic_available"], 1)
