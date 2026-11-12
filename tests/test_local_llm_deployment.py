import json
import unittest
import httpx
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from core.local_llm import load_config, verify_deployment, contained_path, CONFIG_PATH, file_sha256
from core.research_acceptance import llm_prerequisite
from extractors.llm_semantic import LLMSemanticExtractor


class LocalDeploymentTests(unittest.TestCase):
    def test_pinned_model_and_cpu_runtime(self):
        config = load_config()
        self.assertEqual(len(config["model_revision"]), 40)
        self.assertEqual(config["endpoint_url"], "http://127.0.0.1:8081/v1/chat/completions")
        self.assertEqual(sum(a["size"] for a in config["artifacts"] if a["kind"] == "model"), 4683073632)
        for artifact in config["artifacts"]:
            self.assertRegex(artifact["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("cpu-x64", config["artifacts"][0]["url"])

    def test_missing_and_tampered_deployment_fail_closed(self):
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "not installed"):
                verify_deployment(root)
            config = load_config()
            receipt = {"ready": True, "config_sha256": file_sha256(CONFIG_PATH),
                       "model_name": config["model_name"], "model_revision": config["model_revision"]}
            (Path(root) / "deployment.json").write_text(json.dumps(receipt), encoding="utf-8-sig")
            with self.assertRaisesRegex(RuntimeError, "artifact"):
                verify_deployment(root)
            with self.assertRaises(ValueError):
                contained_path(root, "../escape")

    def test_long_cpu_timeout_and_generation_config(self):
        with LLMSemanticExtractor(mode="local_llm") as extractor:
            self.assertGreaterEqual(extractor.timeout_seconds, 120)
            payload = extractor._build_payload("synthetic test")
            self.assertEqual(payload["seed"], load_config()["seed"])
            self.assertEqual(payload["max_tokens"], 768)
            self.assertEqual(payload["temperature"], 0)

    def test_wrong_model_or_truncated_response_rejected(self):
        with LLMSemanticExtractor(mode="local_llm") as extractor:
            response = Mock()
            extractor._client.post = Mock(return_value=response)
            for model, reason in (("wrong-model", "stop"), (extractor.model_name, "length")):
                response.json.return_value = {"model": model, "choices": [{"finish_reason": reason, "message": {"content": '{"markers":[]}'}}]}
                with self.assertRaises(RuntimeError):
                    extractor._call_api("synthetic text")

    def test_preflight_checks_actual_served_alias(self):
        with LLMSemanticExtractor(mode="local_llm") as extractor:
            response = Mock()
            response.json.return_value = {"data": [{"id": "wrong-model"}]}
            extractor._client.get = Mock(return_value=response)
            with patch("extractors.llm_semantic.verify_deployment", return_value={"verified": True}):
                with self.assertRaisesRegex(RuntimeError, "alias"):
                    extractor.verify_runtime()

    def test_heuristic_experiment_cannot_pass_llm_prerequisite(self):
        self.assertFalse(llm_prerequisite({"config": {"semantic_mode": "heuristic"}}, {})["llm_prerequisite_met"])
        provenance = {"weight_sha256": ["weight"], "receipt_sha256": "receipt"}
        manifest = {"config": {"semantic_mode": "local_llm", "semantic_runtime": provenance}}
        probe = {"status": "local_inference_observed", "samples": [{"metadata": {
            "mode": "local_llm", "model_reported": "qwen", "model_requested": "qwen", "deployment": provenance}}]}
        status = llm_prerequisite(manifest, probe)
        self.assertTrue(status["llm_prerequisite_met"])
        self.assertFalse(status["scientific_validity_established"])
        probe["samples"][0]["metadata"]["deployment"] = {"weight_sha256": ["other"]}
        self.assertFalse(llm_prerequisite(manifest, probe)["llm_prerequisite_met"])

    def test_conflicting_modes_are_not_silently_downgraded(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            LLMSemanticExtractor(mode="local_llm", force_offline_mode=True)

    def test_missing_server_has_explicit_start_instruction(self):
        with LLMSemanticExtractor(mode="local_llm") as extractor:
            extractor._client.get = Mock(side_effect=httpx.ConnectError("offline"))
            with patch("extractors.llm_semantic.verify_deployment", return_value={"verified": True}):
                with self.assertRaisesRegex(RuntimeError, "start_local_llm"):
                    extractor.verify_runtime()
