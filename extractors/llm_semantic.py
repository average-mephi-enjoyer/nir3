import logging
from typing import Any, Dict, Optional
import httpx
from core.models import CommunicationMessage, EmployeeContext, MarkerVector
from core.schemas import LLMSemanticAnalysisResponse
from extractors.base import IFeatureExtractor
import numpy as np

logger = logging.getLogger(__name__)


class LLMSemanticExtractor(IFeatureExtractor):
    """
    Extractor of the semantic markers.
    """

    SYSTEM_PROMPT = (
        "You are an objective psycholinguistic analysis module for corporate communications.\n"
        "Your task is to analyze text strictly against an established Insider Language Index (ILI) ontology.\n"
        "Do NOT output verdicts on employee guilt or accuse anyone. Extract only observable semantic risk markers with evidence.\n\n"
        "Recognized markers:\n"
        "- organizational_alienation: severe detachment, cynical alienation, malicious rejection of company norms.\n"
        "- concealment_intent: explicit intent to bypass logging, clear tracks, use unauthorized channels, or hide actions.\n"
        "- rationalization: cognitive justification of ethical violations, entitlement, perceived unfairness as pretext.\n"
        "- access_exfiltration_focus: abnormal interest in confidential assets, sensitive access beyond job role, data staging.\n"
        "- conflict_escalation: destructive hostility, aggressive blaming, personal threats, obstructionism.\n\n"
        "Output requirements:\n"
        "Return ONLY a valid JSON object matching the provided JSON Schema. Every marker MUST have an exact 'evidence_span' quote from text."
    )

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:11434/v1/chat/completions",
        model_name: str = "qwen2.5:7b-instruct",
        timeout_seconds: float = 10.0,
        max_retries: int = 2,
        client: Optional[httpx.Client] = None
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._external_client = client is not None
        self._client = client or httpx.Client(timeout=self.timeout_seconds)
        self.schema_dict = LLMSemanticAnalysisResponse.model_json_schema()

    def close(self) -> None:
        if not self._external_client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _build_payload(self, text: str) -> Dict[str, Any]:
        """
        forming payload, compatible to OpenAI Structured Outputs / vLLM / llama.cpp / Ollama.
        """
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze communication message body:\n\"\"\"\n{text}\n\"\"\""}
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_risk_response",
                    "strict": True,
                    "schema": self.schema_dict
                }
            }
        }

    def _call_api(self, text: str) -> Optional[LLMSemanticAnalysisResponse]:
        payload = self._build_payload(text)
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self.endpoint_url, json=payload)
                if response.status_code == 200:
                    raw_data = response.json()
                    content = raw_data["choices"][0]["message"]["content"]
                    return LLMSemanticAnalysisResponse.model_validate_json(content)
                logger.warning("API error HTTP %s: %s", response.status_code, response.text)
            except Exception as e:
                logger.debug("Attempt %d failed with error: %s", attempt + 1, str(e))
                if attempt == self.max_retries:
                    logger.error("Failed LLM extraction after %d attempts", self.max_retries + 1)
                    return None
        return None

    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        text = message.text_content
        vector = MarkerVector()
        vector.llm_uncertainty = 1.0
        if not text or len(text) < 15:
            return vector
        parsed_resp = self._call_api(text)
        if not parsed_resp or parsed_resp.has_insufficient_context:
            return vector
        scores: Dict[str, tuple[float, float]] = {}
        for item in parsed_resp.markers:
            name = item.marker_name
            if name not in scores or item.score > scores[name][0]:
                scores[name] = (item.score, item.confidence)
        vector.llm_alienation = scores.get("organizational_alienation", (0.0, 1.0))[0]
        vector.llm_concealment = scores.get("concealment_intent", (0.0, 1.0))[0]
        vector.llm_rationalization = scores.get("rationalization", (0.0, 1.0))[0]
        vector.llm_access_focus = scores.get("access_exfiltration_focus", (0.0, 1.0))[0]
        vector.llm_conflict = scores.get("conflict_escalation", (0.0, 1.0))[0]
        if scores:
            mean_conf = sum(conf for _, conf in scores.values()) / len(scores)
            vector.llm_uncertainty = float(np.clip(1.0 - mean_conf, 0.0, 1.0))
        else:
            vector.llm_uncertainty = 0.0
        return vector