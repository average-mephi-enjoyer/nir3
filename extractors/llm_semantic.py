import json
import re
from typing import Dict, List, Optional, Tuple, Any
import httpx

from core.models import CommunicationMessage, EmployeeContext, MarkerVector
from core.schemas import LLMSemanticAnalysisResponse
from extractors.base import IFeatureExtractor


class TextSanitizer:
    """
    Pre-masking module to defend from prompt injection.
    """
    EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior)\s+instructions", re.IGNORECASE),
        re.compile(r"system\s+prompt", re.IGNORECASE),
        re.compile(r"you\s+are\s+now", re.IGNORECASE),
        re.compile(r"disregard\s+rules", re.IGNORECASE),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        if not text:
            return ""
        sanitized = cls.EMAIL_REGEX.sub("[EMAIL_MASKED]", text)
        sanitized = cls.IP_REGEX.sub("[IP_MASKED]", sanitized)
        for pattern in cls.INJECTION_PATTERNS:
            sanitized = pattern.sub("[SUSPICIOUS_PROMPT_INJECTION_STRIPPED]", sanitized)
        return sanitized.strip()


class LLMSemanticExtractor(IFeatureExtractor):
    """
    Local semantic extractor based on OpenAI-compatible API (vLLM, Ollama, llama.cpp).
    Deterministic sampling (T=0), JSON schema validation, evidence_span and fallback.
    """

    SYSTEM_PROMPT = """You are an objective psycholinguistic analysis module for corporate communications.
Your task is to analyze text strictly against an established Insider Language Index (ILI) ontology.
Do NOT output verdicts on employee guilt or accuse anyone. Extract only observable semantic risk markers with evidence.

Recognized markers:
- organizational_alienation: severe detachment, cynical alienation, malicious rejection of company norms.
- concealment_intent: explicit intent to bypass logging, clear tracks, use unauthorized channels, or hide actions.
- rationalization: cognitive justification of ethical violations, entitlement, perceived unfairness as pretext.
- access_exfiltration_focus: abnormal interest in confidential assets, sensitive access beyond job role, data staging.
- conflict_escalation: destructive hostility, aggressive blaming, personal threats, obstructionism.

Output requirements:
Return ONLY a valid JSON object matching the provided JSON Schema. Every marker MUST have an exact 'evidence_span' quote from text."""

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:11434/v1/chat/completions",
        model_name: str = "qwen2.5:7b-instruct",
        timeout_seconds: float = 5.0,
        max_retries: int = 2
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.schema_dict = LLMSemanticAnalysisResponse.model_json_schema()

    def _build_payload(self, text: str) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze communication message body:\n\"\"\"\n{text}\n\"\"\""}
            ],
            "temperature": 0.0,
            "response_format": {
                "type": "json_object",
                "schema": self.schema_dict
            }
        }

    def _call_api_sync(self, sanitized_text: str) -> Optional[LLMSemanticAnalysisResponse]:
        payload = self._build_payload(sanitized_text)
        with httpx.Client(timeout=self.timeout_seconds) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(self.endpoint_url, json=payload)
                    if response.status_code == 200:
                        raw_data = response.json()
                        content = raw_data["choices"][0]["message"]["content"]
                        return LLMSemanticAnalysisResponse.model_validate_json(content)
                except Exception:
                    if attempt == self.max_retries:
                        return None
        return None

    def compute_markers(self, message: CommunicationMessage, context: EmployeeContext) -> MarkerVector:
        text = message.cleaned_body or message.body or ""
        sanitized_text = TextSanitizer.sanitize(text)
        vector = MarkerVector()
        vector.llm_uncertainty = 1.0
        if not sanitized_text or len(sanitized_text) < 10:
            return vector
        parsed_resp = self._call_api_sync(sanitized_text)
        if not parsed_resp:
            return vector
        vector.llm_uncertainty = 0.0
        scores: Dict[str, Tuple[float, float]] = {}
        for item in parsed_resp.markers:
            if item.marker_type not in scores or item.score > scores[item.marker_type][0]:
                scores[item.marker_type] = (item.score, item.uncertainty)
        vector.llm_alienation = scores.get("organizational_alienation", (0.0, 0.0))[0]
        vector.llm_concealment = scores.get("concealment_intent", (0.0, 0.0))[0]
        vector.llm_rationalization = scores.get("rationalization", (0.0, 0.0))[0]
        vector.llm_access_focus = scores.get("access_exfiltration_focus", (0.0, 0.0))[0]
        vector.llm_conflict = scores.get("conflict_escalation", (0.0, 0.0))[0]
        if scores:
            mean_unc = sum(u for _, u in scores.values()) / len(scores)
            vector.llm_uncertainty = float(mean_unc)
        return vector