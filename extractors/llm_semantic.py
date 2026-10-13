import logging
import re
from typing import Any, Dict, Optional
import httpx
import numpy as np
from core.models import CommunicationMessage, EmployeeContext, MarkerVector
from core.schemas import LLMSemanticAnalysisResponse
from extractors.base import IFeatureExtractor

logger = logging.getLogger(__name__)


class TextSanitizer:
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
            sanitized = pattern.sub("[STRIPPED_INJECTION]", sanitized)
        return sanitized.strip()


class LLMSemanticExtractor(IFeatureExtractor):
    """
    Semantic extractor utilizing structured JSON schema outputs from local LLM,
    or fast deterministic ILI ontology rules if server is offline.
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

    ALIENATION_KEYWORDS = {"unfair", "disregard", "hate", "unjust", "toxic", "management", "quit", "leave", "fired", "stupid"}
    CONCEALMENT_KEYWORDS = {"delete", "audit", "wipe", "usb", "drive", "private", "bypass", "hide", "tracks", "clean"}
    ACCESS_KEYWORDS = {"confidential", "financial", "secret", "database", "export", "archive", "tar.gz", "restricted", "dump", "records"}

    def __init__(
        self,
        endpoint_url: str = "http://127.0.0.1:11434/v1/chat/completions",
        model_name: str = "qwen2.5:7b-instruct",
        timeout_seconds: float = 1.0,
        max_retries: int = 0,
        force_offline_mode: bool = False
    ):
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.force_offline_mode = force_offline_mode
        self._server_available = not force_offline_mode
        self._client = httpx.Client(timeout=self.timeout_seconds)
        self.schema_dict = LLMSemanticAnalysisResponse.model_json_schema()

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _build_payload(self, text: str) -> Dict[str, Any]:
        sanitized_text = TextSanitizer.sanitize(text)
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze communication message body:\n\"\"\"\n{sanitized_text}\n\"\"\""}
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
        if not self._server_available:
            return None
        payload = self._build_payload(text)
        try:
            response = self._client.post(self.endpoint_url, json=payload)
            if response.status_code == 200:
                raw_data = response.json()
                content = raw_data["choices"][0]["message"]["content"]
                return LLMSemanticAnalysisResponse.model_validate_json(content)
        except Exception:
            self._server_available = False
            logger.info("Local LLM server offline. Switched to high-throughput ILI ontological analyzer.")
            return None
        return None

    def _heuristic_ili_extraction(self, text: str) -> MarkerVector:
        t_low = text.lower()
        vec = MarkerVector()
        alien_hits = sum(1 for w in self.ALIENATION_KEYWORDS if w in t_low)
        conceal_hits = sum(1 for w in self.CONCEALMENT_KEYWORDS if w in t_low)
        access_hits = sum(1 for w in self.ACCESS_KEYWORDS if w in t_low)
        vec.llm_alienation = float(np.clip(alien_hits * 0.35, 0.0, 1.0))
        vec.llm_concealment = float(np.clip(conceal_hits * 0.40, 0.0, 1.0))
        vec.llm_access_focus = float(np.clip(access_hits * 0.45, 0.0, 1.0))
        vec.llm_conflict = float(np.clip((alien_hits + conceal_hits) * 0.20, 0.0, 1.0))
        vec.llm_uncertainty = 0.2 if (alien_hits + conceal_hits + access_hits > 0) else 0.8
        return vec

    def compute_markers(self, message: CommunicationMessage, context: Optional[EmployeeContext] = None) -> MarkerVector:
        text = message.text_content
        if not text or len(text) < 5:
            return MarkerVector()
        parsed_resp = self._call_api(text)
        if not parsed_resp:
            return self._heuristic_ili_extraction(text)
        scores: Dict[str, tuple[float, float]] = {}
        for item in parsed_resp.markers:
            name = item.marker_name
            if name not in scores or item.score > scores[name][0]:
                scores[name] = (item.score, item.confidence)
        vector = MarkerVector()
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
