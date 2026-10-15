import logging
import math
import re
from typing import Any, Dict, Optional, Set
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

    TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")

    ALIENATION_STEMS = {
        "unfair", "disregard", "hate", "unjust", "toxic", "management",
        "quit", "leave", "fired", "stupid", "worthless", "incompetent", "moron", "useless"
    }
    CONCEALMENT_STEMS = {
        "delete", "audit", "wipe", "usb", "drive", "private", "bypass",
        "hide", "tracks", "clean", "erase", "scrub", "unmonitored", "covert"
    }
    ACCESS_STEMS = {
        "confidential", "financial", "secret", "database", "export", "archive",
        "tar.gz", "restricted", "dump", "records", "credentials", "password", "backup"
    }
    RATIONALIZATION_STEMS = {
        "deserve", "owed", "underpaid", "credit", "entitled", "compensation",
        "cheat", "fair share", "earned", "owe", "underappreciated", "repaid"
    }
    CONFLICT_STEMS = {
        "threat", "sue", "lawyer", "destroy", "retaliate", "fight", "revenge",
        "blame", "fault", "idiot", "hostile", "sabotage"
    }

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
            return None
        return None

    def _smooth_density_transfer(self, hits: int, total_words: int, gain: float = 25.0) -> float:
        if hits == 0 or total_words == 0:
            return 0.0
        density = hits / float(total_words + 5)
        continuous_score = (2.0 / (1.0 + math.exp(-gain * density))) - 1.0
        return float(np.clip(continuous_score, 0.0, 1.0))

    def _count_stem_matches(self, tokens: list[str], stems: Set[str]) -> int:
        count = 0
        for token in tokens:
            if any(stem in token for stem in stems):
                count += 1
        return count

    def _heuristic_ili_extraction(self, text: str) -> MarkerVector:
        tokens = [t.lower() for t in self.TOKEN_PATTERN.findall(text)]
        total_words = len(tokens)
        vec = MarkerVector()
        if total_words == 0:
            return vec
        alien_hits = self._count_stem_matches(tokens, self.ALIENATION_STEMS)
        conceal_hits = self._count_stem_matches(tokens, self.CONCEALMENT_STEMS)
        access_hits = self._count_stem_matches(tokens, self.ACCESS_STEMS)
        ratio_hits = self._count_stem_matches(tokens, self.RATIONALIZATION_STEMS)
        conflict_hits = self._count_stem_matches(tokens, self.CONFLICT_STEMS)
        vec.llm_alienation = self._smooth_density_transfer(alien_hits, total_words, gain=22.0)
        vec.llm_concealment = self._smooth_density_transfer(conceal_hits, total_words, gain=25.0)
        vec.llm_access_focus = self._smooth_density_transfer(access_hits, total_words, gain=20.0)
        vec.llm_rationalization = self._smooth_density_transfer(ratio_hits, total_words, gain=25.0)
        vec.llm_conflict = self._smooth_density_transfer(conflict_hits, total_words, gain=22.0)
        total_signals = alien_hits + conceal_hits + access_hits + ratio_hits + conflict_hits
        vec.llm_uncertainty = float(np.clip(1.0 - (total_signals / float(total_words + 1)) * 5.0, 0.1, 0.9))
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
