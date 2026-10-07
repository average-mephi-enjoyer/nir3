import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from core.models import CommunicationMessage, TelemetryEvent


@dataclass
class MismatchAnalysisResult:
    """
    Verification result of semantic-technical mismatch.
    """
    user_id: str
    timestamp: datetime
    stated_projects: List[str]
    accessed_resources: List[str]
    unrelated_resources: List[str]
    mismatch_ratio: float
    has_usb_exfiltration: bool
    is_suspicious_mismatch: bool
    explanation: str


class SemanticTechnicalMismatchDetector:
    """
    Detector of semantic-technical mismatch.
    """
    PROJECT_KEYWORD_REGEX = re.compile(
        r"\b(?:project|task|working on|ref|deal|contract|team)\s+([A-Za-z0-9_\-]+)\b",
        re.IGNORECASE
    )

    def __init__(self, mismatch_threshold: float = 0.5):
        self.mismatch_threshold = mismatch_threshold

    def extract_stated_topics(self, messages: List[CommunicationMessage]) -> Set[str]:
        """
        Extraction of topics from the sliding window.
        """
        topics = set()
        for msg in messages:
            content = f"{msg.subject or ''} {msg.cleaned_body or msg.body or ''}"
            matches = self.PROJECT_KEYWORD_REGEX.findall(content)
            for m in matches:
                clean_m = m.lower().strip()
                if len(clean_m) > 2:
                    topics.add(clean_m)
        return topics

    def evaluate_mismatch(
        self,
        user_id: str,
        messages: List[CommunicationMessage],
        telemetry_events: List[TelemetryEvent],
        current_time: datetime
    ) -> MismatchAnalysisResult:
        """
        Evaluation of the semantic-technical mismatch for a user based on their communication and telemetry events.
        """
        stated_topics = self.extract_stated_topics(messages)
        file_events = [e for e in telemetry_events if e.event_type == "file_transfer"]
        usb_events = [e for e in telemetry_events if "device_connect" in e.event_type]
        has_usb = len(usb_events) > 0
        accessed_resources = []
        unrelated_resources = []
        for fe in file_events:
            target = fe.target_resource or ""
            if not target:
                continue
            accessed_resources.append(target)
            target_lower = target.lower()
            if stated_topics:
                matches_any_topic = any(topic in target_lower for topic in stated_topics)
                if not matches_any_topic:
                    unrelated_resources.append(target)
            else:
                unrelated_resources.append(target)
        total_files = len(accessed_resources)
        if total_files > 0:
            mismatch_ratio = len(unrelated_resources) / total_files
        else:
            mismatch_ratio = 0.0
        is_suspicious = (
            (mismatch_ratio >= self.mismatch_threshold and total_files >= 3)
            or (has_usb and total_files > 0)
        )
        explanations = []
        if has_usb:
            explanations.append(f"Applying of external USB device detected ({len(usb_events)} times)")
        if stated_topics:
            explanations.append(f"Declared projects in communication: {', '.join(stated_topics)}")
        if unrelated_resources:
            explanations.append(
                f"Unintended accesses to files: {len(unrelated_resources)} out of {total_files} "
                f"(mismatch coefficient: {mismatch_ratio:.2f})"
            )
        full_exp = "; ".join(explanations) if explanations else "Technical activity is consistent with communication."
        return MismatchAnalysisResult(
            user_id=user_id,
            timestamp=current_time,
            stated_projects=list(stated_topics),
            accessed_resources=accessed_resources,
            unrelated_resources=unrelated_resources,
            mismatch_ratio=float(mismatch_ratio),
            has_usb_exfiltration=has_usb,
            is_suspicious_mismatch=is_suspicious,
            explanation=full_exp
        )