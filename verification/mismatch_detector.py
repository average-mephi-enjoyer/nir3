from dataclasses import dataclass
from datetime import datetime
import re
from typing import List, Set
from core.models import CommunicationMessage, TelemetryEvent


@dataclass(slots=True)
class MismatchAnalysisResult:
    """
    Verification result of semantic-technical mismatch analysis.
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
    Evaluates discrepancy between declared intentions in communication channels
    and observable digital footprints in host/network telemetry.
    """

    STOP_WORDS = {
        "the", "and", "for", "with", "about", "from", "that", "this",
        "work", "task", "project", "team", "deal", "contract"
    }

    PROJECT_KEYWORD_REGEX = re.compile(
        r"\b(?:project|task|working\s+on|ref|deal|contract|team)\s+([A-Za-z0-9_\-]+)\b",
        re.IGNORECASE
    )

    FILE_EXTENSION_REGEX = re.compile(r"\b\w+\.(?:doc|docx|pdf|xls|xlsx|zip|tar|rar|csv|txt|ppt|pptx)\b", re.IGNORECASE)

    def __init__(self, mismatch_threshold: float = 0.5):
        self.mismatch_threshold = mismatch_threshold

    def extract_stated_topics(self, messages: List[CommunicationMessage]) -> Set[str]:
        """
        Extract declared projects, keywords, and explicit file artifacts from email text.
        """
        topics: Set[str] = set()
        for msg in messages:
            content = f"{msg.subject or ''} {msg.text_content}"
            if not content.strip():
                continue
            matches = self.PROJECT_KEYWORD_REGEX.findall(content)
            for m in matches:
                clean_m = m.lower().strip()
                if len(clean_m) >= 3 and clean_m not in self.STOP_WORDS:
                    topics.add(clean_m)
            file_refs = self.FILE_EXTENSION_REGEX.findall(content)
            for f in file_refs:
                clean_f = f.lower().strip()
                topics.add(clean_f)
        return topics

    def evaluate_mismatch(
        self,
        user_id: str,
        messages: List[CommunicationMessage],
        telemetry_events: List[TelemetryEvent],
        current_time: datetime
    ) -> MismatchAnalysisResult:
        """
        Compare telemetry resource paths against extracted topics.
        """
        stated_topics = self.extract_stated_topics(messages)
        file_events = [e for e in telemetry_events if e.event_type == "file_transfer"]
        usb_events = [e for e in telemetry_events if "device_connect" in e.event_type]
        has_usb = len(usb_events) > 0
        accessed_resources: List[str] = []
        unrelated_resources: List[str] = []
        for fe in file_events:
            target = fe.target_resource or ""
            if not target:
                continue
            accessed_resources.append(target)
            target_lower = target.lower()
            if stated_topics:
                matches_topic = any(topic in target_lower for topic in stated_topics)
                if not matches_topic:
                    unrelated_resources.append(target)
            else:
                unrelated_resources.append(target)
        total_files = len(accessed_resources)
        mismatch_ratio = (len(unrelated_resources) / total_files) if total_files > 0 else 0.0
        is_suspicious = (
            (total_files >= 2 and mismatch_ratio >= self.mismatch_threshold)
            or (has_usb and total_files > 0)
        )
        explanations: List[str] = []
        if has_usb:
            explanations.append(f"Removable storage connection registered ({len(usb_events)} events)")
        if stated_topics:
            explanations.append(f"Stated communication topics: {', '.join(sorted(stated_topics))}")
        else:
            explanations.append("No project references detected in recent communication")
        if unrelated_resources:
            explanations.append(
                f"Uncorrelated file transfers: {len(unrelated_resources)}/{total_files} "
                f"(mismatch ratio: {mismatch_ratio:.2f})"
            )
        full_explanation = "; ".join(explanations) if explanations else "Technical telemetry aligns with communication."

        return MismatchAnalysisResult(
            user_id=user_id,
            timestamp=current_time,
            stated_projects=sorted(stated_topics),
            accessed_resources=accessed_resources,
            unrelated_resources=unrelated_resources,
            mismatch_ratio=float(mismatch_ratio),
            has_usb_exfiltration=has_usb,
            is_suspicious_mismatch=is_suspicious,
            explanation=full_explanation
        )