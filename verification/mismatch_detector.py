from dataclasses import dataclass
from datetime import datetime
import re
from typing import Dict, List, Set
from core.models import CommunicationMessage, TelemetryEvent


@dataclass(slots=True)
class MismatchAnalysisResult:
    user_id: str
    timestamp: datetime
    stated_projects: List[str]
    accessed_resources: List[str]
    unrelated_resources: List[str]
    mismatch_ratio: float
    has_usb_exfiltration: bool
    is_suspicious_mismatch: bool
    threat_severity: str
    explanation: str


class SemanticTechnicalMismatchDetector:
    DOMAIN_ONTOLOGY: Dict[str, Set[str]] = {
        "finance": {"finance", "financial", "audit", "budget", "billing", "revenue", "payroll", "tax", "cost", "invoice", "excel", "xlsx", "csv"},
        "engineering": {"engineering", "cad", "design", "schematic", "source", "code", "firmware", "build", "rnd", "git", "bin", "zip", "tar"},
        "hr_management": {"hiring", "interview", "resume", "performance", "review", "salary", "contract", "employee", "roster", "staff"},
        "legal": {"legal", "compliance", "policy", "lawyer", "nda", "patent", "intellectual", "agreement", "regulation"},
        "marketing_sales": {"client", "customer", "market", "proposal", "presentation", "deck", "campaign", "lead", "pptx"}
    }

    IGNORED_SYSTEM_PATTERNS = [
        re.compile(r"\\windows\\", re.IGNORECASE),
        re.compile(r"\\program files", re.IGNORECASE),
        re.compile(r"\\appdata\\", re.IGNORECASE),
        re.compile(r"\.(?:dll|exe|sys|tmp|lnk|dat|ini)$", re.IGNORECASE)
    ]

    NOISE_WORDS = {
        "the", "and", "for", "with", "about", "from", "that", "this", "project", "task",
        "team", "work", "worked", "working", "began", "much", "simply", "specific", "award",
        "common", "local", "available", "evidenced", "journalist", "threats", "edward", "glenn",
        "hello", "thanks", "regards", "please", "update", "meeting", "status"
    }

    TOKEN_PATTERN = re.compile(r"\b[a-zA-Z]{3,}\b")

    def __init__(self, mismatch_threshold: float = 0.5):
        self.mismatch_threshold = mismatch_threshold

    def _is_system_background_file(self, filepath: str) -> bool:
        return any(pat.search(filepath) for pat in self.IGNORED_SYSTEM_PATTERNS)

    def extract_stated_categories(self, messages: List[CommunicationMessage]) -> Set[str]:
        active_categories: Set[str] = set()
        for msg in messages:
            content = f"{msg.subject or ''} {msg.text_content}".lower()
            tokens = set(self.TOKEN_PATTERN.findall(content)) - self.NOISE_WORDS
            for category, keywords in self.DOMAIN_ONTOLOGY.items():
                if tokens.intersection(keywords):
                    active_categories.add(category)
        return active_categories

    def _resolve_resource_category(self, filepath: str) -> Set[str]:
        clean_path = filepath.lower().replace("/", "\\")
        matched_categories: Set[str] = set()
        for category, keywords in self.DOMAIN_ONTOLOGY.items():
            if any(kw in clean_path for kw in keywords):
                matched_categories.add(category)
        return matched_categories

    def evaluate_mismatch(
        self,
        user_id: str,
        messages: List[CommunicationMessage],
        telemetry_events: List[TelemetryEvent],
        current_time: datetime
    ) -> MismatchAnalysisResult:
        stated_categories = self.extract_stated_categories(messages)
        file_events = [e for e in telemetry_events if e.event_type == "file_transfer"]
        usb_events = [e for e in telemetry_events if "device_connect" in e.event_type]
        has_usb = len(usb_events) > 0
        meaningful_accessed: List[str] = []
        unrelated_resources: List[str] = []
        for fe in file_events:
            target = fe.target_resource or ""
            if not target or self._is_system_background_file(target):
                continue
            meaningful_accessed.append(target)
            target_lower = target.lower()
            resource_cats = self._resolve_resource_category(target_lower)
            category_match = bool(stated_categories.intersection(resource_cats))
            if not category_match:
                unrelated_resources.append(target)
        total_meaningful_files = len(meaningful_accessed)
        mismatch_ratio = float(len(unrelated_resources) / total_meaningful_files) if total_meaningful_files > 0 else 0.0
        if has_usb and total_meaningful_files > 0:
            severity = "CRITICAL_EXFILTRATION"
            is_suspicious = True
        elif total_meaningful_files >= 2 and mismatch_ratio >= self.mismatch_threshold:
            severity = "PROBING"
            is_suspicious = True
        else:
            severity = "BENIGN"
            is_suspicious = False
        explanations: List[str] = []
        if has_usb:
            explanations.append(f"Removable USB storage attached ({len(usb_events)} events)")
        if stated_categories:
            explanations.append(f"Declared communication domains: {', '.join(sorted(stated_categories))}")
        if unrelated_resources:
            explanations.append(
                f"Out-of-role resource exploration: {len(unrelated_resources)}/{total_meaningful_files} "
                f"(mismatch: {mismatch_ratio:.2f}, status: {severity})"
            )
        elif total_meaningful_files == 0:
            explanations.append("Host telemetry is calm. Early psycholinguistic drift warning.")
        else:
            explanations.append("All accessed files align with authorized communication context.")
        return MismatchAnalysisResult(
            user_id=user_id,
            timestamp=current_time,
            stated_projects=sorted(list(stated_categories)),
            accessed_resources=meaningful_accessed,
            unrelated_resources=unrelated_resources,
            mismatch_ratio=mismatch_ratio,
            has_usb_exfiltration=has_usb,
            is_suspicious_mismatch=is_suspicious,
            threat_severity=severity,
            explanation="; ".join(explanations)
        )
