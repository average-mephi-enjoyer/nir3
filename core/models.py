from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class CommunicationMessage:
    """
    Communication event (Email, мессенджер).
    """
    sender_id: str
    receiver_ids: List[str]
    timestamp: datetime
    channel: str = "email"
    subject: Optional[str] = None
    body: Optional[str] = None
    cleaned_body: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.sender_id and self.receiver_ids and self.timestamp and (self.cleaned_body or self.body))


@dataclass
class EmployeeContext:
    """
    Operational and graph-based context of an employee within the observation window.
    """
    employee_id: str
    role: str = "general"
    indegree: float = 0.0
    outdegree: float = 0.0
    reciprocity: float = 0.0
    betweenness: float = 0.0
    mean_reply_time: float = 0.0
    median_reply_time: float = 0.0
    reply_time_std: float = 0.0
    daily_activity_change: float = 0.0
    weekly_activity_change: float = 0.0
    monthly_activity_change: float = 0.0
    quarterly_activity_change: float = 0.0


@dataclass
class MarkerVector:
    """
    Combined feature vector (statistical NLP + LLM).
    """
    pronoun_marker: float = 0.0
    syntactic_marker: float = 0.0
    activity_marker: float = 0.0
    response_marker: float = 0.0
    social_marker: float = 0.0
    llm_alienation: float = 0.0
    llm_concealment: float = 0.0
    llm_rationalization: float = 0.0
    llm_access_focus: float = 0.0
    llm_conflict: float = 0.0
    llm_uncertainty: float = 1.0


@dataclass
class RiskSignal:
    """
    Early warning signal output.
    """
    employee_id: str
    timestamp: datetime
    raw_markers: MarkerVector
    normalized_deviations: Dict[str, float]
    channel: str = "email"
    explanations: List[str] = field(default_factory=list)
    evidence_spans: List[str] = field(default_factory=list)