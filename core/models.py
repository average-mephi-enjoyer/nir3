from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass(slots=True)
class CommunicationMessage:
    sender_id: str
    receiver_ids: List[str]
    timestamp: datetime
    sender_user_id: Optional[str] = None
    channel: str = "email"
    subject: Optional[str] = None
    body: Optional[str] = None
    cleaned_body: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return bool(self.sender_id and self.receiver_ids and self.timestamp and (self.cleaned_body or self.body))

    @property
    def text_content(self) -> str:
        return (self.cleaned_body or self.body or "").strip()


@dataclass(slots=True)
class EmployeeContext:
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


@dataclass(slots=True)
class MarkerVector:
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

    def to_numpy(self) -> np.ndarray:
        return np.array([
            self.pronoun_marker,
            self.syntactic_marker,
            self.activity_marker,
            self.response_marker,
            self.social_marker,
            self.llm_alienation,
            self.llm_concealment,
            self.llm_rationalization,
            self.llm_access_focus,
            self.llm_conflict,
        ], dtype=np.float64)

    @classmethod
    def feature_names(cls) -> List[str]:
        return [
            "pronoun_marker", "syntactic_marker", "activity_marker", 
            "response_marker", "social_marker", "llm_alienation", 
            "llm_concealment", "llm_rationalization", "llm_access_focus", 
            "llm_conflict"
        ]


@dataclass(slots=True)
class RiskSignal:
    employee_id: str
    timestamp: datetime
    raw_markers: MarkerVector
    normalized_deviations: Dict[str, float]
    channel: str = "email"
    explanations: List[str] = field(default_factory=list)
    evidence_spans: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TelemetryEvent:
    event_id: str
    user_id: str
    timestamp: datetime
    event_type: str
    pc_id: str
    target_resource: Optional[str] = None
    activity_details: Dict[str, Any] = field(default_factory=dict)
