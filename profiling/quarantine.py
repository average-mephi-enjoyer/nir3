from collections import deque
from dataclasses import dataclass
from datetime import timedelta
import math
from core.models import MarkerVector


@dataclass(slots=True)
class QuarantineObservation:
    timestamp: object
    raw_markers: MarkerVector
    marker_dict: dict
    risk_score: float = 0.0
    horizon_risk: float = 0.0
    reason: str = "pending"
    observation_id: str = ""
    threshold: float = 0.0
    normalized_risk: float = 0.0
    status: str = "PENDING"
    included_in_reference: bool = False
    history: list = None

    def transition(self, status, timestamp, reason, included=False):
        self.status, self.reason, self.included_in_reference = status, reason, included
        if self.history is None:
            self.history = []
        self.history.append({"timestamp": str(timestamp), "status": status, "reason": reason,
                             "included_in_reference": included})


class QuarantineBuffer:
    def __init__(self, quarantine_delay_days=14, h_quarantine=2.0):
        if quarantine_delay_days < 0 or h_quarantine <= 0:
            raise ValueError("Invalid quarantine parameters")
        self.delay = timedelta(days=quarantine_delay_days)
        self.h_quarantine = h_quarantine
        self.buffer = deque()
        self.decisions = []
        self.last_timestamp = None
        self.seen_ids = set()
        self.decision_time = None

    def enqueue(self, timestamp, markers, marker_dict, risk_score=0.0, observation_id=""):
        if not math.isfinite(risk_score) or not math.isfinite(self.h_quarantine) or self.h_quarantine <= 0:
            raise ValueError("Finite risk and positive threshold required")
        observation_id = observation_id or f"anonymous:{len(self.seen_ids)}"
        if observation_id in self.seen_ids:
            raise ValueError("Duplicate quarantine observation ID")
        if self.decision_time is not None and timestamp < self.decision_time:
            raise ValueError("Cannot insert events before an already recorded decision")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Observations must be chronological")
        self.last_timestamp = timestamp
        self.seen_ids.add(observation_id)
        normalized = risk_score / self.h_quarantine
        for candidate in self.buffer:
            if timestamp <= candidate.timestamp + self.delay:
                candidate.horizon_risk = max(candidate.horizon_risk, normalized)
        candidate = QuarantineObservation(timestamp, markers, dict(marker_dict), risk_score, normalized,
                                           observation_id=observation_id, threshold=self.h_quarantine,
                                           normalized_risk=normalized)
        candidate.transition("PENDING", timestamp, "pending")
        self.buffer.append(candidate)

    def extract_mature_clean_observations(self, current_time):
        if self.last_timestamp is not None and current_time < self.last_timestamp:
            raise ValueError("Decision cannot use future observations")
        if self.decision_time is not None and current_time < self.decision_time:
            raise ValueError("Decisions must be chronological")
        self.decision_time = current_time
        approved, rejected = [], []
        while self.buffer and self.buffer[0].timestamp + self.delay <= current_time:
            candidate = self.buffer.popleft()
            candidate.reason = "approved" if candidate.horizon_risk < 1.0 else "risk_in_quarantine_horizon"
            candidate.transition("ADMITTED" if candidate.reason == "approved" else "REJECTED", current_time, candidate.reason)
            (approved if candidate.reason == "approved" else rejected).append(candidate)
            self.decisions.append(candidate)
        return approved, rejected

    def purge_suspicious_horizon(self, start_time):
        preserved = deque()
        removed = 0
        for candidate in self.buffer:
            if candidate.timestamp + self.delay >= start_time:
                candidate.transition("REJECTED", self.last_timestamp or start_time, "alert_horizon")
                self.decisions.append(candidate)
                removed += 1
            else:
                preserved.append(candidate)
        self.buffer = preserved
        return removed

    def get_operational_snapshot(self):
        return list(self.buffer)

    def __len__(self):
        return len(self.buffer)
