from collections import deque
from dataclasses import dataclass
from datetime import timedelta
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


class QuarantineBuffer:
    def __init__(self, quarantine_delay_days=14, h_quarantine=2.0):
        if quarantine_delay_days < 0 or h_quarantine <= 0:
            raise ValueError("Invalid quarantine parameters")
        self.delay = timedelta(days=quarantine_delay_days)
        self.h_quarantine = h_quarantine
        self.buffer = deque()
        self.decisions = []
        self.last_timestamp = None

    def enqueue(self, timestamp, markers, marker_dict, risk_score=0.0, observation_id=""):
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Observations must be chronological")
        self.last_timestamp = timestamp
        normalized = risk_score / self.h_quarantine
        for candidate in self.buffer:
            if timestamp <= candidate.timestamp + self.delay:
                candidate.horizon_risk = max(candidate.horizon_risk, normalized)
        self.buffer.append(QuarantineObservation(timestamp, markers, dict(marker_dict), risk_score, normalized,
                                                 observation_id=observation_id))

    def extract_mature_clean_observations(self, current_time):
        approved, rejected = [], []
        while self.buffer and self.buffer[0].timestamp + self.delay <= current_time:
            candidate = self.buffer.popleft()
            candidate.reason = "approved" if candidate.horizon_risk < 1.0 else "risk_in_quarantine_horizon"
            (approved if candidate.reason == "approved" else rejected).append(candidate)
            self.decisions.append(candidate)
        return approved, rejected

    def purge_suspicious_horizon(self, start_time):
        preserved = deque()
        removed = 0
        for candidate in self.buffer:
            if candidate.timestamp + self.delay >= start_time:
                candidate.reason = "alert_horizon"
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
