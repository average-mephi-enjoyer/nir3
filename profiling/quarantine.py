from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from core.models import MarkerVector


@dataclass(slots=True)
class QuarantineObservation:
    timestamp: datetime
    raw_markers: MarkerVector
    marker_dict: Dict[str, float]
    risk_score: float = 0.0


class QuarantineBuffer:
    def __init__(
        self,
        quarantine_delay_days: int = 14,
        h_quarantine: float = 2.0
    ):
        self.delay = timedelta(days=quarantine_delay_days)
        self.h_quarantine = h_quarantine
        self.buffer: deque[QuarantineObservation] = deque()

    def enqueue(
        self,
        timestamp: datetime,
        markers: MarkerVector,
        marker_dict: Dict[str, float],
        risk_score: float = 0.0
    ) -> None:
        obs = QuarantineObservation(
            timestamp=timestamp,
            raw_markers=markers,
            marker_dict=marker_dict,
            risk_score=risk_score
        )
        self.buffer.append(obs)

    def extract_mature_clean_observations(
        self,
        current_time: datetime
    ) -> Tuple[List[QuarantineObservation], List[QuarantineObservation]]:
        mature_approved: List[QuarantineObservation] = []
        rejected_suspicious: List[QuarantineObservation] = []
        admission_cutoff = current_time - self.delay
        while self.buffer and self.buffer[0].timestamp <= admission_cutoff:
            candidate = self.buffer.popleft()
            if candidate.risk_score < self.h_quarantine:
                mature_approved.append(candidate)
            else:
                rejected_suspicious.append(candidate)

        return mature_approved, rejected_suspicious

    def purge_suspicious_horizon(self, start_time: datetime) -> int:
        initial_len = len(self.buffer)
        preserved_elements = [obs for obs in self.buffer if obs.timestamp < start_time]
        self.buffer = deque(preserved_elements)
        return initial_len - len(self.buffer)

    def get_operational_snapshot(self) -> List[QuarantineObservation]:
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)