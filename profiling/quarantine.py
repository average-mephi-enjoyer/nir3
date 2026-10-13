"""
profiling/quarantine.py
Quarantine admission buffer for delay-based baseline protection.
"""
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from core.models import MarkerVector


@dataclass(slots=True)
class QuarantineObservation:
    """
    Observation pending admission in the quarantine buffer.
    """
    timestamp: datetime
    raw_markers: MarkerVector
    marker_dict: Dict[str, float]
    risk_score: float = 0.0


class QuarantineBuffer:
    """
    Quarantine buffer enforcing admission delay (default: 14 days).
    Admissibility condition: an observation at time (t - delay) is admitted to the trusted
    profile if and only if no anomalous risk spikes >= h_quarantine occurred within its evaluation window.
    """

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
        """
        Append a new observation to the quarantine pipeline.
        """
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
        """
        Extract observations older than the quarantine delay.
        Evaluates risk localized to the post-observation quarantine horizon.
        """
        mature_approved: List[QuarantineObservation] = []
        rejected_suspicious: List[QuarantineObservation] = []
        admission_cutoff = current_time - self.delay

        # Evaluate risk of elements that were observed concurrently or during quarantine
        while self.buffer and self.buffer[0].timestamp <= admission_cutoff:
            candidate = self.buffer.popleft()
            if candidate.risk_score < self.h_quarantine:
                mature_approved.append(candidate)
            else:
                rejected_suspicious.append(candidate)

        return mature_approved, rejected_suspicious

    def purge_suspicious_horizon(self, start_time: datetime) -> int:
        """
        Purge all quarantined windows arriving at or after the incident trigger timestamp.
        Prevents slow, persistent drift contamination.
        """
        initial_len = len(self.buffer)
        preserved_elements = [obs for obs in self.buffer if obs.timestamp < start_time]
        self.buffer = deque(preserved_elements)
        return initial_len - len(self.buffer)

    def get_operational_snapshot(self) -> List[QuarantineObservation]:
        """
        Return snapshot of currently pending quarantine items.
        """
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)