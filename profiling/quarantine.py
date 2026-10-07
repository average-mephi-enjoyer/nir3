from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from core.models import MarkerVector


@dataclass
class QuarantineObservation:
    """
    Quarantined element.
    """
    timestamp: datetime
    raw_markers: MarkerVector
    marker_dict: Dict[str, float]
    anomaly_score: float = 0.0


class QuarantineBuffer:
    """
    Quarantine buffer with a delay for admitting observations to the trusted profile strictly after 14 days.
    Excludes self-normalization and adaptive contamination of the baseline norm during slow drift.
    """

    def __init__(
        self,
        quarantine_delay_days: int = 14,
        suspicion_threshold: float = 2.5
    ):
        self.delay = timedelta(days=quarantine_delay_days)
        self.suspicion_threshold = suspicion_threshold
        self.buffer: deque[QuarantineObservation] = deque()

    def enqueue(
        self,
        timestamp: datetime,
        markers: MarkerVector,
        marker_dict: Dict[str, float],
        anomaly_score: float = 0.0
    ) -> None:
        """
        Enqueue a new observation in the quarantine buffer.
        """
        obs = QuarantineObservation(
            timestamp=timestamp,
            raw_markers=markers,
            marker_dict=marker_dict,
            anomaly_score=anomaly_score
        )
        self.buffer.append(obs)

    def extract_mature_clean_observations(
        self,
        current_time: datetime
    ) -> Tuple[List[QuarantineObservation], List[QuarantineObservation]]:
        """
        Extract mature observations (older than 14 days).
        Returns a tuple: (admitted for training, rejected suspicious).
        """
        mature_approved: List[QuarantineObservation] = []
        rejected_suspicious: List[QuarantineObservation] = []
        threshold_time = current_time - self.delay
        while self.buffer and self.buffer[0].timestamp <= threshold_time:
            obs = self.buffer.popleft()
            if obs.anomaly_score < self.suspicion_threshold:
                mature_approved.append(obs)
            else:
                rejected_suspicious.append(obs)
        return mature_approved, rejected_suspicious

    def purge_suspicious_horizon(self, start_time: datetime) -> int:
        """
        Roll back the quarantine buffer when a persistent anomalous signal is detected.
        Removes all observations that arrived after the moment of drift onset fixation.
        """
        initial_len = len(self.buffer)
        filtered = deque([obs for obs in self.buffer if obs.timestamp < start_time])
        self.buffer = filtered
        return initial_len - len(self.buffer)

    def get_operational_snapshot(self) -> List[QuarantineObservation]:
        """
        Returns a snapshot of current operational observations in the quarantine (for operational context).
        """
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)