from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from core.models import MarkerVector


@dataclass
class QuarantineObservation:
    """
    Event in the quarantine buffer with associated risk score and marker vector.
    """
    timestamp: datetime
    raw_markers: MarkerVector
    marker_dict: Dict[str, float]
    risk_score: float = 0.0


class QuarantineBuffer:
    r"""
    Quarantine buffer with a delay for admitting observations to the trusted profile strictly after 14 days.
    Implements the admission condition: max_{\tau \in [t - L, t]} Risk(\tau) < h_{quarantine}.
    If a risk spike >= h_{quarantine} is detected on the window [t - L, t],
    the observation t - L is rejected and does not enter the trusted database.
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
        Place a new observation in the quarantine buffer.
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
        Extract mature observations (older than 14 days).
        """
        mature_approved: List[QuarantineObservation] = []
        rejected_suspicious: List[QuarantineObservation] = []
        threshold_time = current_time - self.delay
        horizon_risk = 0.0
        if self.buffer:
            horizon_risk = max(obs.risk_score for obs in self.buffer)
        while self.buffer and self.buffer[0].timestamp <= threshold_time:
            candidate = self.buffer.popleft()
            max_window_risk = max(candidate.risk_score, horizon_risk)
            if max_window_risk < self.h_quarantine:
                mature_approved.append(candidate)
            else:
                rejected_suspicious.append(candidate)
        return mature_approved, rejected_suspicious

    def purge_suspicious_horizon(self, start_time: datetime) -> int:
        """
        Roll back the quarantine buffer when a persistent anomalous signal is detected.
        Removes all observations that arrived after the moment of drift detection.
        """
        initial_len = len(self.buffer)
        filtered = deque([obs for obs in self.buffer if obs.timestamp < start_time])
        self.buffer = filtered
        return initial_len - len(self.buffer)

    def get_operational_snapshot(self) -> List[QuarantineObservation]:
        """
        Returns a snapshot of the current operational observations in the quarantine buffer.
        """
        return list(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)