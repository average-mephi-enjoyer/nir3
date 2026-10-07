from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np


class TeamWorkloadContextTracker:
    """
    Contextual tracker of team workload and project deadlines.
    Monitors aggregated activity dynamics of departments/roles.
    Contextual tracker of team workload and project deadlines.
    Monitors aggregated activity dynamics of departments/roles.
    If the entire group synchronously experiences increased workload, communication speed, or delays,
    a contextual workload factor is formed in the range [0.0, 1.0],
    which dynamically raises the anomaly thresholds h_quarantine and h_alert,
    reducing the risk of false positives during periods of collective stress.
    """

    def __init__(self, window_days: int = 7, baseline_window_days: int = 30):
        self.window = timedelta(days=window_days)
        self.baseline_window = timedelta(days=baseline_window_days)
        self.role_events: Dict[str, deque] = defaultdict(deque)

    def register_role_activity(self, role: str, timestamp: datetime, intensity: float = 1.0) -> None:
        """
        Registration of a role/department activity event.
        """
        q = self.role_events[role]
        q.append((timestamp, intensity))
        cutoff = timestamp - self.baseline_window
        while q and q[0][0] < cutoff:
            q.popleft()

    def get_role_workload_factor(self, role: str, current_time: datetime) -> float:
        """
        Computes the contextual workload factor for a given role:
        0.0 — normal workload,
        up to 1.0 — extreme collective deadline / project crisis.
        """
        q = self.role_events.get(role)
        if not q or len(q) < 10:
            return 0.0
        recent_cutoff = current_time - self.window
        recent_intensities = [val for ts, val in q if ts >= recent_cutoff]
        past_intensities = [val for ts, val in q if ts < recent_cutoff]
        if not recent_intensities or not past_intensities:
            return 0.0
        recent_rate = sum(recent_intensities) / max(self.window.total_seconds() / 86400.0, 1.0)
        past_rate = sum(past_intensities) / max((self.baseline_window - self.window).total_seconds() / 86400.0, 1.0)
        if past_rate <= 1e-4:
            return 0.0
        relative_surge = (recent_rate - past_rate) / past_rate
        workload_factor = float(np.clip(relative_surge / 2.0, 0.0, 1.0))
        return workload_factor