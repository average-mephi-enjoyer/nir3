from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np


class TeamWorkloadContextTracker:
    def __init__(self, window_days: int = 7, baseline_window_days: int = 30):
        self.window = timedelta(days=window_days)
        self.baseline_window = timedelta(days=baseline_window_days)
        self.role_events: Dict[str, deque[Tuple[datetime, float]]] = defaultdict(deque)

    def register_role_activity(self, role: str, timestamp: datetime, intensity: float = 1.0) -> None:
        event_queue = self.role_events[role]
        event_queue.append((timestamp, intensity))
        cutoff = timestamp - self.baseline_window
        while event_queue and event_queue[0][0] < cutoff:
            event_queue.popleft()

    def get_role_workload_factor(self, role: str, current_time: datetime) -> float:
        event_queue = self.role_events.get(role)
        if not event_queue or len(event_queue) < 10:
            return 0.0
        recent_cutoff = current_time - self.window
        recent_sum = sum(val for ts, val in event_queue if ts >= recent_cutoff)
        past_sum = sum(val for ts, val in event_queue if ts < recent_cutoff)
        recent_days = max(self.window.total_seconds() / 86400.0, 1.0)
        baseline_days = max((self.baseline_window - self.window).total_seconds() / 86400.0, 1.0)
        recent_rate = recent_sum / recent_days
        past_rate = past_sum / baseline_days
        if past_rate <= 1e-4:
            return 0.0
        relative_surge = (recent_rate - past_rate) / past_rate
        workload_factor = float(np.clip(relative_surge / 2.0, 0.0, 1.0))
        return workload_factor
