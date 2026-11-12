from collections import defaultdict, deque
from datetime import timedelta
import numpy as np


class TeamWorkloadContextTracker:
    def __init__(self, window_days=7, baseline_window_days=30):
        if not 0 < window_days < baseline_window_days:
            raise ValueError("Invalid workload windows")
        self.window = timedelta(days=window_days)
        self.baseline_window = timedelta(days=baseline_window_days)
        self.role_events = defaultdict(deque)
        self.last_timestamp = None

    def register_role_activity(self, role, timestamp, intensity=1.0, employee_id=None):
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Workload events must be chronological")
        self.last_timestamp = timestamp
        if not employee_id:
            return
        self.role_events[role].append((timestamp, float(intensity), employee_id.lower()))
        cutoff = timestamp - self.baseline_window
        for events in self.role_events.values():
            while events and events[0][0] < cutoff:
                events.popleft()

    def get_role_workload_factor(self, role, current_time, exclude_user=None):
        cutoff = current_time - self.baseline_window
        recent = current_time - self.window
        events = [(ts, value, user) for ts, value, user in self.role_events.get(role, [])
                  if cutoff <= ts < current_time and user != (exclude_user or "").lower()]
        if len({user for _, _, user in events}) < 2:
            return 0.0
        recent_sum = sum(value for ts, value, _ in events if ts >= recent)
        past_sum = sum(value for ts, value, _ in events if ts < recent)
        past_rate = past_sum / (self.baseline_window - self.window).days
        if past_rate <= 0:
            return 0.0
        surge = (recent_sum / self.window.days - past_rate) / past_rate
        return float(np.clip(surge / 2, 0, 1))
