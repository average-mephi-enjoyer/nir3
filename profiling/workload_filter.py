from collections import defaultdict
from datetime import timedelta
import math


class TeamWorkloadContextTracker:
    """Calendar-day peer load. Uses only completed days, with a one-day lag."""
    def __init__(self, window_days=7, baseline_window_days=30):
        if not 0 < window_days < baseline_window_days:
            raise ValueError("Invalid workload windows")
        self.window_days, self.baseline_days = window_days, baseline_window_days
        self.daily = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.first_seen = {}
        self.last_timestamp = None
        self.cache = {}

    def register_role_activity(self, role, timestamp, intensity=1.0, employee_id=None):
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Workload events must be chronological")
        if not math.isfinite(intensity) or intensity < 0:
            raise ValueError("Invalid activity intensity")
        self.last_timestamp = timestamp
        if employee_id:
            user = employee_id.lower()
            self.first_seen.setdefault((role, user), timestamp.date())
            self.daily[role][timestamp.date()][user] += intensity

    def get_role_workload_factor(self, role, current_time, exclude_user=None):
        date = current_time.date()
        key = role, date
        if key not in self.cache:
            counts = defaultdict(lambda: [0., 0.])
            for days in range(1, self.baseline_days + 1):
                for user, value in self.daily[role].get(date - timedelta(days=days), {}).items():
                    counts[user][0 if days <= self.window_days else 1] += value
            eligible = {user: values for user, values in counts.items() if min(values) > 0 and
                        self.first_seen[(role, user)] <= date - timedelta(days=self.baseline_days)}
            self.cache[key] = eligible
        peers = [values for user, values in self.cache[key].items() if user != (exclude_user or "").lower()]
        if len(peers) < 2:
            return 0.
        recent = sum(v[0] for v in peers) / self.window_days
        past = sum(v[1] for v in peers) / (self.baseline_days - self.window_days)
        return max(0., min(1., (recent / past - 1) / 2)) if past else 0.
