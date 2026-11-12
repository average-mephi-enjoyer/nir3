from collections import deque
from datetime import timedelta
import math
import re
from core.models import EmployeeContext


class SlidingTimeWindow:
    def __init__(self, employee_id, role="general", email=None):
        self.employee_id = employee_id.lower()
        self.role = role
        self.email = (email or employee_id).lower()
        self.events = deque()
        self.pending = {}
        self.reply_latencies = deque()
        self.last_timestamp = None

    def add_message(self, message):
        timestamp = message.timestamp
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Messages must be chronological")
        self.last_timestamp = timestamp
        sender = message.sender_id.lower()
        recipients = {r.lower() for r in message.receiver_ids}
        is_sender = (message.sender_user_id or "").lower() == self.employee_id or sender == self.email
        subject = re.sub(r"^(?:(?:re|fw|fwd):\s*)+", "", message.subject or "", flags=re.IGNORECASE).strip().lower()
        if is_sender:
            self.events.append((timestamp, "sent"))
            for peer in recipients:
                key = (peer, subject)
                if subject and key in self.pending and re.match(r"^re:", message.subject or "", re.IGNORECASE):
                    incoming = self.pending.pop(key)
                    latency = (timestamp - incoming).total_seconds() / 3600
                    if 0 <= latency <= 168:
                        self.reply_latencies.append((timestamp, latency))
        elif self.email in recipients or self.employee_id in recipients:
            self.events.append((timestamp, "received"))
            if subject:
                self.pending.setdefault((sender, subject), timestamp)
        self._evict_expired(timestamp)

    def _evict_expired(self, current_time):
        while self.events and self.events[0][0] < current_time - timedelta(days=180):
            self.events.popleft()
        while self.reply_latencies and self.reply_latencies[0][0] < current_time - timedelta(days=30):
            self.reply_latencies.popleft()
        self.pending = {key: ts for key, ts in self.pending.items() if ts >= current_time - timedelta(days=7)}

    def get_activity_dynamics(self):
        if self.last_timestamp is None:
            return 0.0, 0.0, 0.0, 0.0
        result = []
        for days in (1, 7, 30, 90):
            boundary = self.last_timestamp - timedelta(days=days)
            previous = boundary - timedelta(days=days)
            current = sum(1 for ts, kind in self.events if ts > boundary and kind == "sent")
            past = sum(1 for ts, kind in self.events if previous < ts <= boundary and kind == "sent")
            result.append((current - past) / max(past, 1))
        return tuple(result)

    def get_reply_statistics(self):
        values = sorted(value for _, value in self.reply_latencies)
        if not values:
            return 0.0, 0.0, 0.0
        mean = sum(values) / len(values)
        middle = len(values) // 2
        median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
        return mean, median, math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    def export_context(self):
        daily, weekly, monthly, quarterly = self.get_activity_dynamics()
        mean, median, std = self.get_reply_statistics()
        return EmployeeContext(employee_id=self.employee_id, role=self.role, mean_reply_time=mean,
                               median_reply_time=median, reply_time_std=std, daily_activity_change=daily,
                               weekly_activity_change=weekly, monthly_activity_change=monthly,
                               quarterly_activity_change=quarterly)
