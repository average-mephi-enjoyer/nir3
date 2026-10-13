from collections import deque
from datetime import datetime, timedelta
import math
from typing import Deque, List, Optional, Tuple
from core.models import CommunicationMessage, EmployeeContext


class SlidingTimeWindow:
    def __init__(self, employee_id: str, role: str = "general"):
        self.employee_id = employee_id.lower().strip()
        self.role = role
        self.queue_1d: Deque[Tuple[datetime, str, str]] = deque()
        self.queue_7d: Deque[Tuple[datetime, str, str]] = deque()
        self.queue_30d: Deque[Tuple[datetime, str, str]] = deque()
        self.queue_90d: Deque[Tuple[datetime, str, str]] = deque()
        self.pending_inbox: Deque[Tuple[datetime, str]] = deque()
        self.reply_latencies_hours: Deque[Tuple[datetime, float]] = deque()

    def add_message(self, message: CommunicationMessage) -> None:
        ts = message.timestamp
        sender_uid = (message.sender_user_id or "").lower().strip()
        sender_email = message.sender_id.lower().strip()
        is_sender = (sender_uid == self.employee_id or self.employee_id in sender_email)
        if is_sender:
            for peer in message.receiver_ids:
                event = (ts, "sent", peer.lower())
                self.queue_1d.append(event)
                self.queue_7d.append(event)
                self.queue_30d.append(event)
                self.queue_90d.append(event)
            self._resolve_replies(ts)
        else:
            event = (ts, "received", sender_email)
            self.queue_1d.append(event)
            self.queue_7d.append(event)
            self.queue_30d.append(event)
            self.queue_90d.append(event)
            self.pending_inbox.append((ts, sender_email))
        self._evict_expired(ts)

    def _resolve_replies(self, sent_ts: datetime) -> None:
        matched = False
        remaining_inbox: Deque[Tuple[datetime, str]] = deque()
        while self.pending_inbox:
            recv_ts, peer = self.pending_inbox.popleft()
            if not matched and recv_ts <= sent_ts:
                latency_hrs = (sent_ts - recv_ts).total_seconds() / 3600.0
                if 0.0 <= latency_hrs <= 168.0:
                    self.reply_latencies_hours.append((sent_ts, latency_hrs))
                    matched = True
                    continue
            remaining_inbox.append((recv_ts, peer))
        self.pending_inbox = remaining_inbox

    def _evict_expired(self, current_time: datetime) -> None:
        c1 = current_time - timedelta(days=1)
        c7 = current_time - timedelta(days=7)
        c30 = current_time - timedelta(days=30)
        c90 = current_time - timedelta(days=90)
        while self.queue_1d and self.queue_1d[0][0] < c1:
            self.queue_1d.popleft()
        while self.queue_7d and self.queue_7d[0][0] < c7:
            self.queue_7d.popleft()
        while self.queue_30d and self.queue_30d[0][0] < c30:
            self.queue_30d.popleft()
        while self.queue_90d and self.queue_90d[0][0] < c90:
            self.queue_90d.popleft()
        while self.reply_latencies_hours and self.reply_latencies_hours[0][0] < c30:
            self.reply_latencies_hours.popleft()
        cutoff_pending = current_time - timedelta(days=14)
        while self.pending_inbox and self.pending_inbox[0][0] < cutoff_pending:
            self.pending_inbox.popleft()

    def get_activity_dynamics(self) -> Tuple[float, float, float, float]:
        r1 = float(len(self.queue_1d))
        r7 = len(self.queue_7d) / 7.0
        r30 = len(self.queue_30d) / 30.0
        r90 = len(self.queue_90d) / 90.0
        d_daily = (r1 - r7) / max(r7, 1.0)
        d_weekly = (r7 - r30) / max(r30, 1.0)
        d_monthly = (r30 - r90) / max(r90, 1.0)
        d_quarterly = (r90 - max(r90 * 0.9, 0.5)) / max(r90, 0.5)
        return float(d_daily), float(d_weekly), float(d_monthly), float(d_quarterly)

    def get_reply_statistics(self) -> Tuple[float, float, float]:
        if not self.reply_latencies_hours:
            return 0.0, 0.0, 0.0
        latencies = [val for _, val in self.reply_latencies_hours]
        mean_val = float(sum(latencies) / len(latencies))
        sorted_vals = sorted(latencies)
        n = len(sorted_vals)
        median_val = float(sorted_vals[n // 2]) if n % 2 == 1 else float((sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0)
        variance = sum((x - mean_val) ** 2 for x in latencies) / len(latencies)
        std_val = float(math.sqrt(variance))
        return mean_val, median_val, std_val

    def export_context(self) -> EmployeeContext:
        d_daily, d_weekly, d_monthly, d_quarterly = self.get_activity_dynamics()
        mean_rep, med_rep, std_rep = self.get_reply_statistics()
        return EmployeeContext(
            employee_id=self.employee_id,
            role=self.role,
            mean_reply_time=mean_rep,
            median_reply_time=med_rep,
            reply_time_std=std_rep,
            daily_activity_change=d_daily,
            weekly_activity_change=d_weekly,
            monthly_activity_change=d_monthly,
            quarterly_activity_change=d_quarterly
        )
    