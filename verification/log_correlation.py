from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional
from core.models import CommunicationMessage, RiskSignal, TelemetryEvent
from verification.mismatch_detector import MismatchAnalysisResult, SemanticTechnicalMismatchDetector


class TargetedRetrospectiveVerifier:
    def __init__(
        self,
        retro_window_days: int = 7,
        max_telemetry_history_days: int = 30,
        mismatch_threshold: float = 0.5
    ):
        self.retro_window = timedelta(days=retro_window_days)
        self.max_history = timedelta(days=max_telemetry_history_days)
        self.telemetry_store: Dict[str, deque[TelemetryEvent]] = defaultdict(deque)
        self.message_store: Dict[str, deque[CommunicationMessage]] = defaultdict(deque)
        self.user_to_email_map: Dict[str, str] = {}
        self.email_to_user_map: Dict[str, str] = {}
        self.mismatch_detector = SemanticTechnicalMismatchDetector(mismatch_threshold=mismatch_threshold)

    def register_user_mapping(self, user_id: str, email: str) -> None:
        norm_user = user_id.strip().lower()
        norm_email = email.strip().lower()
        self.user_to_email_map[norm_user] = norm_email
        self.email_to_user_map[norm_email] = norm_user

    def register_telemetry_event(self, event: TelemetryEvent) -> None:
        user_key = event.user_id.strip().lower()
        store = self.telemetry_store[user_key]
        store.append(event)
        cutoff = event.timestamp - self.max_history
        while store and store[0].timestamp < cutoff:
            store.popleft()

    def register_communication_message(self, message: CommunicationMessage, user_id: Optional[str] = None) -> None:
        sender_email = message.sender_id.strip().lower()
        resolved_uid = user_id.strip().lower() if user_id else self.email_to_user_map.get(sender_email, sender_email)
        for key in {sender_email, resolved_uid}:
            store = self.message_store[key]
            store.append(message)
            cutoff = message.timestamp - self.max_history
            while store and store[0].timestamp < cutoff:
                store.popleft()

    def verify_alert_signal(self, signal: RiskSignal) -> MismatchAnalysisResult:
        user_id = signal.employee_id.strip().lower()
        current_time = signal.timestamp
        start_time = current_time - self.retro_window
        user_events = [
            e for e in self.telemetry_store.get(user_id, [])
            if start_time <= e.timestamp <= current_time
        ]
        email_alias = self.user_to_email_map.get(user_id, user_id)
        candidate_messages = self.message_store.get(user_id, deque())
        if not candidate_messages and email_alias != user_id:
            candidate_messages = self.message_store.get(email_alias, deque())
        user_messages = [
            m for m in candidate_messages
            if start_time <= m.timestamp <= current_time
        ]
        return self.mismatch_detector.evaluate_mismatch(
            user_id=user_id,
            messages=user_messages,
            telemetry_events=user_events,
            current_time=current_time
        )
