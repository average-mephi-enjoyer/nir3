from collections import defaultdict, deque
from datetime import timedelta
from typing import Dict

from core.models import CommunicationMessage, RiskSignal, TelemetryEvent
from verification.mismatch_detector import MismatchAnalysisResult, SemanticTechnicalMismatchDetector


class TargetedRetrospectiveVerifier:
    """
    Retrospective targeted verification of semantic-technical mismatch.
    """
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
        self.mismatch_detector = SemanticTechnicalMismatchDetector(mismatch_threshold=mismatch_threshold)

    def register_telemetry_event(self, event: TelemetryEvent) -> None:
        """
        Passive registration of a system event into the user's lightweight queue.
        """
        store = self.telemetry_store[event.user_id]
        store.append(event)
        cutoff = event.timestamp - self.max_history
        while store and store[0].timestamp < cutoff:
            store.popleft()

    def register_communication_message(self, message: CommunicationMessage) -> None:
        """
        Context saving.
        """
        store = self.message_store[message.sender_id]
        store.append(message)
        cutoff = message.timestamp - self.max_history
        while store and store[0].timestamp < cutoff:
            store.popleft()

    def verify_alert_signal(self, signal: RiskSignal) -> MismatchAnalysisResult:
        """
        Retrospective targeted verification:
        Triggered strictly by the first-stage signal within the window [signal.timestamp - Delta t_re
        """
        user_id = signal.employee_id
        current_time = signal.timestamp
        start_time = current_time - self.retro_window
        user_events = [
            e for e in self.telemetry_store[user_id]
            if start_time <= e.timestamp <= current_time
        ]
        user_messages = [
            m for m in self.message_store[user_id]
            if start_time <= m.timestamp <= current_time
        ]
        result = self.mismatch_detector.evaluate_mismatch(
            user_id=user_id,
            messages=user_messages,
            telemetry_events=user_events,
            current_time=current_time
        )
        return result