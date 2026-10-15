from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np
from core.models import CommunicationMessage, MarkerVector, TelemetryEvent


@dataclass
class SyntheticScenario:
    scenario_id: str
    scenario_type: str  
    target_user: str
    t0_start: datetime
    t_drift_start: Optional[datetime]
    t_first_action: Optional[datetime]
    synthetic_messages: List[CommunicationMessage] = field(default_factory=list)
    synthetic_telemetry: List[TelemetryEvent] = field(default_factory=list)
    is_malicious: bool = True


class DriftInjector:
    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_acute_scenario(
        self,
        user_id: str,
        start_time: datetime,
        duration_days: int = 30
    ) -> SyntheticScenario:
        scenario = SyntheticScenario(
            scenario_id=f"acute_{user_id}",
            scenario_type="acute_exfiltration",
            target_user=user_id,
            t0_start=start_time,
            t_drift_start=start_time + timedelta(days=15),
            t_first_action=start_time + timedelta(days=20),
            is_malicious=True
        )
        for day in range(duration_days):
            current_date = start_time + timedelta(days=day, hours=10)
            if day < 15:
                msg = CommunicationMessage(
                    sender_id=user_id,
                    receiver_ids=["colleague@company.com"],
                    timestamp=current_date,
                    subject="Regular sync",
                    cleaned_body="Daily progress report for project Alpha. Everything on schedule."
                )
            elif 15 <= day < 20:
                msg = CommunicationMessage(
                    sender_id=user_id,
                    receiver_ids=["manager@company.com"],
                    timestamp=current_date,
                    subject="Refusal",
                    cleaned_body="I hate this management! You are disregarding all agreements. I won't follow this rule."
                )
            else:
                msg = CommunicationMessage(
                    sender_id=user_id,
                    receiver_ids=["external@personal.com"],
                    timestamp=current_date,
                    subject="Private note",
                    cleaned_body="Don't mention this export to anyone. Deleting audit trails now."
                )
            scenario.synthetic_messages.append(msg)
            if day >= 20:
                scenario.synthetic_telemetry.append(
                    TelemetryEvent(
                        event_id=f"usb_{day}",
                        user_id=user_id,
                        timestamp=current_date + timedelta(hours=2),
                        event_type="device_connect",
                        pc_id="PC-042",
                        target_resource="removable_usb_drive"
                    )
                )
                scenario.synthetic_telemetry.append(
                    TelemetryEvent(
                        event_id=f"file_{day}",
                        user_id=user_id,
                        timestamp=current_date + timedelta(hours=2, minutes=5),
                        event_type="file_transfer",
                        pc_id="PC-042",
                        target_resource="C:\\Confidential\\financial_records.zip"
                    )
                )
        return scenario

    def generate_ramp_drift_scenario(
        self,
        user_id: str,
        start_time: datetime,
        duration_days: int = 60,
        l_ramp_days: int = 30,
        delta_max: float = 3.5
    ) -> SyntheticScenario:
        t_drift = start_time + timedelta(days=10)
        t_action = start_time + timedelta(days=45)
        scenario = SyntheticScenario(
            scenario_id=f"ramp_{user_id}",
            scenario_type="slow_ramp_drift",
            target_user=user_id,
            t0_start=start_time,
            t_drift_start=t_drift,
            t_first_action=t_action,
            is_malicious=True
        )
        for day in range(duration_days):
            current_date = start_time + timedelta(days=day, hours=11)
            if current_date < t_drift:
                drift_factor = 0.0
            else:
                elapsed_days = (current_date - t_drift).total_seconds() / 86400.0
                drift_factor = delta_max * min(1.0, elapsed_days / float(l_ramp_days))
            if drift_factor < 1.0:
                body = f"Status update day {day}. All project activities moving standard."
            elif 1.0 <= drift_factor < 2.5:
                body = f"Working on tasks. Nobody listens to our team concerns anyway, process is completely unfair."
            else:
                body = f"Bypassing policy restrictions. They don't deserve this IP. Staging data packages."
            msg = CommunicationMessage(
                sender_id=user_id,
                receiver_ids=["team@company.com"],
                timestamp=current_date,
                subject=f"Update {day}",
                cleaned_body=body
            )
            scenario.synthetic_messages.append(msg)
            if current_date >= t_action:
                scenario.synthetic_telemetry.append(
                    TelemetryEvent(
                        event_id=f"file_ramp_{day}",
                        user_id=user_id,
                        timestamp=current_date + timedelta(hours=3),
                        event_type="file_transfer",
                        pc_id="PC-089",
                        target_resource=f"\\\\server\\rnd_archive_part_{day}.tar.gz"
                    )
                )
        return scenario

    def generate_stress_control_scenario(
        self,
        user_id: str,
        start_time: datetime,
        duration_days: int = 40
    ) -> SyntheticScenario:
        scenario = SyntheticScenario(
            scenario_id=f"stress_{user_id}",
            scenario_type="stress_negative_control",
            target_user=user_id,
            t0_start=start_time,
            t_drift_start=start_time + timedelta(days=15),
            t_first_action=None,
            is_malicious=False
        )
        for day in range(duration_days):
            current_date = start_time + timedelta(days=day, hours=14)
            if 15 <= day < 30:
                body = "Total exhaustion. Working 14 hours straight to deliver the release by deadline. Extremely tired!"
            else:
                body = "Post-release stabilization. Tasks completed successfully."
            msg = CommunicationMessage(
                sender_id=user_id,
                receiver_ids=["pm@company.com"],
                timestamp=current_date,
                subject="Sprint Delivery",
                cleaned_body=body
            )
            scenario.synthetic_messages.append(msg)
            scenario.synthetic_telemetry.append(
                TelemetryEvent(
                    event_id=f"logon_{day}",
                    user_id=user_id,
                    timestamp=current_date - timedelta(hours=5),
                    event_type="logon_logon",
                    pc_id="PC-015"
                )
            )
        return scenario
