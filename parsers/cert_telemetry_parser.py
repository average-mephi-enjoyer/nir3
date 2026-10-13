import csv
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional

from core.models import TelemetryEvent


class TelemetryLogProcessor:
    """
    Parser for system telemetry of the operating environment (CERT r4.2):
    - logon.csv: logon/logoff events;
    - device.csv: external storage connect/disconnect (USB);
    - file.csv: file access and exfiltration events;
    - http.csv: network activity and cloud storage accesses.
    """

    DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

    @classmethod
    def parse_timestamp(cls, raw_ts: Optional[str]) -> Optional[datetime]:
        """
        Fast scalar timestamp parser optimized for the CERT r4.2 datetime format.
        """
        if not raw_ts:
            return None
        cleaned_ts = raw_ts.strip()
        if not cleaned_ts:
            return None
        try:
            dt = datetime.strptime(cleaned_ts, cls.DATETIME_FORMAT)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def stream_logon_events(self, filepath: str) -> Iterator[TelemetryEvent]:
        """
        Stream events from CERT logon.csv (id, date, user, pc, activity).
        """
        with open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = self.parse_timestamp(row.get("date"))
                user = (row.get("user") or "").strip().lower()
                if not ts or not user:
                    continue
                raw_activity = (row.get("activity") or "").strip().lower()
                event_type = f"logon_{raw_activity}" if raw_activity else "logon_unknown"
                yield TelemetryEvent(
                    event_id=(row.get("id") or "").strip(),
                    user_id=user,
                    timestamp=ts,
                    event_type=event_type,
                    pc_id=(row.get("pc") or "").strip(),
                    target_resource=None,
                    activity_details={"activity": raw_activity}
                )

    def stream_device_events(self, filepath: str) -> Iterator[TelemetryEvent]:
        """
        Stream events from CERT device.csv (id, date, user, pc, activity).
        """
        with open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = self.parse_timestamp(row.get("date"))
                user = (row.get("user") or "").strip().lower()
                if not ts or not user:
                    continue
                raw_activity = (row.get("activity") or "").strip().lower()
                event_type = f"device_{raw_activity}" if raw_activity else "device_unknown"
                yield TelemetryEvent(
                    event_id=(row.get("id") or "").strip(),
                    user_id=user,
                    timestamp=ts,
                    event_type=event_type,
                    pc_id=(row.get("pc") or "").strip(),
                    target_resource="removable_usb_drive",
                    activity_details={"activity": raw_activity}
                )

    def stream_file_events(self, filepath: str) -> Iterator[TelemetryEvent]:
        """
        Stream events from CERT file.csv (id, date, user, pc, filename, content).
        """