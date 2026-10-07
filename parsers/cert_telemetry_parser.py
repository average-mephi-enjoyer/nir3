from datetime import datetime, timezone
from typing import Any, Dict, Iterator, Optional
import pandas as pd

from core.models import TelemetryEvent


class TelemetryLogProcessor:
    """
    Parser for system telemetry of the operating environment (CERT r4.2):
    - logon.csv: logon/logoff events;
    - device.csv: external storage connect/disconnect (USB);
    - file.csv: file access and copy (File Copy, Exfiltration);
    - http.csv: network activity and external cloud resources.
    """

    def __init__(self):
        pass

    @staticmethod
    def parse_timestamp(raw_ts: Any) -> Optional[datetime]:
        if pd.isna(raw_ts) or not raw_ts:
            return None
        try:
            dt = pd.to_datetime(raw_ts, errors="coerce")
            if pd.isna(dt):
                return None
            if dt.tzinfo is None:
                dt = dt.tz_localize(timezone.utc)
            else:
                dt = dt.tz_convert(timezone.utc)
            return dt.to_pydatetime()
        except Exception:
            return None

    def stream_logon_events(self, filepath: str, chunk_size: int = 50000) -> Iterator[TelemetryEvent]:
        """
        CERT logon.csv streaming parser: id, date, user, pc, activity.
        """
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=str):
            for _, row in chunk.iterrows():
                ts = self.parse_timestamp(row.get("date"))
                user = str(row.get("user", "")).strip().lower()
                if not ts or not user:
                    continue

                activity = str(row.get("activity", "")).strip().lower()
                yield TelemetryEvent(
                    event_id=str(row.get("id", "")),
                    user_id=user,
                    timestamp=ts,
                    event_type=f"logon_{activity}",
                    pc_id=str(row.get("pc", "")).strip(),
                    target_resource=None,
                    activity_details={"activity": activity}
                )

    def stream_device_events(self, filepath: str, chunk_size: int = 50000) -> Iterator[TelemetryEvent]:
        """
        CERT device.csv streaming parser: id, date, user, pc, activity (Connect / Disconnect).
        """
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=str):
            for _, row in chunk.iterrows():
                ts = self.parse_timestamp(row.get("date"))
                user = str(row.get("user", "")).strip().lower()
                if not ts or not user:
                    continue
                activity = str(row.get("activity", "")).strip().lower()
                yield TelemetryEvent(
                    event_id=str(row.get("id", "")),
                    user_id=user,
                    timestamp=ts,
                    event_type=f"device_{activity}",
                    pc_id=str(row.get("pc", "")).strip(),
                    target_resource="removable_usb_drive",
                    activity_details={"activity": activity}
                )

    def stream_file_events(self, filepath: str, chunk_size: int = 50000) -> Iterator[TelemetryEvent]:
        """
        CERT file.csv streaming parser: id, date, user, pc, filename, content.
        """
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=str):
            for _, row in chunk.iterrows():
                ts = self.parse_timestamp(row.get("date"))
                user = str(row.get("user", "")).strip().lower()
                if not ts or not user:
                    continue
                filename = str(row.get("filename", "")).strip()
                yield TelemetryEvent(
                    event_id=str(row.get("id", "")),
                    user_id=user,
                    timestamp=ts,
                    event_type="file_transfer",
                    pc_id=str(row.get("pc", "")).strip(),
                    target_resource=filename,
                    activity_details={
                        "filename": filename,
                        "file_depth": filename.count("\\") + filename.count("/")
                    }
                )