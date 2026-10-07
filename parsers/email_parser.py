import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional
import pandas as pd

from core.models import CommunicationMessage


class EmailMessageParser:
    """
    Parser for corporate email communications:
    MIME RFC 822 raw messages and CERT r4.2 email.csv format.
    Performs reply-stripping, filtering of system service mailboxes, and validation.
    """

    EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,4}")
    SYSTEM_ACCOUNTS = {
        "mailer-daemon", "postmaster", "no-reply", "noreply",
        "notifications", "calendar-notification", "root"
    }
    QUOTE_PATTERNS = [
        re.compile(r"-{3,}\s*Original Message\s*-{3,}", re.IGNORECASE),
        re.compile(r"-{3,}\s*Forwarded by\s*-{3,}", re.IGNORECASE),
        re.compile(r"From:.*?Sent:.*?To:", re.IGNORECASE | re.DOTALL),
        re.compile(r"^>+.*$", re.MULTILINE),
    ]

    def __init__(self, min_body_length: int = 5):
        self.min_body_length = min_body_length

    def normalize_email(self, raw_address: Any) -> Optional[str]:
        if not raw_address or pd.isna(raw_address):
            return None
        match = self.EMAIL_REGEX.search(str(raw_address))
        if match:
            return match.group(0).lower().strip()
        cleaned = str(raw_address).strip().lower()
        return cleaned if cleaned else None

    def parse_recipient_list(self, raw_recipients: Any) -> List[str]:
        if not raw_recipients or pd.isna(raw_recipients):
            return []
        raw_list = re.split(r"[,;]", str(raw_recipients))
        valid = []
        for r in raw_list:
            norm = self.normalize_email(r)
            if norm and norm not in valid:
                valid.append(norm)
        return valid

    def clean_quoted_text(self, body: str) -> str:
        if not body or pd.isna(body):
            return ""
        text = str(body)
        for pattern in self.QUOTE_PATTERNS:
            text = pattern.split(text)[0]
        cleaned_lines = [
            line for line in text.splitlines()
            if not line.strip().startswith(">")
        ]
        return "\n".join(cleaned_lines).strip()

    def parse_timestamp(self, raw_ts: Any) -> Optional[datetime]:
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

    def parse_cert_email_row(self, row: Dict[str, Any]) -> Optional[CommunicationMessage]:
        sender = self.normalize_email(row.get("from") or row.get("user"))
        if not sender or any(acc in sender for acc in self.SYSTEM_ACCOUNTS):
            return None
        to_list = self.parse_recipient_list(row.get("to"))
        cc_list = self.parse_recipient_list(row.get("cc"))
        bcc_list = self.parse_recipient_list(row.get("bcc"))
        all_receivers = list(dict.fromkeys(to_list + cc_list + bcc_list))
        if not all_receivers:
            return None
        ts = self.parse_timestamp(row.get("date"))
        if not ts:
            return None
        content = str(row.get("content", ""))
        cleaned = self.clean_quoted_text(content)
        if len(cleaned) < self.min_body_length:
            return None
        return CommunicationMessage(
            sender_id=sender,
            receiver_ids=all_receivers,
            timestamp=ts,
            channel="email",
            subject=str(row.get("subject", "")),
            body=content,
            cleaned_body=cleaned
        )

    def stream_cert_emails(self, filepath: str, chunk_size: int = 50000) -> Iterator[CommunicationMessage]:
        """
        Generator for streaming messages from CERT email.csv with low memory usage.
        """
        for chunk in pd.read_csv(filepath, chunksize=chunk_size, dtype=str):
            for _, row in chunk.iterrows():
                msg = self.parse_cert_email_row(row.to_dict())
                if msg and msg.is_valid:
                    yield msg