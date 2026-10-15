import csv
from datetime import datetime, timezone
import os
import re
from typing import Any, Iterator, List, Optional
from core.models import CommunicationMessage


class EmailMessageParser:
    DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"
    EMAIL_REGEX = re.compile(r"[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}")
    
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

    @classmethod
    def parse_timestamp(cls, raw_ts: Optional[str]) -> Optional[datetime]:
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

    def normalize_address(self, raw_address: Any) -> Optional[str]:
        if not raw_address:
            return None
        addr_str = str(raw_address).strip().lower()
        if not addr_str:
            return None
        match = self.EMAIL_REGEX.search(addr_str)
        if match:
            return match.group(0)
        return addr_str

    def parse_recipient_list(self, raw_recipients: Any) -> List[str]:
        if not raw_recipients:
            return []
        raw_list = re.split(r"[;,]", str(raw_recipients))
        valid_recipients: List[str] = []
        for r in raw_list:
            norm = self.normalize_address(r)
            if norm and norm not in valid_recipients:
                valid_recipients.append(norm)
        return valid_recipients

    def clean_quoted_text(self, body: Optional[str]) -> str:
        if not body:
            return ""
        text = str(body)
        for pattern in self.QUOTE_PATTERNS:
            text = pattern.split(text)[0]
        cleaned_lines = [
            line for line in text.splitlines()
            if not line.strip().startswith(">")
        ]
        return "\n".join(cleaned_lines).strip()

    def parse_cert_email_row(self, row: dict) -> Optional[CommunicationMessage]:
        raw_sender = row.get("from") or row.get("user")
        sender = self.normalize_address(raw_sender)
        if not sender or any(acc in sender for acc in self.SYSTEM_ACCOUNTS):
            return None
        user_login = (row.get("user") or "").strip().lower()
        to_list = self.parse_recipient_list(row.get("to"))
        cc_list = self.parse_recipient_list(row.get("cc"))
        bcc_list = self.parse_recipient_list(row.get("bcc"))
        all_receivers = list(dict.fromkeys(to_list + cc_list + bcc_list))
        if not all_receivers:
            return None
        ts = self.parse_timestamp(row.get("date"))
        if not ts:
            return None
        content = row.get("content") or ""
        cleaned = self.clean_quoted_text(content)
        if len(cleaned) < self.min_body_length:
            return None
        return CommunicationMessage(
            sender_id=sender,
            sender_user_id=user_login if user_login else None,
            receiver_ids=all_receivers,
            timestamp=ts,
            channel="email",
            subject=(row.get("subject") or "").strip(),
            body=content,
            cleaned_body=cleaned
        )

    def stream_cert_emails(self, filepath: str) -> Iterator[CommunicationMessage]:
        if not os.path.exists(filepath):
            return
        with open(filepath, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = self.parse_cert_email_row(row)
                if msg and msg.is_valid:
                    yield msg
