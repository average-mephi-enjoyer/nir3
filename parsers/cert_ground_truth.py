import csv
from dataclasses import dataclass, field
from pathlib import Path
from parsers.email_parser import EmailMessageParser


@dataclass
class IncidentTruth:
    user: str
    scenario: str
    start: object
    end: object
    first_action: object = None
    first_action_id: str = ""
    first_action_type: str = ""
    email_ids: set = field(default_factory=set)
    details: str = ""
    label_source: str = "CERT r4.2 answer key"
    label_quality: str = "annotated_event; interval membership alone is unknown"

    def label(self, message):
        if message.message_id in self.email_ids:
            return 1
        if self.start <= message.timestamp <= self.end:
            return -1
        return 0


class UserIncidents(list):
    """Keep every scenario; singular attributes are only safe for a single scenario."""
    def label(self, message):
        labels = [incident.label(message) for incident in self]
        return 1 if 1 in labels else -1 if -1 in labels else 0

    def __getattr__(self, name):
        if len(self) != 1:
            raise ValueError("Select an individual scenario before reading its boundaries")
        return getattr(self[0], name)


def load_cert_ground_truth(answers_filepath):
    path = Path(answers_filepath)
    result = {}
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if row["dataset"] != "4.2":
                continue
            user = row["user"].lower()
            incident = IncidentTruth(user, row["scenario"], EmailMessageParser.parse_timestamp(row["start"]),
                                     EmailMessageParser.parse_timestamp(row["end"]), details=row["details"])
            if incident.start is None or incident.end is None or incident.end < incident.start:
                raise ValueError(f"Invalid incident interval: {user}")
            details = path.parent / f"r4.2-{incident.scenario}" / incident.details
            with details.open(encoding="utf-8", newline="") as observables:
                for event in csv.reader(observables):
                    if len(event) < 4:
                        raise ValueError(f"Invalid observable in {details}")
                    kind, event_id, raw_timestamp, actor = event[:4]
                    timestamp = EmailMessageParser.parse_timestamp(raw_timestamp)
                    if timestamp is None:
                        raise ValueError(f"Invalid timestamp in {details}")
                    if kind == "email" and actor.lower() == user:
                        incident.email_ids.add(event_id)
                    if kind in {"file", "device", "http", "logon"} and actor.lower() == user:
                        if incident.first_action is None or timestamp < incident.first_action:
                            incident.first_action = timestamp
                            incident.first_action_id = event_id
                            incident.first_action_type = kind
            result.setdefault(user, UserIncidents()).append(incident)
    return result


def export_ground_truth(truth, users, output):
    fields = ["user", "scenario", "start", "first_action", "end", "first_action_id", "first_action_type", "details", "label_source", "label_quality"]
    with Path(output).open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for user in users:
            for incident in truth.get(user.lower(), []):
                writer.writerow({key: getattr(incident, key) for key in fields})
