import csv
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path


class RoleDirectory:
    def __init__(self, directory):
        self.dates = []
        self.snapshots = []
        for path in sorted(Path(directory).glob("*.csv")):
            timestamp = datetime.strptime(path.stem, "%Y-%m").replace(tzinfo=timezone.utc)
            with path.open(encoding="utf-8", newline="") as source:
                self.snapshots.append({row["user_id"].lower(): row for row in csv.DictReader(source)})
            self.dates.append(timestamp)
        if not self.dates:
            raise ValueError("LDAP snapshots are required for role profiling")

    def at(self, timestamp):
        position = bisect_right(self.dates, timestamp) - 1
        return self.snapshots[position] if position >= 0 else {}

    def user(self, user_id, timestamp):
        return self.at(timestamp).get(user_id.lower(), {})
