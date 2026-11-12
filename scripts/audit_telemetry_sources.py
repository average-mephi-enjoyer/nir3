import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from parsers.cert_telemetry_parser import TelemetryLogProcessor

processor = TelemetryLogProcessor()
report = {}
for source, stream in (("device", processor.stream_device_events), ("file", processor.stream_file_events), ("logon", processor.stream_logon_events)):
    path = str(Path("archive/r4.2") / f"{source}.csv")
    count = sum(1 for _ in stream(path))
    report[source] = {"status": processor.source_status[path], "valid_rows": count, "invalid_rows": processor.invalid_rows.get(path, 0)}
    print(source, report[source], flush=True)
Path("output/telemetry_source_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
