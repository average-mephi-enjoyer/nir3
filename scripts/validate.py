import json
import subprocess
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
output = ROOT / "output"
output.mkdir(exist_ok=True)
with (output / "tests.log").open("w", encoding="utf-8") as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
checks = []
for command in ([sys.executable, "-m", "ruff", "check", "."], [sys.executable, "scripts/check_source.py"],
                [sys.executable, "main.py", "--help"], [sys.executable, "main.py", "--dataset", "--all-users", "--archive", "output/absent_cert"]):
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    checks.append({"command": command, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr,
                   "expected_nonzero": "output/absent_cert" in command})
report = {"tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
          "skipped": len(result.skipped), "checks": checks}
(output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(0 if result.wasSuccessful() and all((c["exit_code"] != 0) if c["expected_nonzero"] else c["exit_code"] == 0 for c in checks) else 1)
