"""Replay measured review scenarios after hardening; never edit measured metrics."""
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.baseline_comparison import BaselineBenchmarkRunner
from benchmark.drift_generator import DriftInjector
from run_stress_test import review_for

directory = ROOT / "output/final_stress"
frame = pd.read_csv(directory / "metrics_all_all.csv")
parameters = json.loads((directory / "selected_parameters.json").read_text())["parameters"]
cases = frame[frame.scenario == "false_alert_then_attack"]
checked, mismatches = 0, []
for keys, group in cases.groupby(["split", "experiment", "drift_days", "intensity", "seed"]):
    partition, experiment, days, intensity, seed = keys
    args = DriftInjector().generate_feature_scenario("false_alert_then_attack", duration_days=180,
        drift_days=int(days), intensity=float(intensity), seed=int(seed))
    rows, _ = BaselineBenchmarkRunner.evaluate_stream(*args, **parameters, experiment=experiment,
        review_schedule=review_for("false_alert_then_attack", args[0], args[2], args[1]))
    for result in rows:
        original = group[group.method_name == result.method_name].iloc[0]
        for name, value in asdict(result).items():
            if isinstance(value, (int, float)) and not np.isclose(original[name], value, equal_nan=True, rtol=1e-10, atol=1e-10):
                mismatches.append({"case": str(keys), "method": result.method_name, "metric": name})
        checked += 1
report = {"replayed_method_results": checked, "numeric_mismatches": mismatches,
    "current_profile_sha256": hashlib.sha256((ROOT / "profiling/profile.py").read_bytes()).hexdigest(),
    "measurement_source_snapshot": "output/measured_sources_before_review_hardening.zip",
    "scope": "all scenarios with scheduled review; other scenarios fit independently and have no review"}
(ROOT / "output/review_replay_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(bool(mismatches))
