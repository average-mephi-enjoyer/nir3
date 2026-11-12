import json
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.reporting import aggregate_results
directory = ROOT / "output/final_stress"
config = json.loads((directory / "manifest.json").read_text())["config"]
aggregate_results(pd.read_csv(directory / "metrics_all_all.csv").to_dict("records"), directory,
                  config["bootstrap_seed"], config["bootstrap_repetitions"])
print("Rebuilt measured tables and figures")
