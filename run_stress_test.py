import argparse
import csv
from dataclasses import asdict
from pathlib import Path
from benchmark.drift_generator import DriftInjector
from benchmark.baseline_comparison import BaselineBenchmarkRunner


def run_suite(scenario_type="all", split="test", output_dir="output/stress"):
    from main import write_event_log
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    kinds = ("ramp", "step", "adaptive", "role_change", "stress", "burnout", "normal") if scenario_type == "all" else (scenario_type,)
    partitions = ("tune", "test") if split == "all" else (split,)
    results = []
    for partition in partitions:
        seeds = (11, 12) if partition == "tune" else (21, 22)
        for kind in kinds:
            settings = ((days, intensity) for days in (30, 90) for intensity in (1.0, 3.0, 6.0))
            for days, intensity in settings:
                for seed in seeds:
                    training, calibration, testing, priors, start, action = DriftInjector(seed).generate_feature_scenario(
                        kind, duration_days=180, drift_days=days, intensity=intensity, seed=seed)
                    rows, traces = BaselineBenchmarkRunner.evaluate_stream(training, calibration, testing, priors, start, action)
                    name = f"{partition}_{kind}_days{days}_intensity{intensity}_seed{seed}"
                    destination = output / name
                    destination.mkdir(exist_ok=True)
                    write_event_log(destination / "events.csv", testing, traces)
                    (destination / "comparison.md").write_text(BaselineBenchmarkRunner.format_markdown_table(rows), encoding="utf-8")
                    for row in rows:
                        results.append({"split": partition, "scenario": kind, "drift_days": days,
                                        "intensity": intensity, "seed": seed, **asdict(row)})
    with (output / f"metrics_{split}_{scenario_type}.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} method evaluations in {output}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all", choices=["all", "ramp", "step", "adaptive", "role_change", "stress", "burnout", "normal"])
    parser.add_argument("--split", default="test", choices=["tune", "test", "all"])
    parser.add_argument("--output", default="output/stress")
    args = parser.parse_args()
    run_suite(args.scenario, args.split, args.output)
