import argparse
import csv
import gzip
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from benchmark.drift_generator import DriftInjector
from benchmark.baseline_comparison import BaselineBenchmarkRunner
from benchmark.reporting import aggregate_results, save_manifest

CONFIG_PATH = Path(__file__).parent / "configs/experiments.json"


def review_for(kind, training, testing, calibration):
    if kind != "false_alert_then_attack":
        return []
    # Scripted external review at day 15, specified without access to detector output or labels.
    from profiling.profile import ContaminationResistantProfile
    reviewed = ContaminationResistantProfile()
    reviewed.fit([o.markers for o in training])
    scores = [reviewed.compute_deviations(o.markers)[1] for o in calibration]
    return [{"timestamp": testing[15].timestamp, "training": [o.markers for o in training],
             "calibration_scores": scores, "approved": True,
             "source": "controlled exercise: reuse independently verified pre-exercise training"}]


def tune_parameters(config, output):
    candidates = [("default", dict(config["parameters"]))]
    for key, values in config["sensitivity"].items():
        candidates.extend((f"{key}={value}", {**config["parameters"], key: value}) for value in values)
    measurements = []
    for candidate, parameters in candidates:
        for kind in config["scenarios"]:
            for seed in range(config["tune_seed_start"], config["tune_seed_start"] + config["repetitions"]):
                args = DriftInjector().generate_feature_scenario(kind, seed=seed, duration_days=config["duration_days"], drift_days=90)
                rows, _ = BaselineBenchmarkRunner.evaluate_stream(*args, **parameters, methods=["Protected"],
                                                                 review_schedule=review_for(kind, args[0], args[2], args[1]))
                row = asdict(rows[0])
                # Predeclared loss, not post-hoc selection on test scenarios.
                loss = (1 - row["detected_scenario"] if np.isfinite(row["detected_scenario"]) else 0.)
                loss += row["contamination_rate"] / 100 if np.isfinite(row["contamination_rate"]) else 0.
                loss += row["false_alert_episodes"] + (1 - row["available_fraction"])
                measurements.append({"candidate": candidate, "scenario": kind, "seed": seed, "loss": loss, **row})
        print(f"Tune completed: {candidate}", flush=True)
    scores = {name: np.mean([r["loss"] for r in measurements if r["candidate"] == name]) for name, _ in candidates}
    selected_name, selected = min(candidates, key=lambda pair: scores[pair[0]])
    write_csv(output / "sensitivity_tune.csv", measurements)
    (output / "selected_parameters.json").write_text(json.dumps({"candidate": selected_name, "parameters": selected,
        "tune_loss": scores[selected_name], "criterion": "miss + contamination_fraction + false_episodes + unavailable_fraction",
        "selection_scope": "Protected; one-at-a-time candidates; fixed before test"}, indent=2), encoding="utf-8")
    return selected


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_trajectories(path, observations, traces):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "id", "timestamp", "label", "risk", "alarm", "state", "markers", "clean_markers", "mean", "variance"])
        for method, trace in traces.items():
            alerts = {id(o) for o in trace.alerts}
            for i, o in enumerate(observations):
                writer.writerow([method, o.observation_id, o.timestamp, o.label, trace.scores[i], id(o) in alerts,
                    trace.states[i], o.markers.to_numpy().tolist(), o.clean_markers.to_numpy().tolist(),
                    list(trace.baselines[i].values()), list(trace.variances[i].values())])
    with gzip.open(path.with_name(path.name.replace("events", "decisions")).with_suffix(".jsonl.gz"), "wt", encoding="utf-8") as stream:
        for method, trace in traces.items():
            for d in trace.profile.quarantine.decisions + list(trace.profile.quarantine.buffer):
                stream.write(json.dumps({"method": method, "id": d.observation_id, "risk": d.risk_score,
                    "threshold": d.threshold, "normalized_risk": d.normalized_risk, "horizon_risk": d.horizon_risk,
                    "status": d.status, "history": d.history}) + "\n")


def run_suite(scenario_type="all", split="test", output_dir="output/final_stress", config_path=CONFIG_PATH, repetitions=None, tune=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if repetitions is not None:
        if repetitions < 2:
            raise ValueError("At least two independent seeds required")
        config["repetitions"] = repetitions
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    kinds = config["scenarios"] if scenario_type == "all" else [scenario_type]
    partitions = ("tune", "test") if split == "all" else (split,)
    if tune or split == "all":
        parameters = tune_parameters(config, output)
    elif (output / "selected_parameters.json").exists():
        parameters = json.loads((output / "selected_parameters.json").read_text())["parameters"]
    else:
        parameters = config["parameters"]
    save_manifest(output / "manifest.json", {**config, "selected_parameters": parameters, "split": split, "scenario": scenario_type})
    results = []
    for partition in partitions:
        start_seed = config[f"{partition}_seed_start"]
        for experiment in ("A", "B"):
            for kind in kinds:
                settings = config[f"settings_{experiment}"]
                for days, intensity in settings:
                    for seed in range(start_seed, start_seed + config["repetitions"]):
                        args = DriftInjector().generate_feature_scenario(kind, duration_days=config["duration_days"], drift_days=days, intensity=intensity, seed=seed)
                        rows, traces = BaselineBenchmarkRunner.evaluate_stream(*args, **parameters, experiment=experiment,
                            review_schedule=review_for(kind, args[0], args[2], args[1]))
                        name = f"{partition}_{experiment}_{kind}_days{days}_intensity{intensity}_seed{seed}"
                        save_trajectories(output / f"{name}_events.csv.gz", args[2], traces)
                        for row in rows:
                            results.append({"split": partition, "scenario": kind, "drift_days": days,
                                "intensity": intensity, "rate_sd_per_day": intensity / days, "seed": seed,
                                **parameters, **asdict(row)})
                write_csv(output / f"metrics_{split}_{scenario_type}.csv", results)
                print(f"Completed {partition}/{experiment}/{kind}: {len(results)} method evaluations", flush=True)
    aggregate_results(results, output, config["bootstrap_seed"], config["bootstrap_repetitions"])
    print(f"Saved {len(results)} measured method evaluations in {output}", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--split", default="test", choices=["tune", "test", "all"])
    parser.add_argument("--output", default="output/final_stress")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    run_suite(args.scenario, args.split, args.output, args.config, args.repetitions, args.tune)
