import argparse
import csv
import json
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from benchmark.baseline_comparison import BaselineBenchmarkRunner
from benchmark.cert_pipeline import extract_cohort, prepare_user, stratified_users, validate_archive
from benchmark.reporting import save_manifest, write_event_log
from core.models import RiskSignal
from parsers.cert_ground_truth import load_cert_ground_truth, export_ground_truth
from parsers.cert_telemetry_parser import TelemetryLogProcessor
from parsers.role_directory import RoleDirectory
from verification.log_correlation import TargetedRetrospectiveVerifier


def write_rows(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cert_cohort(archive_dir="archive", users=None, output_dir="output/cert_local_llm", use_conformal=False,
                    semantic_mode="local_llm", seed=20261112, controls_per_role=2):
    validate_archive(archive_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        previous_mode = json.loads((output / "manifest.json").read_text(encoding="utf-8"))["config"].get("semantic_mode")
        if previous_mode != semantic_mode:
            raise ValueError("Output already contains another semantic mode; choose a separate --output directory")
    truth = load_cert_ground_truth(Path(archive_dir) / "answers/insiders.csv")
    directory = RoleDirectory(Path(archive_dir) / "r4.2/LDAP")
    selected = {u for group in stratified_users(directory.snapshots[0], controls_per_role, seed + 1).values() for u in group}
    targets = sorted(set(users) if users is not None else set(truth) | selected)
    export_ground_truth(truth, sorted(truth), output / "cert_ground_truth.csv")
    print(f"CERT: {len(targets)} users; {sum(len(v) for v in truth.values())} annotated scenarios", flush=True)
    features, peers, boundaries, metadata = extract_cohort(archive_dir, targets, semantic_mode, seed)
    metadata.update({"users": targets, "controls_per_role": controls_per_role, "control_selection_seed": seed + 1,
                     "control_selection": "role-stratified before excluding any incident labels; labels applied only in evaluation"})
    save_manifest(output / "manifest.json", metadata)
    verifier = TargetedRetrospectiveVerifier()
    processor = TelemetryLogProcessor()
    for source, stream in (("device", processor.stream_device_events), ("file", processor.stream_file_events), ("logon", processor.stream_logon_events)):
        path = Path(archive_dir) / "r4.2" / f"{source}.csv"
        verifier.register_source(source, "available" if path.exists() else "missing")
        for event in stream(str(path)):
            if event.user_id in targets:
                verifier.register_telemetry_event(event)
        verifier.register_source(source, processor.source_status[str(path)])
    verifier.register_source("http", "not_loaded" if (Path(archive_dir) / "r4.2/http.csv").exists() else "missing")
    rows, status, last_result = [], [], None
    for user in targets:
        incidents = truth.get(user, [])
        try:
            training, calibration, testing, prior, role_info = prepare_user(user, features[user], peers, boundaries, incidents)
            destination = output / user
            destination.mkdir(exist_ok=True)
            (destination / "roles.json").write_text(json.dumps(role_info, indent=2), encoding="utf-8")
            for incident_index, incident in enumerate(incidents or [None]):
                scenario = f"{incident.scenario}_{incident_index}" if incident else "control"
                for experiment in ("A", "B"):
                    metrics, traces = BaselineBenchmarkRunner.evaluate_stream(training, calibration, testing, prior,
                        incident.start if incident else None, incident.first_action if incident else None,
                        incident.end if incident else None, use_conformal=use_conformal, experiment=experiment,
                        exposure_end=boundaries[-1] + timedelta(seconds=1), scenario_intervals=[(i.start, i.end) for i in incidents])
                    write_event_log(destination / f"{scenario}_{experiment}_events.csv.gz", testing, traces)
                    (destination / f"{scenario}_{experiment}_comparison.md").write_text(BaselineBenchmarkRunner.format_markdown_table(metrics), encoding="utf-8")
                    rows.extend({"user": user, "scenario": scenario, "role": training[0].role, **asdict(row)} for row in metrics)
                    if experiment == "A":
                        audits = [asdict(verifier.verify_alert_signal(RiskSignal(user, o.timestamp, o.markers, {}))) for o in traces["Protected"].alerts]
                        (destination / f"{scenario}_audit.json").write_text(json.dumps(audits, default=str, indent=2), encoding="utf-8")
                        last_result = metrics, traces, testing
                status.append({"user": user, "scenario": scenario, "status": "evaluated", "reason": ""})
            print(f"CERT evaluated {user}: {len(testing)} test observations", flush=True)
        except Exception as error:
            status.append({"user": user, "scenario": ",".join(i.scenario for i in incidents) or "control",
                           "status": "not_evaluable" if str(error).startswith("not_evaluable") else "error",
                           "reason": f"{type(error).__name__}: {error}"})
            print(f"CERT {user}: {status[-1]['reason']}", flush=True)
        write_rows(output / "metrics.csv", rows)
        write_rows(output / "status.csv", status)
    (output / "limitations.json").write_text(json.dumps({"http": verifier.source_status["http"],
        "local_llm_validated": False, "cross_user_generalization": False, "source_semantics": "see archive/r4.2/readme.txt",
        "technical_verification_ablation": "not evaluated: independent quality validation unavailable",
        "scenario_count": sum(len(v) for v in truth.values()), "status": status}, indent=2), encoding="utf-8")
    if rows:
        import pandas as pd
        frame = pd.DataFrame(rows)
        summary = frame.groupby(["experiment", "method_name"])[["detected_scenario", "contamination_rate", "available_fraction", "false_freezes"]].mean()
        summary.to_csv(output / "summary.csv", na_rep="N/A")
    return last_result, status


def run_cert_dataset_evaluation(archive_dir="archive", target_user="aam0658", use_mock_llm=False,
                                output_dir="output/cert_local_llm", use_conformal=False):
    result, statuses = run_cert_cohort(archive_dir, [target_user.lower()], output_dir, use_conformal,
                                       "heuristic" if use_mock_llm else "local_llm")
    if result is None:
        raise ValueError(statuses)
    return result


def run_synthetic_benchmark(scenario_type="ramp"):
    from run_stress_test import run_suite
    return run_suite(scenario_type=scenario_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible insider-risk research prototype")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--user", default="aam0658")
    parser.add_argument("--all-users", action="store_true", help="all annotated scenarios plus seeded role-stratified controls")
    parser.add_argument("--archive", default="archive")
    parser.add_argument("--output", default="output/cert_local_llm")
    parser.add_argument("--conformal", action="store_true", help="experimental empirical e-process; no anytime guarantee")
    parser.add_argument("--semantic-mode", default="local_llm", choices=["heuristic", "local_llm", "disabled"])
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--controls-per-role", type=int, default=2)
    parser.add_argument("--scenario", default="ramp")
    args = parser.parse_args()
    if args.dataset:
        try:
            _, statuses = run_cert_cohort(args.archive, None if args.all_users else [args.user.lower()], args.output,
                                          args.conformal, args.semantic_mode, args.seed, args.controls_per_role)
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            parser.exit(2, f"CERT evaluation failed: {error}\n")
        if any(s["status"] == "error" for s in statuses):
            parser.exit(1, "Some users failed; see status.csv.\n")
    else:
        run_synthetic_benchmark(args.scenario)
