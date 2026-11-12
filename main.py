import argparse
import csv
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import numpy as np
from benchmark.baseline_comparison import BaselineBenchmarkRunner, FeatureObservation
from benchmark.visualizer import BenchmarkVisualizer
from core.models import MarkerVector, RiskSignal
from extractors.graph_topology import DynamicCommunicationGraph
from extractors.llm_semantic import LLMSemanticExtractor
from extractors.spacy_statistical import SpacyStatisticalExtractor
from parsers.cert_ground_truth import load_cert_ground_truth, export_ground_truth
from parsers.role_directory import RoleDirectory
from parsers.cert_telemetry_parser import TelemetryLogProcessor
from parsers.email_parser import EmailMessageParser
from profiling.profile import ContaminationResistantProfile
from profiling.sliding_window import SlidingTimeWindow
from profiling.workload_filter import TeamWorkloadContextTracker
from verification.log_correlation import TargetedRetrospectiveVerifier


CERT_USERS = ("aam0658", "ajr0932", "btl0226", "cah0936", "aaf0535", "abc0174", "ccl0068", "bbs0039", "csc0217")


def collect_cert_features(archive_dir, target_user, truth, use_mock_llm=True):
    archive = Path(archive_dir)
    directory = RoleDirectory(archive / "r4.2" / "LDAP")
    parser = EmailMessageParser()
    statistical = SpacyStatisticalExtractor()
    semantic = LLMSemanticExtractor(force_offline_mode=use_mock_llm)
    workload = TeamWorkloadContextTracker()
    graphs = {}
    windows, references, observations, reference_vectors = {}, set(), [], {}
    verifier = TargetedRetrospectiveVerifier()
    start, train_end, calibration_end, test_start, previous = None, None, None, None, None
    initial_role = None
    try:
        for message in parser.stream_cert_emails(str(archive / "r4.2" / "email.csv")):
            timestamp = message.timestamp
            if previous is not None and timestamp < previous:
                raise ValueError("email.csv must be chronologically ordered")
            previous = timestamp
            if start is None:
                start = timestamp
                train_end = start + timedelta(days=30)
                calibration_end = train_end + timedelta(days=30)
                test_start = calibration_end + timedelta(days=14)
                roster = directory.at(start)
                if target_user not in roster:
                    raise ValueError(f"Unknown employee: {target_user}")
                initial_role = roster[target_user]["role"]
                held_out = set(sorted(user for user in roster if user not in truth)[-5:])
                eligible = sorted(user for user, info in roster.items() if info["role"] == initial_role
                                  and user not in truth and user != target_user and user not in held_out)
                references = set(eligible[:5])
                if len(references) < 2:
                    raise ValueError(f"Insufficient independent reference employees for role {initial_role}")
                for user in references | {target_user}:
                    windows[user] = SlidingTimeWindow(user, roster[user]["role"], roster[user]["email"])
                    graphs[user] = DynamicCommunicationGraph()
                    for peer, info in roster.items():
                        graphs[user].register_alias(peer, info["email"])
            sender = (message.sender_user_id or "").lower()
            sender_info = directory.user(sender, timestamp)
            role = sender_info.get("role", "unknown")
            workload.register_role_activity(role, timestamp, employee_id=sender)
            recipients = {recipient.lower() for recipient in message.receiver_ids}
            involved = []
            for user, window in windows.items():
                if user in references and timestamp >= train_end:
                    continue
                info = directory.user(user, timestamp)
                window.role = info.get("role", "unknown")
                window.email = info.get("email", window.email).lower()
                if sender == user or window.email in recipients:
                    window.add_message(message)
                    involved.append(user)
            if not involved:
                continue
            for user in involved:
                graphs[user].add_communication(message)
            if target_user in involved:
                verifier.register_communication_message(message, user_id=target_user)
            if sender not in involved:
                continue
            window = windows[sender]
            context = graphs[sender].update_context(window.export_context())
            stat_vector = statistical.compute_markers(message, context)
            semantic_vector = semantic.compute_markers(message, context)
            values = {marker: getattr(semantic_vector if marker.startswith("llm_") else stat_vector, marker)
                      for marker in MarkerVector.feature_names()}
            vector = MarkerVector(**values, llm_uncertainty=semantic_vector.llm_uncertainty)
            if sender in references:
                reference_vectors.setdefault(window.role, []).append(vector)
            else:
                incident = truth.get(target_user)
                label = incident.label(message) if incident else 0
                factor = workload.get_role_workload_factor(window.role, timestamp, exclude_user=target_user)
                observations.append(FeatureObservation(timestamp, vector, label, factor, message.message_id or ""))
    finally:
        semantic.close()
    if start is None:
        raise ValueError("No usable emails")
    incident = truth.get(target_user)
    if incident and incident.start < test_start:
        raise ValueError("Incident overlaps the fixed training/calibration period")
    training = [o for o in observations if o.timestamp < train_end]
    calibration = [o for o in observations if train_end <= o.timestamp < test_start]
    testing = [o for o in observations if o.timestamp >= test_start]
    peers = reference_vectors.get(initial_role, [])
    if len(peers) < 2:
        raise ValueError("Insufficient independent role observations")
    matrix = np.asarray([vector.to_numpy() for vector in peers])
    role_priors = {marker: {"mean": float(matrix[:, i].mean()),
                           "variance": float(max(matrix[:, i].var(ddof=1), 1e-4))}
                   for i, marker in enumerate(MarkerVector.feature_names())}
    metadata = {"training_end": str(train_end), "calibration_end": str(calibration_end),
                "test_start": str(test_start), "reference_users": ",".join(sorted(references)),
                "role": initial_role, "training_observations": len(training),
                "calibration_observations": len(calibration), "test_observations": len(testing),
                "semantic_mode": "lexical_fallback" if use_mock_llm else "local_llm_requested_with_fallback",
                "response_marker": "unavailable_without_reply_link"}
    return training, calibration, testing, role_priors, verifier, graphs[target_user], metadata


def write_event_log(path, observations, traces):
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["method", "message_id", "timestamp", "label", "risk", "alert", "sequential_score", "baseline"])
        for method, trace in traces.items():
            alerts = {id(observation) for observation in trace.alerts}
            for i, observation in enumerate(observations):
                writer.writerow([method, observation.observation_id, observation.timestamp, observation.label,
                                 trace.scores[i], id(observation) in alerts, trace.sequential_scores[i], trace.baselines[i]])
    with path.with_name("admissions.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["method", "message_id", "timestamp", "reason", "risk", "horizon_risk"])
        for method, trace in traces.items():
            decisions = trace.profile.quarantine.decisions if method in ("Protected", "EMA quarantine") else trace.admissions
            for observation in decisions:
                writer.writerow([method, observation.observation_id, observation.timestamp, getattr(observation, "reason", "updated"),
                                 getattr(observation, "risk_score", ""), getattr(observation, "horizon_risk", "")])


def run_cert_dataset_evaluation(archive_dir="archive", target_user="aam0658", use_mock_llm=True,
                                output_dir="output", use_conformal=False):
    target_user = target_user.lower()
    output = Path(output_dir) / target_user
    output.mkdir(parents=True, exist_ok=True)
    truth = load_cert_ground_truth(Path(archive_dir) / "answers" / "insiders.csv")
    export_ground_truth(truth, CERT_USERS, Path(output_dir) / "cert_ground_truth.csv")
    training, calibration, testing, priors, verifier, graph, metadata = collect_cert_features(archive_dir, target_user, truth, use_mock_llm)
    incident = truth.get(target_user)
    rows, traces = BaselineBenchmarkRunner.evaluate_stream(training, calibration, testing, priors,
                    incident.start if incident else None, incident.first_action if incident else None,
                    incident.end if incident else None, use_conformal=use_conformal)
    table = BaselineBenchmarkRunner.format_markdown_table(rows)
    (output / "baseline_comparison.md").write_text(table, encoding="utf-8")
    write_event_log(output / "events.csv", testing, traces)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    telemetry = TelemetryLogProcessor()
    for filename, stream in (("device.csv", telemetry.stream_device_events), ("file.csv", telemetry.stream_file_events)):
        for event in stream(str(Path(archive_dir) / "r4.2" / filename)):
            if event.user_id == target_user:
                verifier.register_telemetry_event(event)
    trace = traces["Protected"]
    audit = []
    for observation in trace.alerts:
        signal = RiskSignal(target_user, observation.timestamp, observation.markers, {})
        result = verifier.verify_alert_signal(signal)
        audit.append(f"- {observation.timestamp}: {result.explanation}")
    metadata_text = "\n".join(f"- {key}: {value}" for key, value in metadata.items())
    report = (f"# CERT {target_user}\n\n{metadata_text}\n\n"
              f"- Profile state: {trace.profile.state}\n"
              "- Sequential control: empirical; no proven anytime FPR guarantee.\n"
              "- First technical action: earliest annotated non-email observable, not necessarily exfiltration.\n"
              "- Technical verdicts remain exploratory pending checklist items 16–17.\n\n"
              + "\n".join(audit) + "\n\n" + table)
    (output / "incident_report.md").write_text(report, encoding="utf-8")
    histories = {marker: [getattr(o.markers, marker) for o in testing] for marker in MarkerVector.feature_names()}
    baselines = {marker: [baseline[marker] for baseline in trace.baselines] for marker in MarkerVector.feature_names()}
    steps = list(range(len(testing)))
    BenchmarkVisualizer.plot_all_markers_grid(steps, histories, baselines, str(output / "cert_all_markers.png"))
    BenchmarkVisualizer.plot_top_markers_dynamics(steps, histories, baselines,
                                                 ["activity_marker", "social_marker", "pronoun_marker"],
                                                 str(output / "cert_top_markers.png"))
    BenchmarkVisualizer.plot_communication_graph(graph.build_ego_graph(target_user), target_user, str(output / "cert_ego_graph.png"))
    print(table)
    return rows, traces, testing


def run_synthetic_benchmark(scenario_type="ramp"):
    from run_stress_test import run_suite
    return run_suite(scenario_type=scenario_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insider Risk Early Detection Pipeline")
    parser.add_argument("--dataset", action="store_true")
    parser.add_argument("--user", default="aam0658")
    parser.add_argument("--all-users", action="store_true")
    parser.add_argument("--archive", default="archive")
    parser.add_argument("--output", default="output")
    parser.add_argument("--conformal", action="store_true")
    parser.add_argument("--scenario", default="ramp", choices=["ramp", "stress", "all"])
    args = parser.parse_args()
    if args.dataset:
        users = list(CERT_USERS) if args.all_users else [args.user]
        if args.all_users:
            directory = RoleDirectory(Path(args.archive) / "r4.2" / "LDAP")
            truth = load_cert_ground_truth(Path(args.archive) / "answers" / "insiders.csv")
            eligible = sorted(user for user in directory.snapshots[0] if user not in truth)
            users.extend(eligible[-5:])
        for user in users:
            run_cert_dataset_evaluation(args.archive, user, output_dir=args.output, use_conformal=args.conformal)
    else:
        run_synthetic_benchmark(args.scenario)
