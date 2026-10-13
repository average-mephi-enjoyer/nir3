import argparse
import csv
from datetime import datetime, timezone
import os
from typing import Dict, List, Optional
import numpy as np
from benchmark.drift_generator import DriftInjector
from benchmark.metrics import BenchmarkEvaluator
from benchmark.visualizer import BenchmarkVisualizer
from core.models import CommunicationMessage, MarkerVector, RiskSignal, TelemetryEvent
from extractors.graph_topology import DynamicCommunicationGraph
from extractors.llm_semantic import LLMSemanticExtractor
from extractors.spacy_statistical import SpacyStatisticalExtractor
from parsers.cert_telemetry_parser import TelemetryLogProcessor
from parsers.email_parser import EmailMessageParser
from profiling.profile import ContaminationResistantProfile
from profiling.sliding_window import SlidingTimeWindow
from profiling.workload_filter import TeamWorkloadContextTracker
from verification.log_correlation import TargetedRetrospectiveVerifier


def load_cert_ground_truth(answers_filepath: str) -> Dict[str, Dict[str, datetime]]:
    """
    Parse archive/answers/insiders.csv: scenario, user, start, end.
    """
    ground_truth: Dict[str, Dict[str, datetime]] = {}
    if not os.path.exists(answers_filepath):
        return ground_truth
    dt_format = "%m/%d/%Y %H:%M:%S"
    with open(answers_filepath, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user = (row.get("user") or "").strip().lower()
            start_str = (row.get("start") or "").strip()
            end_str = (row.get("end") or "").strip()
            if not user or not start_str:
                continue
            try:
                t_start = datetime.strptime(start_str, dt_format).replace(tzinfo=timezone.utc)
                t_end = datetime.strptime(end_str, dt_format).replace(tzinfo=timezone.utc) if end_str else None
                ground_truth[user] = {"start": t_start, "end": t_end}
            except Exception:
                continue
    return ground_truth


def run_cert_dataset_evaluation(
    archive_dir: str = "archive",
    target_user: str = "aam0658",
    use_mock_llm: bool = True
) -> None:
    """
    Stream real CERT r4.2 dataset files for the target subject.
    """
    email_path = os.path.join(archive_dir, "r4.2", "email.csv")
    file_path = os.path.join(archive_dir, "r4.2", "file.csv")
    device_path = os.path.join(archive_dir, "r4.2", "device.csv")
    answers_path = os.path.join(archive_dir, "answers", "insiders.csv")
    target_norm = target_user.strip().lower()
    print(f"[*] Initializing CERT r4.2 evaluation pipeline for employee '{target_norm}'...")
    gt = load_cert_ground_truth(answers_path)
    target_gt = gt.get(target_norm)
    t_drift_start = target_gt["start"] if target_gt else None
    t_first_action = target_gt["end"] if target_gt else None
    email_parser = EmailMessageParser()
    telem_parser = TelemetryLogProcessor()
    protected_profile = ContaminationResistantProfile(alpha_ema=0.05, quarantine_delay_days=14)
    verifier = TargetedRetrospectiveVerifier(retro_window_days=7, mismatch_threshold=0.5)
    workload_tracker = TeamWorkloadContextTracker()
    sliding_window = SlidingTimeWindow(employee_id=target_norm, role="general")
    comm_graph = DynamicCommunicationGraph(decay_window_days=30)
    statistical_extractor = SpacyStatisticalExtractor()
    llm_extractor = LLMSemanticExtractor(force_offline_mode=use_mock_llm)
    comm_graph.register_alias(target_norm, f"{target_norm}@")
    print("[*] Ingesting telemetry logs (device.csv, file.csv)...")
    if os.path.exists(device_path):
        for event in telem_parser.stream_device_events(device_path):
            if event.user_id == target_norm:
                verifier.register_telemetry_event(event)
    if os.path.exists(file_path):
        for event in telem_parser.stream_file_events(file_path):
            if event.user_id == target_norm:
                verifier.register_telemetry_event(event)
    print("[*] Streaming email.csv and calculating linguistic and graph markers...")
    time_steps: List[int] = []
    marker_histories: Dict[str, List[float]] = {m: [] for m in ContaminationResistantProfile.TRACKED_MARKERS}
    baseline_histories: Dict[str, List[float]] = {m: [] for m in ContaminationResistantProfile.TRACKED_MARKERS}
    risk_scores: List[float] = []
    alert_thresholds: List[float] = []
    first_alert_time: Optional[datetime] = None
    step = 0
    if os.path.exists(email_path):
        for msg in email_parser.stream_cert_emails(email_path):
            sender_uid = (msg.sender_user_id or "").lower()
            sender_email = msg.sender_id.lower()
            receivers = [r.lower() for r in msg.receiver_ids]
            is_target_sender = (sender_uid == target_norm or target_norm in sender_email)
            is_target_receiver = any(target_norm in r for r in receivers)
            if not is_target_sender and not is_target_receiver:
                continue
            verifier.register_communication_message(msg, user_id=target_norm)
            workload_tracker.register_role_activity(role="general", timestamp=msg.timestamp)
            sliding_window.add_message(msg)
            comm_graph.add_communication(msg)
            if not is_target_sender:
                continue
            context = sliding_window.export_context()
            context = comm_graph.update_context(context)
            stat_vector = statistical_extractor.compute_markers(msg, context=context)
            llm_vector = llm_extractor.compute_markers(msg, context=context)
            vector = MarkerVector(
                pronoun_marker=stat_vector.pronoun_marker,
                syntactic_marker=stat_vector.syntactic_marker,
                activity_marker=stat_vector.activity_marker,
                response_marker=stat_vector.response_marker,
                social_marker=stat_vector.social_marker,
                llm_alienation=llm_vector.llm_alienation,
                llm_concealment=llm_vector.llm_concealment,
                llm_rationalization=llm_vector.llm_rationalization,
                llm_access_focus=llm_vector.llm_access_focus,
                llm_conflict=llm_vector.llm_conflict,
                llm_uncertainty=llm_vector.llm_uncertainty
            )
            workload_factor = workload_tracker.get_role_workload_factor("general", msg.timestamp)
            z_scores, risk_score, is_alert = protected_profile.register_observation(
                markers=vector,
                timestamp=msg.timestamp,
                team_workload_factor=workload_factor
            )
            _, current_h_alert = protected_profile.compute_dynamic_thresholds(workload_factor)
            time_steps.append(step)
            step += 1
            risk_scores.append(risk_score)
            alert_thresholds.append(current_h_alert)
            v_dict = ContaminationResistantProfile.marker_vector_to_dict(vector)
            for m in ContaminationResistantProfile.TRACKED_MARKERS:
                marker_histories[m].append(v_dict[m])
                baseline_histories[m].append(protected_profile.trusted_means.get(m, 0.0))
            if is_alert and first_alert_time is None:
                first_alert_time = msg.timestamp
                print(f"[!] Risk Alert Triggered on CERT Event at {msg.timestamp}")
                sig = RiskSignal(employee_id=target_norm, timestamp=msg.timestamp, raw_markers=vector, normalized_deviations=z_scores)
                audit_res = verifier.verify_alert_signal(sig)
                print(f"    --> Stage 2 Technical Audit: {audit_res.explanation}")
                protected_profile.freeze_on_alert(msg.timestamp)
    lead_time = BenchmarkEvaluator.calculate_alarm_lead_time(first_alert_time, t_first_action)
    print("\n" + "=" * 80)
    print("CERT r4.2 EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Target Subject:               {target_norm}")
    print(f"Total Sent Messages Analyzed: {step}")
    print(f"First Detected Alert:         {first_alert_time}")
    print(f"Ground Truth Incident Start:  {t_drift_start}")
    print(f"Ground Truth Host Action:     {t_first_action}")
    print(f"Lead Time (T_lead):           {lead_time:.2f} days")
    print("=" * 80)
    ego_subgraph = comm_graph.build_ego_graph(target_norm, radius=2)
    BenchmarkVisualizer.plot_communication_graph(ego_subgraph, target_norm, "output/cert_ego_graph.png")
    BenchmarkVisualizer.plot_all_markers_grid(time_steps, marker_histories, baseline_histories, "output/cert_all_markers.png")
    BenchmarkVisualizer.plot_top_markers_dynamics(time_steps, marker_histories, baseline_histories, ["activity_marker", "social_marker", "pronoun_marker"], "output/cert_top_markers.png")
    print("[+] Visualizations saved to output/cert_ego_graph.png, cert_top_markers.png, and cert_all_markers.png")


def run_controlled_stress_benchmark(scenario_type: str = "ramp") -> None:
    """
    Run benchmark with realistic noise injection and stress control.
    """
    injector = DriftInjector(seed=42)
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    target_user = "employee_u17"
    if scenario_type == "ramp":
        scenario = injector.generate_ramp_drift_scenario(
            user_id=target_user,
            start_time=start_time,
            duration_days=60,
            l_ramp_days=30,
            delta_max=3.5
        )
    else:
        scenario = injector.generate_stress_control_scenario(
            user_id=target_user,
            start_time=start_time,
            duration_days=40
        )
    protected_profile = ContaminationResistantProfile(
        alpha_ema=0.05,
        quarantine_delay_days=14,
        base_h_quarantine=2.0,
        base_h_alert=2.8
    )
    verifier = TargetedRetrospectiveVerifier(retro_window_days=7, mismatch_threshold=0.5)
    verifier.register_user_mapping(user_id=target_user, email=f"{target_user}@company.com")
    workload_tracker = TeamWorkloadContextTracker()
    sliding_window = SlidingTimeWindow(employee_id=target_user, role="general")
    comm_graph = DynamicCommunicationGraph(decay_window_days=30)
    statistical_extractor = SpacyStatisticalExtractor()
    for telem_event in scenario.synthetic_telemetry:
        verifier.register_telemetry_event(telem_event)
    time_steps: List[int] = []
    risk_scores: List[float] = []
    alert_thresholds: List[float] = []
    alerts_count = 0
    print(f"[*] Running evaluation for scenario: '{scenario.scenario_type}' (Malicious={scenario.is_malicious})...")
    for idx, msg in enumerate(scenario.synthetic_messages):
        verifier.register_communication_message(msg, user_id=target_user)
        intensity = 2.5 if (scenario_type == "stress" and 15 <= idx < 30) else 1.0
        workload_tracker.register_role_activity(role="general", timestamp=msg.timestamp, intensity=intensity)
        sliding_window.add_message(msg)
        comm_graph.add_communication(msg)
        context = sliding_window.export_context()
        context = comm_graph.update_context(context)
        stat_vector = statistical_extractor.compute_markers(msg, context=context)
        noise = float(np.random.normal(0.0, 0.02))
        if scenario_type == "ramp" and scenario.t_drift_start and msg.timestamp >= scenario.t_drift_start:
            elapsed = (msg.timestamp - scenario.t_drift_start).total_seconds() / 86400.0
            drift_val = 3.5 * min(1.0, elapsed / 30.0)
            alienation = float(np.clip(drift_val * 0.28 + noise, 0.0, 1.0))
            concealment = float(np.clip(drift_val * 0.25 + noise, 0.0, 1.0))
            access_focus = float(np.clip(drift_val * 0.32 + noise, 0.0, 1.0))
        elif scenario_type == "stress" and 15 <= idx < 30:
            alienation = float(np.clip(0.45 + noise, 0.0, 1.0))
            concealment = float(np.clip(0.02 + noise, 0.0, 1.0))
            access_focus = float(np.clip(0.01 + noise, 0.0, 1.0))
        else:
            alienation = float(np.clip(0.02 + noise, 0.0, 1.0))
            concealment = float(np.clip(0.01 + noise, 0.0, 1.0))
            access_focus = float(np.clip(0.01 + noise, 0.0, 1.0))
        vector = MarkerVector(
            pronoun_marker=stat_vector.pronoun_marker,
            syntactic_marker=stat_vector.syntactic_marker,
            activity_marker=stat_vector.activity_marker,
            response_marker=stat_vector.response_marker,
            social_marker=stat_vector.social_marker,
            llm_alienation=alienation,
            llm_concealment=concealment,
            llm_access_focus=access_focus
        )
        workload_factor = workload_tracker.get_role_workload_factor("general", msg.timestamp)
        z_scores, risk_score, is_alert = protected_profile.register_observation(
            markers=vector,
            timestamp=msg.timestamp,
            team_workload_factor=workload_factor
        )
        _, current_h_alert = protected_profile.compute_dynamic_thresholds(workload_factor)
        time_steps.append(idx)
        risk_scores.append(risk_score)
        alert_thresholds.append(current_h_alert)
        if is_alert:
            alerts_count += 1
            top_drivers = [
                f"{marker}: Z={score:+.2f}"
                for marker, score in z_scores.items()
                if abs(score) >= 2.0
            ]
            drivers_str = ", ".join(top_drivers) if top_drivers else "Composite variance surge"
            print(f"    [!] Alert on day {idx}: Risk={risk_score:.2f} >= Threshold={current_h_alert:.2f} | Drivers: [{drivers_str}]")
    print("\n" + "=" * 60)
    print(f"SCENARIO VERIFICATION: {scenario.scenario_type}")
    print(f"Total Observations:        {len(time_steps)}")
    print(f"Generated Alerts:          {alerts_count}")
    if scenario_type == "stress":
        print(f"False Positives in Stress: {alerts_count} (Goal: 0 FP)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insider Risk Early Detection Pipeline")
    parser.add_argument("--dataset", action="store_true", help="Execute on real CERT r4.2 dataset files in archive/")
    parser.add_argument("--user", type=str, default="aam0658", help="Target CERT employee ID (e.g. aam0658, ajr0932)")
    parser.add_argument("--scenario", type=str, default="ramp", choices=["ramp", "stress"], help="Synthetic scenario to run if --dataset is not set")
    args = parser.parse_args()
    if args.dataset:
        run_cert_dataset_evaluation(archive_dir="archive", target_user=args.user, use_mock_llm=True)
    else:
        run_controlled_stress_benchmark(args.scenario)
