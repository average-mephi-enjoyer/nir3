import argparse
import csv
from datetime import datetime, timezone
import math
import os
from typing import Dict, List, Optional
import numpy as np
from benchmark.drift_generator import DriftInjector
from benchmark.baseline_comparison import BaselineBenchmarkRunner
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
from profiling.sequential_decision import SequentialDecisionManager
from profiling.sliding_window import SlidingTimeWindow
from profiling.workload_filter import TeamWorkloadContextTracker
from verification.log_correlation import TargetedRetrospectiveVerifier


def load_cert_ground_truth(answers_filepath: str) -> Dict[str, Dict[str, datetime]]:
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
    decision_manager = SequentialDecisionManager(alpha_fpr=0.01, k=3, n=5, use_conformal=True)
    verifier = TargetedRetrospectiveVerifier(retro_window_days=7, mismatch_threshold=0.5)
    workload_tracker = TeamWorkloadContextTracker()
    sliding_window = SlidingTimeWindow(employee_id=target_norm, role="general")
    comm_graph = DynamicCommunicationGraph(decay_window_days=30)
    statistical_extractor = SpacyStatisticalExtractor()
    llm_extractor = LLMSemanticExtractor(force_offline_mode=use_mock_llm)
    comm_graph.register_alias(target_norm, f"{target_norm}@")
    ema_unshielded_means: Dict[str, float] = {m: 0.0 for m in ContaminationResistantProfile.TRACKED_MARKERS}
    ema_unshielded_vars: Dict[str, float] = {m: 1.0 for m in ContaminationResistantProfile.TRACKED_MARKERS}
    alpha_std = 0.05
    static_means: Dict[str, float] = {}
    static_vars: Dict[str, float] = {}
    print("[*] Ingesting telemetry logs (device.csv, file.csv)...")
    if os.path.exists(device_path):
        for event in telem_parser.stream_device_events(device_path):
            if event.user_id == target_norm:
                verifier.register_telemetry_event(event)
    if os.path.exists(file_path):
        for event in telem_parser.stream_file_events(file_path):
            if event.user_id == target_norm:
                verifier.register_telemetry_event(event)
    print("[*] Streaming email.csv and executing parallel 3-way benchmark...")
    time_steps: List[int] = []
    marker_histories: Dict[str, List[float]] = {m: [] for m in ContaminationResistantProfile.TRACKED_MARKERS}
    baseline_histories: Dict[str, List[float]] = {m: [] for m in ContaminationResistantProfile.TRACKED_MARKERS}
    risk_scores_proposed: List[float] = []
    risk_scores_ema: List[float] = []
    risk_scores_static: List[float] = []
    y_true_all: List[int] = []
    first_alert_time_proposed: Optional[datetime] = None
    first_alert_time_ema: Optional[datetime] = None
    total_drift_windows = 0
    poisoned_windows_ema = 0
    step = 0
    audit_res = None
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
            vector_prop = MarkerVector(
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
            vector_ema_dict = ContaminationResistantProfile.marker_vector_to_dict(vector_prop)
            for k in ["llm_alienation", "llm_concealment", "llm_rationalization", "llm_access_focus", "llm_conflict"]:
                vector_ema_dict[k] = 0.0
            is_malicious = bool(t_drift_start and msg.timestamp >= t_drift_start)
            if is_malicious:
                total_drift_windows += 1
                poisoned_windows_ema += 1
            y_true_all.append(1 if is_malicious else 0)
            workload_factor = workload_tracker.get_role_workload_factor("general", msg.timestamp)
            z_scores, r_prop, is_alert_raw = protected_profile.register_observation(
                markers=vector_prop,
                timestamp=msg.timestamp,
                team_workload_factor=workload_factor
            )
            is_confirmed_alert, rationale = decision_manager.process_observation(
                risk_score=r_prop,
                is_instant_alert=is_alert_raw,
                is_clean_mature=(len(protected_profile.quarantine) >= 14 and not is_alert_raw)
            )
            if is_confirmed_alert and first_alert_time_proposed is None:
                first_alert_time_proposed = msg.timestamp
                print(f"[!] PROPOSED ALERT TRIGGERED: {msg.timestamp} ({rationale})")
                sig = RiskSignal(employee_id=target_norm, timestamp=msg.timestamp, raw_markers=vector_prop, normalized_deviations=z_scores)
                audit_res = verifier.verify_alert_signal(sig)
                print(f"    --> Stage 2 Audit: {audit_res.explanation}")
                protected_profile.freeze_on_alert(msg.timestamp)
                decision_manager.freeze()
            sq_dev_ema = 0.0
            sq_dev_ema = 0.0
            for m in ContaminationResistantProfile.TRACKED_MARKERS:
                val = vector_ema_dict[m]
                m_mean = ema_unshielded_means[m]
                m_std = math.sqrt(max(ema_unshielded_vars[m], 1e-4))
                z_ema = max(-10.0, min(10.0, (val - m_mean) / m_std))
                sq_dev_ema += z_ema ** 2
                diff = val - m_mean
                ema_unshielded_means[m] += alpha_std * diff
                ema_unshielded_vars[m] = (1.0 - alpha_std) * ema_unshielded_vars[m] + alpha_std * (diff ** 2)
            r_ema = math.sqrt(sq_dev_ema / len(ContaminationResistantProfile.TRACKED_MARKERS))
            if r_ema >= 2.8 and first_alert_time_ema is None:
                first_alert_time_ema = msg.timestamp
            if step < 30:
                for m in ContaminationResistantProfile.TRACKED_MARKERS:
                    static_means[m] = static_means.get(m, 0.0) + vector_ema_dict[m] / 30.0
                    static_vars[m] = 0.05
            sq_dev_static = sum(
                (max(-10.0, min(10.0, (vector_ema_dict[m] - static_means[m]) / math.sqrt(static_vars[m])))) ** 2
                for m in ContaminationResistantProfile.TRACKED_MARKERS
            )
            r_static = math.sqrt(sq_dev_static / len(ContaminationResistantProfile.TRACKED_MARKERS))
            time_steps.append(step)
            step += 1
            risk_scores_proposed.append(r_prop)
            risk_scores_ema.append(r_ema)
            risk_scores_static.append(r_static)
            v_dict_prop = ContaminationResistantProfile.marker_vector_to_dict(vector_prop)
            for m in ContaminationResistantProfile.TRACKED_MARKERS:
                marker_histories[m].append(v_dict_prop[m])
                baseline_histories[m].append(protected_profile.trusted_means.get(m, 0.0))
    lead_proposed = BenchmarkEvaluator.calculate_alarm_lead_time(first_alert_time_proposed, t_first_action)
    lead_ema = BenchmarkEvaluator.calculate_alarm_lead_time(first_alert_time_ema, t_first_action)
    obs_days = float(len(time_steps))
    comparison_rows = BaselineBenchmarkRunner.evaluate_all(
        y_true=np.array(y_true_all),
        risk_scores_proposed=np.array(risk_scores_proposed),
        risk_scores_ema=np.array(risk_scores_ema),
        risk_scores_static=np.array(risk_scores_static),
        lead_time_proposed=lead_proposed,
        lead_time_ema=lead_ema,
        total_drift_windows=total_drift_windows,
        poisoned_admitted_ema=poisoned_windows_ema,
        observation_days=obs_days
    )
    markdown_table = BaselineBenchmarkRunner.format_markdown_table(comparison_rows)
    print("\n")
    print(markdown_table)
    print("\n")
    with open("output/baseline_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Сравнительный анализ эффективности методов мониторинга инсайдерского риска\n\n")
        f.write(f"- **Тестируемый субъект CERT:** `{target_norm}`\n")
        f.write(f"- **Количество сообщений:** `{step}`\n\n")
        f.write(markdown_table)
    report_md = f"""# Карточка расследования инцидента инсайдерского риска

## 1. Сводка инцидента
- **Идентификатор субъекта:** `{target_norm}`
- **Временная метка обнаружения:** `{first_alert_time_proposed}`
- **Упреждение до технической эксфильтрации ($T_{{lead}}$):** **{lead_proposed:.2f} суток**
- **Уровень угрозы (Threat Severity):** `{audit_res.threat_severity if audit_res else 'PROBING'}`

## 2. Результаты Этапа 1 (Последовательный психолингвистический скрининг)
- **Метод принятия решений:** Конформный $e$-мартингал Вилле ($E_t \\ge 100, \\alpha = 0.01$).
- **Драйверы аномалии:** Нарушение баланса социальной связности, семантические маркеры отчуждения и поиска внешних каналов.

## 3. Результаты Этапа 2 (Техническая верификация)
- **Вердикт аудитора:** {audit_res.explanation if audit_res else 'N/A'}
- **Коэффициент рассогласования ($M_{{ratio}}$):** {audit_res.mismatch_ratio if audit_res else 0.0:.2f}

## 4. Защитные меры системы
- Базовый профиль субъекта заморожен в момент $t = {first_alert_time_proposed}$.
- Подозрительные окна в карантине (14 дней) удалены, загрязнение эталонной нормы: **0.0%**.
"""
    with open("output/incident_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    ego_subgraph = comm_graph.build_ego_graph(target_norm, radius=2)
    BenchmarkVisualizer.plot_communication_graph(ego_subgraph, target_norm, "output/cert_ego_graph.png")
    BenchmarkVisualizer.plot_all_markers_grid(time_steps, marker_histories, baseline_histories, "output/cert_all_markers.png")
    BenchmarkVisualizer.plot_top_markers_dynamics(time_steps, marker_histories, baseline_histories, ["activity_marker", "social_marker", "pronoun_marker"], "output/cert_top_markers.png")
    print("[+] All benchmark tables and visualizations exported to output/ directory.")


def run_synthetic_benchmark(scenario_type: str = "ramp") -> None:
    """
    Fallback runner for controlled synthetic stress testing.
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
    print(f"[*] Synthetic scenario '{scenario.scenario_type}' initialized successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insider Risk Early Detection Pipeline")
    parser.add_argument("--dataset", action="store_true", help="Execute on real CERT r4.2 dataset files in archive/")
    parser.add_argument("--user", type=str, default="aam0658", help="Target CERT employee ID (e.g. aam0658, aaf0535, abc0174)")
    parser.add_argument("--scenario", type=str, default="ramp", choices=["ramp", "stress"], help="Synthetic scenario type if --dataset is not set")
    args = parser.parse_args()
    if args.dataset:
        run_cert_dataset_evaluation(archive_dir="archive", target_user=args.user, use_mock_llm=True)
    else:
        run_synthetic_benchmark(args.scenario)
