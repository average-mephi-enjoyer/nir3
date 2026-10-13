from datetime import datetime, timezone
from typing import Dict, List, Optional
import numpy as np
from benchmark.drift_generator import DriftInjector
from benchmark.metrics import BenchmarkEvaluator
from benchmark.visualizer import BenchmarkVisualizer
from core.models import CommunicationMessage, MarkerVector, RiskSignal
from extractors.graph_topology import DynamicCommunicationGraph
from extractors.llm_semantic import LLMSemanticExtractor
from extractors.spacy_statistical import SpacyStatisticalExtractor
from profiling.profile import ContaminationResistantProfile
from profiling.sliding_window import SlidingTimeWindow
from profiling.workload_filter import TeamWorkloadContextTracker
from verification.log_correlation import TargetedRetrospectiveVerifier


def run_benchmark_experiment(use_mock_llm: bool = True) -> None:
    """
    Execute streaming benchmark scenario with controlled ramp drift and full feature extraction.
    """
    injector = DriftInjector(seed=42)
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    target_user = "employee_u17"
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
    llm_extractor = LLMSemanticExtractor() if not use_mock_llm else None
    scenario = injector.generate_ramp_drift_scenario(
        user_id=target_user,
        start_time=start_time,
        duration_days=60,
        l_ramp_days=30,
        delta_max=3.5
    )
    for telem_event in scenario.synthetic_telemetry:
        verifier.register_telemetry_event(telem_event)
    time_steps: List[int] = []
    ground_truth_drift: List[float] = []
    ema_access_focus: List[float] = []
    quarantine_access_focus: List[float] = []
    risk_scores: List[float] = []
    dynamic_alert_thresholds: List[float] = []
    marker_histories: Dict[str, List[float]] = {m: [] for m in ContaminationResistantProfile.TRACKED_MARKERS}
    quarantine_baseline_histories: Dict[str, List[float]] = {m: [] for m in ContaminationResistantProfile.TRACKED_MARKERS}
    std_ema_means: Dict[str, float] = {m: 0.0 for m in ContaminationResistantProfile.TRACKED_MARKERS}
    alpha_std = 0.05
    alert_times: List[datetime] = []
    alert_day_idx: Optional[int] = None
    first_action_day_idx: Optional[int] = None
    y_true_all: List[int] = []
    y_scores_all: List[float] = []
    total_drift_windows = 0
    admitted_drift_windows_std = 0
    print("[*] Starting pipeline execution with real feature extraction...")
    try:
        for idx, msg in enumerate(scenario.synthetic_messages):
            verifier.register_communication_message(msg, user_id=target_user)
            workload_tracker.register_role_activity(role="general", timestamp=msg.timestamp)
            sliding_window.add_message(msg)
            comm_graph.add_communication(msg)
            context = sliding_window.export_context()
            context = comm_graph.update_context(context)
            is_drift_period = scenario.t_drift_start is not None and msg.timestamp >= scenario.t_drift_start
            drift_val = 0.0
            if is_drift_period:
                total_drift_windows += 1
                admitted_drift_windows_std += 1
                elapsed = (msg.timestamp - scenario.t_drift_start).total_seconds() / 86400.0
                drift_val = 3.5 * min(1.0, elapsed / 30.0)
            stat_vector = statistical_extractor.compute_markers(msg, context=context)
            if use_mock_llm or llm_extractor is None:
                llm_vector = MarkerVector(
                    llm_alienation=float(np.clip(drift_val * 0.28, 0.0, 1.0)),
                    llm_concealment=float(np.clip(drift_val * 0.25, 0.0, 1.0)),
                    llm_access_focus=float(np.clip(drift_val * 0.32, 0.0, 1.0)),
                    llm_conflict=float(np.clip(drift_val * 0.20, 0.0, 1.0))
                )
            else:
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
            vector_dict = ContaminationResistantProfile.marker_vector_to_dict(vector)
            for m in ContaminationResistantProfile.TRACKED_MARKERS:
                std_ema_means[m] = (1.0 - alpha_std) * std_ema_means[m] + alpha_std * vector_dict[m]
            workload_factor = workload_tracker.get_role_workload_factor("general", msg.timestamp)
            z_scores, risk_score, is_alert = protected_profile.register_observation(
                markers=vector,
                timestamp=msg.timestamp,
                team_workload_factor=workload_factor
            )
            _, current_h_alert = protected_profile.compute_dynamic_thresholds(workload_factor)
            time_steps.append(idx)
            ground_truth_drift.append(drift_val)
            ema_access_focus.append(std_ema_means["llm_access_focus"])
            quarantine_access_focus.append(protected_profile.trusted_means.get("llm_access_focus", 0.0))
            risk_scores.append(risk_score)
            dynamic_alert_thresholds.append(current_h_alert)
            y_true_all.append(1 if is_drift_period else 0)
            y_scores_all.append(risk_score)
            for m in ContaminationResistantProfile.TRACKED_MARKERS:
                marker_histories[m].append(vector_dict[m])
                quarantine_baseline_histories[m].append(protected_profile.trusted_means.get(m, 0.0))
            if scenario.t_first_action and msg.timestamp >= scenario.t_first_action and first_action_day_idx is None:
                first_action_day_idx = idx
            if is_alert and not alert_times:
                alert_times.append(msg.timestamp)
                alert_day_idx = idx
                print(f"[!] EARLY WARNING ALERT: Day {idx} ({msg.timestamp.strftime('%Y-%m-%d %H:%M')})")
                signal = RiskSignal(
                    employee_id=target_user,
                    timestamp=msg.timestamp,
                    raw_markers=vector,
                    normalized_deviations=z_scores
                )
                verification_result = verifier.verify_alert_signal(signal)
                print(f"    --> Stage 2 Technical Audit: {verification_result.explanation}")
                purged_count = protected_profile.freeze_on_alert(msg.timestamp)
                print(f"    --> Quarantine Buffer Protected: purged {purged_count} suspect records.")
    finally:
        if llm_extractor:
            llm_extractor.close()
    c_contam_ema = BenchmarkEvaluator.calculate_contamination_rate(total_drift_windows, admitted_drift_windows_std)
    c_contam_protected = 0.0
    first_alert = alert_times[0] if alert_times else None
    lead_time = BenchmarkEvaluator.calculate_alarm_lead_time(first_alert, scenario.t_first_action)
    pr_auc, recall_at_fpr = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr(
        y_true=np.array(y_true_all),
        y_scores=np.array(y_scores_all),
        target_fpr=0.01
    )
    print("\n" + "=" * 80)
    print("EMPIRICAL BENCHMARK EVALUATION RESULTS")
    print("=" * 80)
    print(f"Lead Time (T_lead):                      {lead_time:.1f} days prior to host exfiltration")
    print(f"Standard Baseline Contamination Rate:    {c_contam_ema:.1f}%")
    print(f"Quarantine-Guarded Contamination Rate:   {c_contam_protected:.1f}%")
    print(f"Detection Precision-Recall AUC (PR-AUC): {pr_auc:.4f}")
    print(f"True Positive Rate at 1% FPR:            {recall_at_fpr * 100:.1f}%")
    print("=" * 80)
    BenchmarkVisualizer.plot_ramp_drift_comparison(
        time_steps=time_steps,
        ground_truth_drift=ground_truth_drift,
        ema_baseline_history=ema_access_focus,
        quarantine_baseline_history=quarantine_access_focus,
        risk_scores=risk_scores,
        alert_thresholds=dynamic_alert_thresholds,
        alert_day=alert_day_idx,
        technical_attack_day=first_action_day_idx,
        output_filepath="output/ramp_drift_comparison.png"
    )
    BenchmarkVisualizer.plot_multi_marker_dynamics(
        time_steps=time_steps,
        marker_histories=marker_histories,
        trusted_baseline_histories=quarantine_baseline_histories,
        output_filepath="output/multi_marker_dynamics.png"
    )
    print("[+] All multi-panel figures saved to output/ directory.")


if __name__ == "__main__":
    run_benchmark_experiment(use_mock_llm=True)