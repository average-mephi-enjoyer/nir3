import os
import matplotlib.pyplot as plt
from parsers.cert_telemetry_parser import TelemetryLogProcessor
from parsers.email_parser import EmailMessageParser
from profiling.profile import ContaminationResistantProfile
from profiling.sequential_decision import SequentialDecisionManager
from profiling.workload_filter import TeamWorkloadContextTracker
from profiling.sliding_window import SlidingTimeWindow
from extractors.spacy_statistical import SpacyStatisticalExtractor
from extractors.llm_semantic import LLMSemanticExtractor
from core.models import MarkerVector


def main():
    target_user = "csc0217"
    email_path = os.path.join("archive", "r4.2", "email.csv")
    if not os.path.exists(email_path):
        print(f"[-] Файл {email_path} не найден! Проверьте путь к архиву CERT.")
        return
    print(f"[*] Запуск извлечения траекторий мартингала и маркеров для {target_user}...")
    email_parser = EmailMessageParser()
    protected_profile = ContaminationResistantProfile(alpha_ema=0.05, quarantine_delay_days=14)
    decision_manager = SequentialDecisionManager(alpha_fpr=0.01, k=3, n=5, use_conformal=True)
    workload_tracker = TeamWorkloadContextTracker()
    sliding_window = SlidingTimeWindow(employee_id=target_user, role="admin")
    statistical_extractor = SpacyStatisticalExtractor()
    llm_extractor = LLMSemanticExtractor(force_offline_mode=True)
    steps = []
    dates = []
    martingale_vals = []
    alienation_vals = []
    conflict_vals = []
    access_vals = []
    step = 0
    alert_step = None
    for msg in email_parser.stream_cert_emails(email_path):
        sender_uid = (msg.sender_user_id or "").lower()
        sender_email = msg.sender_id.lower()
        receivers = [r.lower() for r in msg.receiver_ids]
        is_target_sender = (sender_uid == target_user or target_user in sender_email)
        is_target_receiver = any(target_user in r for r in receivers)
        if not is_target_sender and not is_target_receiver:
            continue
        workload_tracker.register_role_activity(role="admin", timestamp=msg.timestamp)
        sliding_window.add_message(msg)
        if not is_target_sender:
            continue
        context = sliding_window.export_context()
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
        workload_factor = workload_tracker.get_role_workload_factor("admin", msg.timestamp)
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
        current_martingale = decision_manager.martingale_detector.martingale_value
        steps.append(step)
        dates.append(msg.timestamp.strftime("%Y-%m-%d"))
        martingale_vals.append(current_martingale)
        alienation_vals.append(vector_prop.llm_alienation)
        conflict_vals.append(vector_prop.llm_conflict)
        access_vals.append(vector_prop.llm_access_focus)
        if is_confirmed_alert and alert_step is None:
            alert_step = step
            print(f"[!] Срабатывание границы Вилле на шаге {step}: {rationale}")
            protected_profile.freeze_on_alert(msg.timestamp)
            decision_manager.freeze()
        step += 1
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, dpi=300)
    ax1.plot(steps, martingale_vals, color="#1f77b4", linewidth=2.0, label=r"Конформный мартингал")
    ax1.axhline(y=100.0, color="crimson", linestyle="--", linewidth=1.8, label=r"Граница Вилле")
    if alert_step is not None:
        ax1.axvline(x=alert_step, color="darkred", linestyle=":", linewidth=1.5, label=f"Срабатывание тревоги")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"Значение мартингала", fontsize=10)
    ax1.set_title(f"Динамика конформного мартингала и маркеров риска субъекта CERT {target_user.upper()}", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.plot(steps, alienation_vals, color="#d62728", alpha=0.7, label="Отчуждение")
    ax2.plot(steps, conflict_vals, color="#ff7f0e", alpha=0.7, label="Конфликт")
    ax2.plot(steps, access_vals, color="#2ca02c", alpha=0.7, label="Фокус на доступе")
    if alert_step is not None:
        ax2.axvline(x=alert_step, color="darkred", linestyle=":", linewidth=1.5)
    ax2.set_xlabel("Номер исходящего сообщения", fontsize=10)
    ax2.set_ylabel("Интенсивность маркеров", fontsize=10)
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    out_img = os.path.join("output", "csc0217_martingale.png")
    plt.savefig(out_img)
    plt.close()
    print(f"[+] График успешно сохранен: {out_img}")


if __name__ == "__main__":
    main()
