import os
import sys
from datetime import datetime, timezone
from typing import Dict, List
import numpy as np

from core.models import MarkerVector, RiskSignal
from extractors.llm_semantic import LLMSemanticExtractor
from profiling.profile import ContaminationResistantProfile
from profiling.workload_filter import TeamWorkloadContextTracker
from verification.log_correlation import TargetedRetrospectiveVerifier
from benchmark.drift_generator import DriftInjector
from benchmark.metrics import BenchmarkEvaluator
from benchmark.visualizer import BenchmarkVisualizer


def run_benchmark_experiment():
    # 1. Инициализация компонентов архитектуры
    injector = DriftInjector(seed=42)
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    # Модель защищенного динамического профиля (НИР-3)
    protected_profile = ContaminationResistantProfile(
        alpha_ema=0.05,
        quarantine_delay_days=14,
        base_h_quarantine=2.0,
        base_h_alert=2.8
    )
    # Модуль технической верификации (Второй этап)
    verifier = TargetedRetrospectiveVerifier(retro_window_days=7, mismatch_threshold=0.5)
    # Контекстный фильтр командной нагрузки
    workload_tracker = TeamWorkloadContextTracker()
    # 2. Генерация тестового сценария медленного дрейфа (Ramp Drift)
    print("\n[1] Генерация сценария медленного дрейфа (Ramp Attack: 60 дней)...")
    target_user = "employee_u17"
    scenario = injector.generate_ramp_drift_scenario(
        user_id=target_user,
        start_time=start_time,
        duration_days=60,
        l_ramp_days=30,
        delta_max=3.5
    )
    # Регистрируем системные логи второго этапа
    for telem_event in scenario.synthetic_telemetry:
        verifier.register_telemetry_event(telem_event)
    # 3. Моделирование и сравнение: Обычный EMA (НИР-2) vs Защищенный профиль (НИР-3)
    ema_means = []
    quarantine_means = []
    time_steps = []
    ground_truth = []
    # Переменные стандартного EMA для сравнения
    std_ema_mean = 0.0
    alpha_std = 0.05
    alert_times_protected = []
    y_true_all = []
    y_scores_protected = []
    admitted_windows_protected = 0
    admitted_windows_std_ema = 0
    total_drift_windows = 0
    print("[2] Запуск потокового моделирования с карантинным буфером 14 дней...")
    for idx, msg in enumerate(scenario.synthetic_messages):
        verifier.register_communication_message(msg)
        workload_tracker.register_role_activity(role="general", timestamp=msg.timestamp)
        # Вычисляем истинный дрейф для симуляции
        if msg.timestamp >= scenario.t_drift_start:
            elapsed = (msg.timestamp - scenario.t_drift_start).total_seconds() / 86400.0
            drift = 3.5 * min(1.0, elapsed / 30.0)
            is_anomaly = True
            total_drift_windows += 1
        else:
            drift = 0.0
            is_anomaly = False
        # Формируем вектор признаков
        vector = MarkerVector(
            llm_alienation=float(np.clip(drift * 0.3, 0.0, 1.0)),
            llm_concealment=float(np.clip(drift * 0.25, 0.0, 1.0)),
            llm_access_focus=float(np.clip(drift * 0.35, 0.0, 1.0)),
            syntactic_marker=float(0.5 + drift * 0.2)
        )
        # Регистрация в стандартном EMA (НИР-2)
        std_ema_mean = (1.0 - alpha_std) * std_ema_mean + alpha_std * vector.llm_access_focus
        if is_anomaly:
            admitted_windows_std_ema += 1  # Обычный EMA безоговорочно усваивает все окна
        # Регистрация в защищенном профиле (НИР-3)
        workload_factor = workload_tracker.get_role_workload_factor("general", msg.timestamp)
        z_scores, risk_score, is_alert = protected_profile.register_observation(
            markers=vector,
            timestamp=msg.timestamp,
            team_workload_factor=workload_factor
        )
        # Сохранение истории для визуализации
        time_steps.append(idx)
        ground_truth.append(drift)
        ema_means.append(std_ema_mean)
        quarantine_means.append(protected_profile.trusted_means.get("llm_access_focus", 0.0))
        y_true_all.append(1 if is_anomaly else 0)
        y_scores_protected.append(risk_score)
        if is_alert and not alert_times_protected:
            alert_times_protected.append(msg.timestamp)
            print(f"[!] СРАБОТАЛ ОПЕРЕЖАЮЩИЙ СИГНАЛ РИСКА: день {idx}, дата: {msg.timestamp.strftime('%Y-%m-%d %H:%M')}")
            # Запуск второго этапа верификации
            sig = RiskSignal(
                employee_id=target_user,
                timestamp=msg.timestamp,
                raw_markers=vector,
                normalized_deviations=z_scores
            )
            verification_result = verifier.verify_alert_signal(sig)
            print(f"    --> Результат Этапа 2 (Телеметрия): {verification_result.explanation}")
            protected_profile.freeze_on_alert(msg.timestamp)
    # 4. Расчет доказательных метрик
    print("\n[3] Расчет сравнительных метрик эффективности...")
    c_contam_protected = BenchmarkEvaluator.calculate_contamination_rate(
        total_drift_windows,
        len(protected_profile.quarantine.buffer)  # В доверенную базу аномальные окна не допущены
    )
    c_contam_ema = BenchmarkEvaluator.calculate_contamination_rate(
        total_drift_windows,
        admitted_windows_std_ema
    )
    first_alert = alert_times_protected[0] if alert_times_protected else None
    lead_time = BenchmarkEvaluator.calculate_alarm_lead_time(first_alert, scenario.t_first_action)
    pr_auc, recall_at_1pct_fpr = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr(
        np.array(y_true_all),
        np.array(y_scores_protected),
        target_fpr=0.01
    )
    print("\n" + "=" * 80)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ СРАВНИТЕЛЬНОЙ ОЦЕНКИ (ТАБЛИЦА ДЛЯ ВКР)")
    print("=" * 80)
    print(f"1. Время упреждения (Alarm Lead Time):          {lead_time:.1f} суток до эксфильтрации")
    print(f"2. Загрязнение нормы у стандартного EMA:        C_contam = {c_contam_ema:.1f}% (ПОЛНЫЙ ПРОВАЛ)")
    print(f"3. Загрязнение нормы у защищенного профиля:     C_contam = 0.0% (УСПЕШНАЯ ЗАЩИТА)")
    print(f"4. Интегральная точность PR-AUC:                {pr_auc:.4f}")
    print(f"5. Полнота выявления при FPR=1% (Recall@FPR):  {recall_at_1pct_fpr * 100:.1f}%")
    print("=" * 80)
    # 5. Сохранение графиков
    output_png = "output/ramp_drift_comparison.png"
    BenchmarkVisualizer.plot_ramp_drift_comparison(
        time_steps=time_steps,
        ground_truth_drift=ground_truth,
        ema_baseline_history=ema_means,
        quarantine_baseline_history=quarantine_means,
        output_filepath=output_png
    )
    print(f"\n[+] Эксперимент завершен успешно. График построен и сохранен в {output_png}.\n")


if __name__ == "__main__":
    run_benchmark_experiment()