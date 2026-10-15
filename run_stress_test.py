import os
from datetime import datetime, timezone
import matplotlib.pyplot as plt
from benchmark.drift_generator import DriftInjector
from core.models import MarkerVector
from profiling.profile import ContaminationResistantProfile
from extractors.spacy_statistical import SpacyStatisticalExtractor
from extractors.llm_semantic import LLMSemanticExtractor


def main():
    print("[*] Запуск стресс-тестирования устойчивости профиля к медленному дрейфу (Slow Ramp Drift)...")
    injector = DriftInjector(seed=42)
    start_time = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)
    target_user = "employee_u17"
    scenario = injector.generate_ramp_drift_scenario(
        user_id=target_user,
        start_time=start_time,
        duration_days=60,
        l_ramp_days=30,
        delta_max=3.5
    )
    print(f"[*] Сгенерирован сценарий: {scenario.scenario_type}, сообщений: {len(scenario.synthetic_messages)}")
    protected_profile = ContaminationResistantProfile(alpha_ema=0.05, quarantine_delay_days=14)
    stat_extractor = SpacyStatisticalExtractor()
    llm_extractor = LLMSemanticExtractor(force_offline_mode=True)
    alpha_ema = 0.05
    unshielded_ema_mean = 0.0
    days = []
    observed_values = []
    ema_baseline = []
    protected_baseline = []
    for msg in scenario.synthetic_messages:
        day = (msg.timestamp - start_time).total_seconds() / 86400.0 
        stat_v = stat_extractor.compute_markers(msg)
        llm_v = llm_extractor.compute_markers(msg)
        if scenario.t_drift_start and msg.timestamp >= scenario.t_drift_start:
            elapsed = (msg.timestamp - scenario.t_drift_start).total_seconds() / 86400.0
            drift_val = 3.5 * min(1.0, elapsed / 30.0)
            llm_v.llm_alienation = float(drift_val)
        else:
            drift_val = 0.0
        vec = MarkerVector(
            pronoun_marker=stat_v.pronoun_marker,
            syntactic_marker=stat_v.syntactic_marker,
            activity_marker=stat_v.activity_marker,
            response_marker=stat_v.response_marker,
            social_marker=stat_v.social_marker,
            llm_alienation=llm_v.llm_alienation,
            llm_concealment=llm_v.llm_concealment,
            llm_rationalization=llm_v.llm_rationalization,
            llm_access_focus=llm_v.llm_access_focus,
            llm_conflict=llm_v.llm_conflict,
            llm_uncertainty=llm_v.llm_uncertainty
        )
        val = vec.llm_alienation
        unshielded_ema_mean += alpha_ema * (val - unshielded_ema_mean)
        z_scores, r_prop, is_alert = protected_profile.register_observation(
            markers=vec,
            timestamp=msg.timestamp,
            team_workload_factor=0.0
        )
        if is_alert:
            protected_profile.freeze_on_alert(msg.timestamp)
        days.append(day)
        observed_values.append(val)
        ema_baseline.append(unshielded_ema_mean)
        protected_baseline.append(protected_profile.trusted_means.get("llm_alienation", 0.0))
    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "ramp_drift_comparison.png")
    plt.figure(figsize=(10, 5.2), dpi=300)
    plt.plot(days, observed_values, label="Наблюдаемое поведение инсайдера", color="#d62728", linestyle="--", alpha=0.75, linewidth=1.8)
    plt.plot(days, ema_baseline, label="Стандартный профиль EMA", color="#ff7f0e", linewidth=2.2)
    plt.plot(days, protected_baseline, label="Предложенный защищенный профиль", color="#2ca02c", linewidth=2.5)
    plt.axvline(x=10.0, color="#7f7f7f", linestyle=":", linewidth=1.5, label="Начало дрейфа")
    plt.axvline(x=45.0, color="#9467bd", linestyle="-.", linewidth=1.5, label="Техническая эксфильтрация")
    plt.title("Стресс-тестирование базовой линии на устойчивость к медленному дрейфу", fontsize=11)
    plt.xlabel("Время наблюдения, суток", fontsize=10)
    plt.ylabel("Значение маркера риска", fontsize=10)
    plt.legend(loc="upper left", framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Успешно сгенерирован график: '{out_path}'")


if __name__ == "__main__":
    main()
