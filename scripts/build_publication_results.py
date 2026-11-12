"""Build article tables strictly from measured CSV files, never hard-coded metrics."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.research_acceptance import llm_prerequisite
STRESS = ROOT / "output/final_stress"
CERT = ROOT / "output/final_cert"
METRICS = {"contamination_rate": "Загрязнение ever, %", "retained_contamination_rate": "Загрязнение retained, %",
           "detected_scenario": "Обнаруженные сценарии", "false_alarms_per_1000_days": "FA/1000 normal days",
           "available_fraction": "Доступность", "oracle_mean_error": "Oracle mean RMS",
           "pr_auc": "AP по доступным scores", "recall_at_1pct_fpr": "Recall@1%FPR по доступным scores",
           "lead_time_days": "Упреждение, суток", "detection_delay_days": "Задержка, суток"}


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return "N/A"
    if len(values) < 2:
        return f"{values.mean():.4f} (CI N/A)"
    samples = values[rng.integers(0, len(values), (2000, len(values)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return f"{values.mean():.4f} [{low:.4f}; {high:.4f}]"


def table(frame, cluster, rng):
    lines = ["| Метод | " + " | ".join(METRICS.values()) + " |", "|---|" + "---|" * len(METRICS)]
    for method, group in frame.groupby("method_name", sort=False):
        units = group.groupby(cluster)[list(METRICS)].mean()
        lines.append("| " + method + " | " + " | ".join(interval(units[m], rng) for m in METRICS) + " |")
    return "\n".join(lines)


def main():
    global CERT
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert-dir", default=str(CERT), help="Use output/cert_local_llm after a completed real-LLM run")
    args = parser.parse_args()
    CERT = Path(args.cert_dir).resolve()
    frame = pd.read_csv(STRESS / "metrics_all_all.csv")
    expected = json.loads((STRESS / "manifest.json").read_text())["config"]
    expected_rows = 2 * len(expected["scenarios"]) * expected["repetitions"] * (6 * len(expected["settings_A"]) + 9 * len(expected["settings_B"]))
    if len(frame) != expected_rows or not (STRESS / "aggregate.csv").exists():
        raise RuntimeError("Stress suite is incomplete; refusing to publish partial results")
    cert = pd.read_csv(CERT / "metrics.csv")
    status = pd.read_csv(CERT / "status.csv")
    if not (CERT / "limitations.json").exists():
        raise RuntimeError("CERT evaluation is incomplete")
    rng = np.random.default_rng(20261112)
    selected = json.loads((STRESS / "selected_parameters.json").read_text())
    probe = json.loads((ROOT / "output/local_llm_probe.json").read_text())
    acceptance = llm_prerequisite(json.loads((CERT / "manifest.json").read_text()), probe)
    lines = ["# Данные для статьи: автоматически сформированные результаты", "",
        "Источник чисел — локальные CSV измерений. Скрипт: `scripts/build_publication_results.py`. Интервалы — percentile bootstrap 95%; коррелированные сообщения не считаются независимыми.", "",
        "## Постановка проблемы", "",
        "Адаптивный эталон может включать аномальные наблюдения, постепенно уменьшая их отклонение от самого себя. Оцениваются доля фактических допусков, остаточное загрязнение после откатов и независимая чистая траектория, если она известна.", "",
        "## Проверяемый алгоритм", "",
        "Обновление откладывается на календарный горизонт; допуск требует, чтобы весь наблюдаемый горизонт оставался ниже нормированного порога. Затем до изменения эталона проверяются отдельные ограничения среднего и дисперсии относительно исторического якоря. Последовательная тревога замораживает профиль и детектор, отзывает недавние обновления. Восстановление требует внешнего решения, а не истинной метки атаки.", "",
        f"Измерено {len(frame)} сочетаний метода, сценария, параметров и seed, включая tune и test. Выбор на tune: `{selected['candidate']}`. Параметры итогового синтетического test: `{json.dumps(selected['parameters'], ensure_ascii=False)}`.", "",
        "Подбор оптимизировал Protected по предзаданной потере; это ограничивает выводы о сравнении методов с их индивидуально оптимальными настройками. CERT использует отдельный фиксированный протокол по умолчанию, без настройки на его инсайдерских сценариях.", ""]
    test = frame[frame.split == "test"]
    lines.insert(2, "**ПРЕДВАРИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ. Требование реальной LLM не выполнено; эксперимент не принят.**" if not acceptance["llm_prerequisite_met"] else
                 "**Необходимое условие реальной LLM выполнено; научная валидность требует отдельной проверки качества маркеров.**")
    (ROOT / "output/research_acceptance.json").write_text(json.dumps(acceptance, indent=2), encoding="utf-8")
    for experiment in ("A", "B"):
        lines += [f"## Синтетический test: эксперимент {experiment}", "",
                  "Внутри каждого seed усредняются настройки и сценарии, затем ресемплируются seed-кластеры. N/A исключены из среднего соответствующей метрики; агрегат по разнородным сценариям носит описательный характер.", "",
                  table(test[test.experiment == experiment], "seed", rng), "",
                  f"Подробные сценарии: `output/final_stress/summary_{experiment}.md`. График: `output/final_stress/summary_{experiment}.png`.", ""]
    lines += ["## CERT", "", f"Статусы обработки: {status.status.value_counts().to_dict()}. Эти отказы не удалены из отчёта.", "",
              "Интервалы ниже ресемплируют пользователей; несколько сценариев одного пользователя входят в один кластер. Неоцениваемые сценарии перечислены отдельно в status.csv и не входят в качество на оцениваемых сценариях. Интервалы описывают выбранную когорту, не реальную популяцию сотрудников.", ""]
    for experiment in ("A", "B"):
        lines += [f"### CERT {experiment}", "", table(cert[cert.experiment == experiment], "user", rng), ""]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        averages = cert[cert.experiment == experiment].groupby("method_name", sort=False)[["detected_scenario", "available_fraction"]].mean()
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for axis, column, title in zip(axes, ("detected_scenario", "available_fraction"),
                                       ("Detected annotated scenarios (%)", "Mean monitoring availability (%)")):
            bars = axis.bar(averages.index, averages[column] * 100, color=plt.cm.tab10(np.arange(len(averages))))
            axis.bar_label(bars, fmt="%.1f%%", padding=3)
            axis.set_title(title)
            axis.set_ylim(0, 110)
            axis.tick_params(axis="x", rotation=35, labelsize=9)
            for label in axis.get_xticklabels():
                label.set_horizontalalignment("right")
            axis.grid(axis="y", alpha=.2)
        fig.suptitle(f"CERT r4.2: experiment {experiment}; unevaluable users listed separately")
        fig.tight_layout()
        fig.savefig(CERT / f"summary_{experiment}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        lines += [f"График: `output/final_cert/summary_{experiment}.png`. Частоты FA в таблице — макросредние индивидуальных частот, не отношение суммарных тревог к суммарной экспозиции.", ""]
    lines += ["## Ограничения и случаи неэффективности", "",
        "AP и Recall по доступным scores условны: остановка мониторинга может улучшать видимую AP одновременно с пропуском последующей атаки. Нулевое загрязнение Static или рано замороженного профиля не является достаточным преимуществом. Сравнивать нужно также полноту сценариев, ложные остановки, длительность заморозки и доступность.", "",
        "Истинный чистый oracle CERT неизвестен; ошибки относительно него — N/A. Начальный якорь не выдаётся за истинную норму. CERT содержит ключевые слова, поэтому числовые и CERT результаты не доказывают качество анализа естественной речи.", "",
        f"Реальный local_llm: `{probe['status']}`. Независимой экспертной разметки, калибровки confidence и качества технического подтверждения нет. Семантика основной серии — heuristic. HTTP не загружен в основной технический аудит. Вариант абляции с техническим подтверждением не выдаётся за измеренный.", "",
        "Для e-process не доказано условное ожидание e-values и нет anytime/FPR-гарантии. Основные таблицы относятся к эмпирическому k-of-n. Межпользовательская переносимость, множественный контроль ошибок и устойчивость настоящей Qwen к инъекциям требуют отдельных экспериментов.", ""]
    index = ["scenario", "drift_days", "intensity", "seed"]
    qa = test[(test.experiment == "A") & (test.method_name == "EMA quarantine")].set_index(index).sort_index()
    anchor = test[(test.experiment == "A") & (test.method_name == "EMA quarantine + anchor")].set_index(index).sort_index()
    compared = ["contamination_rate", "detected_scenario", "baseline_shift"]
    if np.allclose(qa[compared], anchor[compared], equal_nan=True):
        lines += ["В изолированной серии A карантин и карантин+якорь дали одинаковые измеренные загрязнение, обнаружение и сдвиг эталона во всех условиях. Добавочный эффект якоря в этих настройках не установлен; наличие ограничения в коде не является экспериментальным доказательством его преимущества.", ""]
    pivot = test[test.experiment == "A"].groupby(["scenario", "method_name"])[["detected_scenario", "available_fraction"]].mean()
    lines += ["### Измеренные ухудшения Protected относительно EMA quarantine + anchor", ""]
    found = False
    for scenario in test.scenario.unique():
        p, a = pivot.loc[(scenario, "Protected")], pivot.loc[(scenario, "EMA quarantine + anchor")]
        if p.available_fraction < a.available_fraction or (np.isfinite(p.detected_scenario) and p.detected_scenario < a.detected_scenario):
            found = True
            fmt = lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A"
            lines.append(f"- {scenario}: доступность {p.available_fraction:.4f} против {a.available_fraction:.4f}; доля обнаружения {fmt(p.detected_scenario)} против {fmt(a.detected_scenario)}. Это описательная разница, не автоматически значимый эффект.")
    if not found:
        lines.append("В этих двух агрегатах ухудшений не наблюдалось; это не доказательство универсального превосходства.")
    replay_path = ROOT / "output/review_replay_validation.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text())
        lines += ["", "## Проверка происхождения результатов", "",
            "Исходники на момент измерений сохранены в `output/measured_sources_before_review_hardening.zip`; исходные хеши сохранены в manifest.json. Последующие проверки усилили транзакционность восстановления, учёт замены старого эталона и диагностику неполных журналов.",
            f"Повторно рассчитано на итоговой реализации {replay['replayed_method_results']} результатов сценариев с восстановлением; числовых расхождений: {len(replay['numeric_mismatches'])}. Журналы CERT отдельно проверены итоговым парсером: `output/telemetry_source_validation.json`. Числа исходных CSV вручную не менялись."]
    (ROOT / "docs/publication_results.md").write_text("\n".join(lines), encoding="utf-8")
    print("Generated docs/publication_results.md from completed measured series")


if __name__ == "__main__":
    main()
