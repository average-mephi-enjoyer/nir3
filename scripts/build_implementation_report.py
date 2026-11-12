import json
import subprocess
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
items = [
    (1, "pyproject.toml, uv.lock, .python-version, .github/workflows/tests.yml", "Чистое Python 3.13 окружение, unittest, Ruff, CI; зафиксированные зависимости", "Проверено локально; удалённый CI не запускался без push"),
    (2, "parsers/cert_ground_truth.py, benchmark/cert_pipeline.py", "Список сценариев на пользователя, качество/источник меток, неизвестные письма, явная неоцениваемость", "Fixture и локальный answer key; сценарно-зависимое переобучение не реализовано, выбран разрешённый явный отказ"),
    (3, "verification/log_correlation.py, parsers/cert_telemetry_parser.py", "Исторические запросы, сортировка равных timestamp, статусы источников", "Пустые/неупорядоченные потоки проверены; HTTP основной серии не загружен"),
    (4, "benchmark/baseline_comparison.py, benchmark/drift_generator.py, benchmark/cert_pipeline.py", "Оценка до обновления; prior без будущих меток; отдельный RNG; фиксированная калибровка", "Тесты добавления будущего, изменения меток и длины генерации"),
    (5, "core/models.py, extractors/*, profiling/sliding_window.py", "Маски доступности, календарная предыстория, точные ID и алиасы", "Проверены пропуски, риск по доступным признакам, смена адреса"),
    (6, "profiling/quarantine.py, profiling/profile.py", "ID, риск/порог, статусы/история, фактическое включение, запрет дубликатов", "Тесты горизонта, будущего и неизменности порога решения"),
    (7, "profiling/profile.py, profiling/sequential_decision.py, benchmark/baseline_comparison.py", "Согласованная заморозка и проверенное восстановление; измерение остановок", "Ложная тревога до атаки и последующее обнаружение проверены"),
    (8, "profiling/profile.py, benchmark/*", "Раздельные ограничения среднего/дисперсии до обновления; измерение блокировок", "Hand-calculation тесты; сравнение без карантина/якоря"),
    (9, "benchmark/cert_pipeline.py, profiling/profile.py", "Независимые priors по нескольким ролям; смена prior или индивидуальный режим", "Тесты исключения target, малых групп и подтверждения смены роли"),
    (10, "profiling/workload_filter.py, benchmark/drift_generator.py", "Общий состав коллег, независимая история, суточный лаг, защита concealment/access", "Негативные stress/burnout/role_change; A/B разделены"),
    (11, "profiling/sequential_decision.py, docs/experimental_methodology.md", "k-of-n отдельно от экспериментального e-process; фиксированная calibration", "Эмпирическая реализация проверена; условная валидность и множественный FPR НЕ доказаны"),
    (12, "core/schemas.py, extractors/llm_semantic.py", "Проверка evidence, дубликатов, JSON, offsets/ID/хеша и альтернатив", "Pydantic и mock-контракты проходят"),
    (13, "extractors/llm_semantic.py, scripts/probe_local_llm.py", "Три режима без скрытого fallback; provenance и причины отказов", "Проверяемый статус: output/local_llm_probe.json; отдельный эксперимент: docs/aam0658_llm_experiment.md"),
    (14, "extractors/llm_semantic.py, tests/test_semantics_verification.py", "Loopback-only, proxy/redirect off, недоверенный текст отдельно от system", "Частично: mock-инъекции не доказывают устойчивость реальной LLM; confidence некалиброван"),
    (15, "docs/semantic_annotation.md", "Авторские категории, рубрика, положительные/отрицательные/неоднозначные примеры", "Частично: нет независимых экспертов, agreement и сравнение LLM — N/A"),
    (16, "parsers/cert_telemetry_parser.py", "Тип file_activity с документированной семантикой removable-copy и происхождением", "Копирование не объявляется неправомерной эксфильтрацией"),
    (17, "verification/mismatch_detector.py", "NOT_COMPARABLE / POTENTIAL_MISMATCH / CONFIRMED_VIOLATION; evidence IDs", "Частично: нет независимой матрицы полномочий и валидации качества"),
    (18, "benchmark/baseline_comparison.py", "A: шесть вариантов с общими условиями; B: отдельная компонентная абляция", "Эквивалентность при отключённых различиях проверена; технический вариант честно исключён"),
    (19, "profiling/profile.py, benchmark/baseline_comparison.py", "Выборочные mean/variance, явные правила EMA/clipped/quarantine, независимая calibration", "Ручные математические примеры и тест эквивалентности"),
    (20, "benchmark/metrics.py, benchmark/baseline_comparison.py", "Ever/retained/отзывы/oracle, AP/ties, календарная экспозиция, эпизоды, заморозка", "Ручные примеры, неизвестные метки и N/A; AP условна по доступным scores"),
    (21, "benchmark/cert_pipeline.py, main.py", "Train/calibration/test; seeded role sampling; все сценарии и отчёт об ошибках", "Переносимость между непересекающимися пользовательскими популяциями не заявляется"),
    (22, "benchmark/drift_generator.py, run_stress_test.py", "30 seed-кластеров, скорости/амплитуды/длительности, EMA proxy, ложная тревога до атаки", "Полные измеренные tune/test траектории; интервалы исследовательские"),
    (23, "configs/experiments.json, run_stress_test.py", "Автоматические абляции и one-at-a-time sensitivity шести параметров", "Выбор только на tune; техническая абляция недоступна без независимой валидации"),
    (24, "benchmark/reporting.py, scripts/*, configs/*", "Автоматические CSV/gzip, provenance, bootstrap, таблицы/графики, CI", "Локальные проверки и обе серии; CI-конфигурация не выдаётся за выполненный удалённый workflow"),
    (25, "README.md, docs/experimental_methodology.md", "Формулы, разделение, определения метрик, ограничения гарантий и точный LLM-режим", "Документация согласована с итоговыми измерениями"),
    (26, "scripts/build_publication_results.py, docs/publication_results.md", "Формальная проблема, алгоритм, измеренные сравнения/абляции/CI и ухудшения", "Числа генерируются из CSV; универсальное превосходство не заявляется"),
]


def main():
    validation = json.loads((ROOT / "output/validation.json").read_text())
    status = pd.read_csv(ROOT / "output/final_cert/status.csv")
    stress = pd.read_csv(ROOT / "output/final_stress/metrics_all_all.csv")
    replay = json.loads((ROOT / "output/review_replay_validation.json").read_text())
    if replay["numeric_mismatches"]:
        raise RuntimeError("Review replay has unresolved measurement differences")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    changed = subprocess.check_output(["git", "diff", "--name-only"], cwd=ROOT, text=True).splitlines()
    added = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    lines = ["# Отчёт о доработке прототипа НИР3", "",
        "**Условие приёмки пользователя: без реальной локальной LLM эксперимент не принимается. Предыдущие эвристические результаты — только предварительный контроль.**", "",
        "Код и воспроизводимые эксперименты выполнены локально. Статус отдельного LLM-прогона: [aam0658](aam0658_llm_experiment.md). Полная научная приёмка остаётся ограничена отсутствием независимой экспертной разметки и валидации технического подтверждения; эти пункты не объявляются выполненными.", "",
        "## Таблица по пунктам ТЗ", "", "| № | Файлы | Изменение | Проверка / ограничение |", "|---|---|---|---|"]
    lines += [f"| {number} | {files} | {description} | {verification} |" for number, files, description, verification in items]
    lines += ["", "## Изменённые и новые файлы", ""]
    lines += [f"- `{file}`" for file in sorted(set(changed + added + ["docs/implementation_report.md"]))]
    lines += ["", "## Проверки", "",
        f"- Итоговый набор: {validation['tests_run']} тестов; failures={validation['failures']}, errors={validation['errors']}, skipped={validation['skipped']}. Лог: `output/tests.log`.",
        "- `uv sync --locked` проверен в отдельном чистом окружении Python 3.13.15 в output/clean_env. Повторный запуск тестов без CERT: output/clean_environment_tests.log.",
        "- `uv run python scripts/validate.py`: unittest, Ruff, статический разбор, `main.py --help`, отрицательный тест отсутствующего CERT. Отсутствующий CERT ожидаемо возвращает код 2.",
        f"- `uv run python run_stress_test.py --scenario all --split all`: {len(stress)} измеренных оценок методов; отдельные CSV tune sensitivity и финального test.",
        f"- `uv run python main.py --dataset --all-users`: статусы {status.status.value_counts().to_dict()}; подробности в output/final_cert/status.csv.",
        "- Исторический прогон одного сценария: `uv run python main.py --dataset --user aam0658 --semantic-mode heuristic --output output/single_cert`; результат в output/single_cert. Отдельный LLM-прогон: docs/aam0658_llm_experiment.md.",
        "- `uv run python scripts/probe_local_llm.py`: фактический статус и provenance находятся в output/local_llm_probe.json. Это проверка интеграции, не независимая экспертная оценка качества.",
        "- `uv run python scripts/build_publication_results.py`: таблицы статьи автоматически из завершённых CSV.",
        f"- После усиления проверок восстановления повторно рассчитано {replay['replayed_method_results']} результатов: числовых расхождений нет. Отчёт: output/review_replay_validation.json.",
        "- Итоговый парсер отдельно проверил 1 705 820 строк file/device/logon, некорректных обязательных полей — 0; output/telemetry_source_validation.json.",
        "- Git diff проверен на whitespace-ошибки; ветки/коммиты/публикация не выполнялись.",
        "", "## Артефакты экспериментов", "",
        "- `docs/publication_results.md` — сводные таблицы и ограничения.",
        "- `output/final_stress/aggregate.csv`, `summary_A.md`, `summary_B.md`, `summary_A.png`, `summary_B.png`.",
        "- `output/final_stress/metrics_all_all.csv`, `sensitivity_tune.csv`, `selected_parameters.json` и gzip-траектории/журналы решений.",
        "- `output/final_cert/metrics.csv`, `summary.csv`, `status.csv`, `cert_ground_truth.csv`, ролевые сведения и технические аудиты.",
        "- `manifest.json` в каталогах результатов фиксирует параметры, зависимости и хеши реализации.",
        "- `output/measured_sources_before_review_hardening.zip` сохраняет исходники основной серии до заключительного усиления защитных проверок; повторный расчёт затронутых сценариев подтверждает неизменность измерений. Графики затем перестроены для читаемости из тех же CSV.",
        "", "## Ограничения", "",
        "- Независимая экспертная agreement/confidence calibration и техническая абляция остаются непроверенными; приведена рубрика и явные причины N/A.",
        "- Истинный чистый профиль CERT неизвестен. HTTP не загружен; матрица полномочий не предоставлена. Технические совпадения не объявлены установленной эксфильтрацией.",
        "- Неоцениваемые по предыстории сценарии остаются в status.csv. Переносимость между непересекающимися пользовательскими группами не проверена.",
        "- 30 seeds дают исследовательские интервалы, но не точную оценку редких ошибок. One-at-a-time чувствительность не заменяет анализ взаимодействий.",
        "- Anytime/FPR-гарантии e-process и множественный контроль сотрудников не доказаны. AP после заморозки требует совместной интерпретации с доступностью и пропусками.",
        "", "## Git", "", f"Рабочая ветка: `{branch}`. Исходный HEAD: `{head}`.",
        "Новые ветки и коммиты не создавались, согласно прямому запрету пользователя. Pull request не создавался. Все изменения доступны в рабочей папке."]
    (ROOT / "docs/implementation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Generated docs/implementation_report.md")


if __name__ == "__main__":
    main()
