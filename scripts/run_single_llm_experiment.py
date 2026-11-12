"""Paired single-CERT-case experiment; labels never enter semantic inference."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pydantic import ValidationError
from benchmark.baseline_comparison import BaselineBenchmarkRunner
from benchmark.cert_pipeline import extract_cohort, prepare_user
from benchmark.reporting import save_manifest, write_event_log
from core.local_llm import file_sha256
from core.models import MarkerVector
from extractors.llm_semantic import LLMSemanticExtractor
from parsers.cert_ground_truth import load_cert_ground_truth

SEMANTIC = {name for name in MarkerVector.feature_names() if name.startswith("llm_")}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def encode_vector(vector):
    data = asdict(vector)
    data["unavailable"] = sorted(vector.unavailable)
    return data


def decode_vector(data):
    return MarkerVector(**{**data, "unavailable": set(data["unavailable"])})


def merge_semantics(base, semantic):
    return replace(base, **{name: getattr(semantic, name) for name in SEMANTIC},
        llm_uncertainty=semantic.llm_uncertainty,
        unavailable=(base.unavailable - SEMANTIC) | (semantic.unavailable & SEMANTIC),
        semantic_metadata=semantic.semantic_metadata)


def cached_inference(extractor, message, cache, fingerprint):
    # Message identity is included: no reuse across different observations or prompts.
    identity = {"protocol": fingerprint, "message_id": message.message_id,
                "text_sha256": hashlib.sha256(message.text_content.encode()).hexdigest()}
    path = cache / (digest(identity) + ".json")
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"] != identity or record["vector_sha256"] != digest(record["vector"]):
            raise ValueError("Semantic cache integrity check failed")
        return decode_vector(record["vector"]), True
    try:
        vector = extractor.compute_markers(message)
    except RuntimeError as error:
        # Invalid model output is missing measurement, not a zero risk observation.
        # Transport, identity and runtime failures stop the entire experiment.
        cause = error.__cause__
        if not isinstance(cause, ValidationError) and not (
            isinstance(cause, ValueError) and str(cause).startswith("LLM output was truncated")
        ):
            raise
        vector = MarkerVector(unavailable=set(SEMANTIC), semantic_metadata={
            **extractor.last_call, "message_id": message.message_id,
            "measurement_status": "invalid_model_output; semantic features unavailable"})
    data = encode_vector(vector)
    write_json(path, {"identity": identity, "vector": data, "vector_sha256": digest(data)})
    return vector, False


def evaluate(arm, observations, peers, boundaries, incidents, output, user, methods):
    training, calibration, testing, _, _ = prepare_user(user, observations, peers, boundaries, incidents)
    rows = []
    for index, incident in enumerate(incidents):
        metrics, traces = BaselineBenchmarkRunner.evaluate_stream(
            training, calibration, testing, scenario_start=incident.start, first_action=incident.first_action,
            scenario_end=incident.end, exposure_end=boundaries[-1] + timedelta(seconds=1),
            scenario_intervals=[(i.start, i.end) for i in incidents], experiment="A", methods=methods,
            use_role=False, use_context=False)
        write_event_log(output / f"{arm}_{incident.scenario}_{index}_events.csv.gz", testing, traces)
        rows.extend({"arm": arm, "case": user, "scenario": f"{incident.scenario}_{index}", **asdict(row)} for row in metrics)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage(observations, boundaries):
    result = {}
    for name, left, right in (("train", None, boundaries[0]), ("calibration", boundaries[0], boundaries[1]),
                              ("gap", boundaries[1], boundaries[2]), ("test", boundaries[2], None)):
        part = [o for o in observations if (left is None or o.timestamp >= left) and (right is None or o.timestamp < right)]
        result[name] = {"messages": len(part),
            "semantic_available": sum(bool(SEMANTIC - o.markers.unavailable) for o in part),
            "invalid_model_output": sum("failure" in o.markers.semantic_metadata for o in part),
            "insufficient_context": sum(bool(o.markers.semantic_metadata.get("insufficient_context")) for o in part),
            "nonzero_semantic_messages": sum(any(getattr(o.markers, m) > 0 for m in SEMANTIC - o.markers.unavailable) for o in part)}
    return result


def report(output, rows, protocol, semantic_coverage, costs, historical):
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, float) and math.isfinite(value) else "N/A" if isinstance(value, float) else str(value)
    labels = {"llm_protected": "Qwen + защищённый профиль", "no_llm_protected": "Защищённый профиль без LLM", "ema": "Обычная EMA без LLM"}
    lines = [f"# Реальный эксперимент CERT: {protocol['case']}", "",
        "Три подхода, одинаковые исходные письма и хронологические границы. Выборка не сокращалась по меткам или риску.", "",
        "| Подход | Сценарий обнаружен | Ложные эпизоды | Ложные / 1000 дней | Задержка, дней | Доступность мониторинга | Допущено обновлений |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([labels[row["arm"]]] + [fmt(row[key]) for key in (
            "detected_scenario", "false_alert_episodes", "false_alarms_per_1000_days", "detection_delay_days", "available_fraction", "admitted_total")]) + " |")
    lines += ["", "## Протокол", "", f"Обучение до {protocol['train_end']}; калибровка до {protocol['calibration_end']}; тест с {protocol['test_start']}.",
        "EMA: alpha=0.05, без карантина, привязки к исходному профилю и заморозки. Защищённые варианты: карантин 14 дней, ограничение смещения и дисперсии относительно исходного профиля, заморозка/откат по тревоге.",
        "Порог каждого подхода — 99-й процентиль его независимой калибровки. Последовательное правило тревоги одинаково; conformal выключен.",
        "Сохранён протокол A прежнего прогона: роль и поправка на командную нагрузку выключены во всех трёх вариантах. Графовые и статистические признаки одинаковы. Другие сценарии через LLM не анализировались.",
        "Истинная разметка применяется только при оценке. Некорректный ответ LLM после одной попытки исправления помечается как недоступное измерение; эвристика и фиктивные нули его не заменяют.",
        "", "## Покрытие LLM", "", "```json", json.dumps(semantic_coverage, ensure_ascii=False, indent=2), "```", "",
        "## Измеренная стоимость inference", "", "```json", json.dumps(costs, ensure_ascii=False, indent=2), "```", "",
        "## Что было проведено ранее", "",
        "Предыдущие output/single_cert и output/final_cert использовали словарную эвристику (semantic_mode=heuristic), а не Qwen и не полностью отключённую семантику.",
        "```json", json.dumps(historical, ensure_ascii=False, indent=2), "```", "",
        "## Ограничения", "",
        "Для данного сценария в answer key нет размеченных положительных email-событий. Поэтому PR-AUC, Recall@FPR и загрязнение положительными письмами — N/A. Принадлежность письма временному интервалу атаки не заменяет его разметку.",
        "Обнаружение сценария означает тревогу внутри размеченного временного интервала; оно не доказывает семантическую точность или причинную связь. Один сценарий не позволяет оценить обобщаемость метода.",
        "Нулевая доля обнаружения не означает отсутствие реального запуска LLM. Источники метрик: metrics.csv и журналы каждого подхода; provenance и контрольные суммы: manifest.json; ответы: semantic_cache/."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="aam0658")
    parser.add_argument("--archive", default="archive")
    parser.add_argument("--output", default="output/aam0658_llm_comparison")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    messages = []
    print("Extracting common chronological features for one case; no peer LLM inference...", flush=True)
    features, peers, boundaries, config = extract_cohort(args.archive, [args.user], "disabled",
        include_references=False, message_sink=messages.append)
    observations = features[args.user]
    if len(observations) != len(messages) or any(o.observation_id != m.message_id for o, m in zip(observations, messages)):
        raise ValueError("Message/observation alignment failed")
    truth = load_cert_ground_truth(Path(args.archive) / "answers/insiders.csv")
    incidents = truth[args.user]
    training, calibration, testing, _, _ = prepare_user(args.user, observations, peers, boundaries, incidents)
    selected = [pair for pair in zip(observations, messages) if pair[0].timestamp < boundaries[1] or pair[0].timestamp >= boundaries[2]]
    # Gap messages contribute to rolling context, but are never scored or used for fitting.
    protocol = {**config, "case": args.user, "semantic_mode": "local_llm",
        "arms": ["llm_protected", "no_llm_protected", "ema"], "experiment": "A",
        "use_role": False, "use_context": False, "seed": 20261112,
        "alpha": .05, "quarantine_days": 14, "threshold_quantile": .99,
        "failure_policy": "invalid model output => semantic unavailable; infrastructure errors => stop",
        "counts": {"all": len(messages), "train": len(training), "calibration": len(calibration), "test": len(testing), "inference": len(selected)},
        "test_labels": dict(Counter(o.label for o in testing)),
        "input_messages_sha256": digest([(m.message_id, str(m.timestamp), m.text_content) for m in messages]),
        "incidents": [{"start": str(i.start), "end": str(i.end), "first_action": str(i.first_action), "email_ids": len(i.email_ids)} for i in incidents],
        "runner_sha256": file_sha256(Path(__file__)),
        "input_files_sha256": {str(path): file_sha256(path) for path in [Path(args.archive) / "r4.2/email.csv", Path(args.archive) / "answers/insiders.csv"]}}
    write_json(output / "plan.json", protocol)
    print(json.dumps(protocol["counts"]), flush=True)
    if args.prepare_only:
        return
    rows = evaluate("no_llm_protected", observations, peers, boundaries, incidents, output, args.user, ["Protected"])
    rows += evaluate("ema", observations, peers, boundaries, incidents, output, args.user, ["EMA"])
    write_csv(output / "non_llm_metrics.csv", rows)
    historical = []
    history_path = Path("output/single_cert/metrics.csv")
    if history_path.exists():
        with history_path.open(encoding="utf-8") as stream:
            historical = [row for row in csv.DictReader(stream) if row["user"] == args.user and row["experiment"] == "A" and row["method_name"] in {"Protected", "EMA"}]
    cache = output / "semantic_cache"
    cache.mkdir(exist_ok=True)
    with LLMSemanticExtractor(mode="local_llm") as extractor:
        protocol["semantic_runtime"] = extractor.verify_runtime()
        protocol["generation"] = {"system_prompt": extractor.SYSTEM_PROMPT, "repair_prompt": extractor.REPAIR_PROMPT,
            "seed": extractor.seed, "temperature": 0, "max_tokens": extractor.max_tokens,
            "model": extractor.model_name, "schema": extractor.schema_dict, "validation_retries": 1,
            "extractor_sha256": file_sha256(ROOT / "extractors/llm_semantic.py")}
        fingerprint = digest({"generation": protocol["generation"], "runtime": protocol["semantic_runtime"]})
        protocol["cache_fingerprint"] = fingerprint
        save_manifest(output / "manifest.json", protocol)
        observed = {}
        started = time.perf_counter()
        reused = invalid = 0
        progress = {"status": "running", "case": args.user, "total": len(selected)}
        for index, (observation, message) in enumerate(selected, 1):
            try:
                semantic, cached = cached_inference(extractor, message, cache, fingerprint)
            except Exception as error:
                write_json(output / "progress.json", {**progress, "status": "failed", "completed": index - 1,
                    "message_id": message.message_id, "error": str(error), "metadata": extractor.last_call})
                raise
            observed[observation.observation_id] = replace(observation, markers=merge_semantics(observation.markers, semantic))
            reused += int(cached)
            invalid += int("failure" in semantic.semantic_metadata)
            progress.update({"completed": index, "cache_hits": reused, "invalid_outputs": invalid,
                "elapsed_seconds": time.perf_counter() - started, "last_message_id": message.message_id})
            write_json(output / "progress.json", progress)
            if index == 1 or index % 10 == 0 or index == len(selected):
                print(f"Qwen {index}/{len(selected)}; cached={reused}; invalid={invalid}; elapsed={progress['elapsed_seconds']:.1f}s", flush=True)
    llm_observations = [observed.get(o.observation_id, o) for o in observations]
    rows = evaluate("llm_protected", llm_observations, peers, boundaries, incidents, output, args.user, ["Protected"]) + rows
    write_csv(output / "metrics.csv", rows)
    semantic_coverage = coverage(llm_observations, boundaries)
    calls = [o.markers.semantic_metadata for o in observed.values()]
    costs = {"messages": len(calls), "requests": sum(len(c.get("attempts", [])) for c in calls),
        "inference_seconds_sum": sum(c.get("latency_seconds", 0) for c in calls),
        "session_seconds": time.perf_counter() - started, "cache_hits": reused,
        "prompt_tokens": sum(a.get("usage", {}).get("prompt_tokens", 0) for c in calls for a in c.get("attempts", [])),
        "completion_tokens": sum(a.get("usage", {}).get("completion_tokens", 0) for c in calls for a in c.get("attempts", []))}
    write_json(output / "coverage.json", semantic_coverage)
    write_json(output / "costs.json", costs)
    report(output, rows, protocol, semantic_coverage, costs, historical)
    write_json(output / "progress.json", {**progress, "status": "complete"})
    print(f"Complete: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
