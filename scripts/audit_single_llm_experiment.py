"""Independently check saved three-arm evidence and plot measured risk traces."""
import argparse
import ast
import csv
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.local_llm import file_sha256
from core.research_acceptance import llm_prerequisite
from parsers.cert_ground_truth import load_cert_ground_truth
from scripts.run_single_llm_experiment import digest, SEMANTIC, write_json


def read_events(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="output/aam0658_llm_comparison")
    args = parser.parse_args()
    output = Path(args.output)
    progress = json.loads((output / "progress.json").read_text())
    if progress["status"] != "complete":
        raise ValueError("Cannot audit an incomplete experiment")
    manifest = json.loads((output / "manifest.json").read_text())
    protocol = manifest["config"]
    assert protocol["case"] == "aam0658", "This report is specifically for aam0658"
    assert protocol["runner_sha256"] == file_sha256(ROOT / "scripts/run_single_llm_experiment.py")
    for name, expected in manifest["source_sha256"].items():
        assert file_sha256(ROOT / name) == expected, f"Source changed since run: {name}"
    for name, expected in protocol["input_files_sha256"].items():
        assert file_sha256(ROOT / name) == expected, f"Input changed since run: {name}"
    email_path = next(ROOT / name for name in protocol["input_files_sha256"] if name.endswith("email.csv"))
    archive = email_path.parent.parent
    truth = load_cert_ground_truth(archive / "answers/insiders.csv")[protocol["case"]]
    assert len(truth) == 1 and len(truth[0].email_ids) == 0
    assert str(truth[0].start) == protocol["incidents"][0]["start"]
    assert str(truth[0].end) == protocol["incidents"][0]["end"]
    supplementary = sorted((archive / "r4.2/LDAP").glob("*.csv")) + [archive / "r4.2/readme.txt"]
    supplementary += [archive / "answers" / f"r4.2-{i.scenario}" / i.details for i in truth]
    write_json(output / "data_audit.json", {
        "email_and_incident_index_hashes_match_pre_inference_manifest": True,
        "supplementary_hashes_captured_at_audit": {str(p.relative_to(ROOT)): file_sha256(p) for p in supplementary},
        "first_annotated_technical_action_type": truth[0].first_action_type,
        "positive_email_ids": len(truth[0].email_ids)})
    records = []
    for path in (output / "semantic_cache").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["identity"]["protocol"] != protocol["cache_fingerprint"]:
            continue
        assert path.stem == digest(record["identity"])
        assert record["vector_sha256"] == digest(record["vector"])
        metadata = record["vector"]["semantic_metadata"]
        assert metadata["mode"] == "local_llm"
        assert metadata["model_reported"] == protocol["generation"]["model"]
        assert metadata["deployment"] == protocol["semantic_runtime"]
        assert record["identity"]["message_id"] == metadata["message_id"]
        assert metadata["attempts"], "Missing real inference request metadata"
        if "failure" not in metadata:
            assert metadata["attempts"][-1]["validation"] == "passed"
        else:
            assert SEMANTIC <= set(record["vector"]["unavailable"])
        records.append(record)
    assert len(records) == protocol["counts"]["inference"]
    lookup = {r["identity"]["message_id"]: r for r in records}
    assert len(lookup) == len(records)
    with (output / "metrics.csv").open(encoding="utf-8") as stream:
        metrics = {row["arm"]: row for row in csv.DictReader(stream)}
    assert set(metrics) == {"llm_protected", "no_llm_protected", "ema"}
    events = {arm: read_events(output / f"{arm}_1_0_events.csv.gz") for arm in metrics}
    identities = [[(e["message_id"], e["timestamp"], e["label"]) for e in part] for part in events.values()]
    assert identities[0] == identities[1] == identities[2]
    assert len(identities[0]) == protocol["counts"]["test"]
    for event in events["llm_protected"]:
        assert json.loads(event["semantic_metadata"]) == lookup[event["message_id"]]["vector"]["semantic_metadata"]
    for arm in ("no_llm_protected", "ema"):
        for event in events[arm]:
            assert not SEMANTIC.intersection(ast.literal_eval(event["available_features"]))
    for arm, part in events.items():
        assert sum(e["alert"] == "True" for e in part) == int(metrics[arm]["alert_episodes"])
    numeric = lambda arm: [float(e["risk"]) / float(metrics[arm]["threshold"]) for e in events[arm]]
    protected_llm, protected_no = numeric("llm_protected"), numeric("no_llm_protected")
    differences = [abs(a-b) for a, b in zip(protected_llm, protected_no) if math.isfinite(a) and math.isfinite(b)]
    audit = {"status": "passed", "case": protocol["case"], "same_test_ids_dates_labels": True,
        "test_observations_per_arm": len(identities[0]), "test_labels": dict(Counter(e[2] for e in identities[0])),
        "cache_records_verified": len(records), "real_request_attempts": sum(len(r["vector"]["semantic_metadata"]["attempts"]) for r in records),
        "invalid_model_outputs": sum("failure" in r["vector"]["semantic_metadata"] for r in records),
        "failure_reasons": dict(Counter(r["vector"]["semantic_metadata"]["failure_detail"] for r in records if "failure" in r["vector"]["semantic_metadata"])),
        "semantic_available_messages": sum(bool(SEMANTIC - set(r["vector"]["unavailable"])) for r in records),
        "nonzero_semantic_messages": sum(any(r["vector"][m] > 0 for m in SEMANTIC - set(r["vector"]["unavailable"])) for r in records),
        "marker_message_counts": {m: sum(r["vector"][m] > 0 and m not in r["vector"]["unavailable"] for r in records) for m in sorted(SEMANTIC)},
        "protected_normalized_risk_max_absolute_difference": max(differences) if differences else None,
        "source_hashes_match_manifest": True,
        "test_feature_coverage": {arm: dict(Counter(name for event in part for name in ast.literal_eval(event["available_features"]))) for arm, part in events.items()},
        "limitations": "Consistency audit, not independent semantic expert validation; only one CERT case"}
    write_json(output / "audit.json", audit)
    probe = json.loads((ROOT / "output/local_llm_probe.json").read_text())
    acceptance = llm_prerequisite(manifest, probe)
    acceptance["scope"] = "single case aam0658 only; does not replace the historical all-case heuristic experiment"
    acceptance["observed_dataset_inference_records"] = len(records)
    write_json(output / "research_acceptance.json", acceptance)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, sharey=True)
    names = {"llm_protected": "Qwen + protected dynamic profile", "no_llm_protected": "Protected dynamic profile, no LLM", "ema": "Ordinary EMA, no LLM"}
    for ax, arm, color in zip(axes, ("llm_protected", "no_llm_protected", "ema"), ("#2266aa", "#228866", "#aa5522")):
        dates = [datetime.fromisoformat(e["timestamp"]) for e in events[arm]]
        scores = numeric(arm)
        ax.plot(dates, scores, color=color, linewidth=.8, label="Risk / calibrated threshold")
        ax.axhline(1, color="black", linestyle="--", linewidth=.8, label="Local threshold (not final sequential alarm)")
        incident = protocol["incidents"][0]
        ax.axvspan(datetime.fromisoformat(incident["start"]), datetime.fromisoformat(incident["end"]), color="red", alpha=.14, label="Annotated scenario interval")
        alarms = [(d, s) for d, s, e in zip(dates, scores, events[arm]) if e["alert"] == "True"]
        if alarms:
            ax.scatter(*zip(*alarms), marker="^", color="red", zorder=5, label="Sequential alarm")
        ax.set_title(names[arm], loc="left", fontsize=11)
        ax.set_ylabel("Normalized risk")
        ax.grid(alpha=.2)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("CERT aam0658: identical 506 test observations; measured traces", fontsize=13)
    fig.tight_layout()
    fig.savefig(output / "risk_comparison.png", dpi=160)
    plt.close(fig)
    note = ["", "## Проверка сохранённых результатов", "",
        f"Проверено {len(records)} сохранённых измерений Qwen и совпадение идентификаторов, дат и меток всех {len(identities[0])} тестовых наблюдений в трёх подходах. Результат проверки: audit.json.",
        f"Ненулевые доступные семантические признаки обнаружены в {audit['nonzero_semantic_messages']} из {len(records)} сообщений. Некорректных ответов после повторного запроса: {audit['invalid_model_outputs']}.",
        f"Максимальное абсолютное различие нормированного риска защищённых вариантов с LLM и без неё: {audit['protected_normalized_risk_max_absolute_difference']}.",
        "График показывает риск, делённый на калиброванный порог каждого подхода. Пересечение локального порога само по себе не означает финальную последовательную тревогу.",
        "", "![Измеренные траектории риска](risk_comparison.png)", "",
        "Важное ограничение источника: archive/r4.2/readme.txt определяет email.content как список ключевых слов через пробел, без различения темы и тела письма. Это не корпус естественных рабочих писем; психолингвистические выводы по нему ограничены."]
    report_path = output / "comparison.md"
    existing = report_path.read_text(encoding="utf-8").split("\n## Проверка сохранённых результатов")[0].rstrip()
    report_path.write_text(existing + "\n" + "\n".join(note) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
