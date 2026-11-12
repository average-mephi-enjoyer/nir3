"""Measured output only: provenance, cluster bootstrap, plots and markdown."""
import hashlib
import importlib.metadata
import json
import platform
import csv
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd


def write_event_log(path, observations, traces):
    import gzip
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["method", "message_id", "timestamp", "label", "risk", "alert", "state", "sequential_score", "baseline", "variance", "available_features", "semantic_metadata"])
        for method, trace in traces.items():
            alerts = {id(o) for o in trace.alerts}
            for i, o in enumerate(observations):
                writer.writerow([method, o.observation_id, o.timestamp, o.label, trace.scores[i], id(o) in alerts,
                    trace.states[i], trace.sequential_scores[i], trace.baselines[i], trace.variances[i],
                    o.markers.available_names(), json.dumps(o.markers.semantic_metadata)])
    with gzip.open(path.with_name(path.stem + "_admissions.jsonl.gz"), "wt", encoding="utf-8") as destination:
        for method, trace in traces.items():
            decisions = trace.profile.quarantine.decisions + list(trace.profile.quarantine.buffer)
            for o in decisions or trace.admissions:
                destination.write(json.dumps({"method": method, "id": o.observation_id, "timestamp": str(o.timestamp),
                    "status": getattr(o, "status", "ADMITTED"), "risk": getattr(o, "risk_score", None),
                    "threshold": getattr(o, "threshold", None), "normalized_risk": getattr(o, "normalized_risk", None),
                    "included": getattr(o, "included_in_reference", True), "history": getattr(o, "history", [])}) + "\n")


def save_manifest(destination, config):
    root = Path(__file__).resolve().parents[1]
    sources = sorted(p for folder in ("benchmark", "profiling", "core", "extractors", "parsers", "verification") for p in (root / folder).glob("*.py"))
    sources += [root / "main.py", root / "run_stress_test.py"]
    manifest = {"config": config, "python": platform.python_version(), "platform": platform.platform(),
                "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()}
    Path(destination).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def aggregate_results(results, output, seed=20261112, repetitions=2000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(results)
    metrics = ["pr_auc", "contamination_rate", "retained_contamination_rate", "detected_scenario",
               "false_alarms_per_1000_days", "available_fraction", "oracle_mean_error", "oracle_variance_error",
               "detection_delay_days", "lead_time_days", "false_freezes", "blocked_normal_update_fraction"]
    rows = []
    rng = np.random.default_rng(seed)
    for keys, group in frame.groupby(["split", "experiment", "scenario", "method_name"], sort=True):
        # Same seed across parameter settings is one cluster, not independent observations.
        clusters = group.groupby("seed")[metrics].mean()
        for metric in metrics:
            values = clusters[metric].dropna().to_numpy()
            if len(values) >= 2:
                resampled = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
                lower, upper = np.quantile(resampled, [.025, .975])
            else:
                lower = upper = float("nan")
            rows.append(dict(zip(("split", "experiment", "scenario", "method_name"), keys),
                             metric=metric, mean=float(values.mean()) if len(values) else float("nan"),
                             ci_low=lower, ci_high=upper, independent_clusters=len(values),
                             reason="" if len(values) >= 2 else "insufficient defined independent clusters"))
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(output) / "aggregate.csv", index=False, na_rep="N/A")
    for experiment in sorted(frame.experiment.unique()):
        subset = summary[(summary.experiment == experiment) & (summary.split == "test")]
        lines = [f"# Experiment {experiment}: test, seed-cluster bootstrap 95% intervals", "",
                 "| Scenario | Method | Metric | Mean | 95% interval | Clusters |", "|---|---|---|---|---|---|"]
        for row in subset.itertuples():
            fmt = lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A"
            lines.append(f"| {row.scenario} | {row.method_name} | {row.metric} | {fmt(row.mean)} | [{fmt(row.ci_low)}, {fmt(row.ci_high)}] | {row.independent_clusters} |")
        (Path(output) / f"summary_{experiment}.md").write_text("\n".join(lines), encoding="utf-8")
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        for axis, metric in zip(axes, ("contamination_rate", "detected_scenario", "available_fraction")):
            data = subset[subset.metric == metric].pivot(index="scenario", columns="method_name", values="mean")
            if not data.empty:
                data.index = [str(v).replace("false_alert_then_attack", "false alarm then attack").replace("ema_proxy_drift", "EMA proxy drift").replace("_", " ") for v in data.index]
                data.plot.bar(ax=axis, legend=False)
                for position, (_, row) in enumerate(data.iterrows()):
                    if row.isna().all():
                        axis.text(position, .02, "N/A", transform=axis.get_xaxis_transform(), ha="center", fontsize=8)
            axis.set_title({"contamination_rate": "Anomalous windows ever admitted (%)",
                            "detected_scenario": "Detected attack scenarios (fraction)",
                            "available_fraction": "Monitoring availability (fraction)"}[metric])
            axis.tick_params(axis="x", rotation=60, labelsize=9)
            axis.set_xlabel("")
            axis.grid(axis="y", alpha=.2)
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .97), ncol=3, frameon=False)
        fig.suptitle(f"Experiment {experiment}: test means across configured settings", y=1.)
        fig.tight_layout(rect=(0, 0, 1, .84))
        fig.savefig(Path(output) / f"summary_{experiment}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    return summary
