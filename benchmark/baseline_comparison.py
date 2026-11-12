from dataclasses import dataclass, field
from datetime import timedelta
import math
import numpy as np
from benchmark.metrics import BenchmarkEvaluator
from profiling.profile import ContaminationResistantProfile
from profiling.quarantine import QuarantineBuffer
from profiling.sequential_decision import SequentialDecisionManager


@dataclass(slots=True)
class FeatureObservation:
    timestamp: object
    markers: object
    label: int = -1
    workload: float = 0.0
    observation_id: str = ""


@dataclass(slots=True)
class BaselineComparisonRow:
    method_name: str
    pr_auc: float
    recall_at_1pct_fpr: float
    contamination_rate: float
    lead_time_days: float
    false_alarms_per_1000_days: float
    detection_delay_days: float = float("nan")
    baseline_shift: float = 0.0
    admitted_anomalies: int = 0
    admitted_total: int = 0


@dataclass
class MethodTrace:
    scores: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    admissions: list = field(default_factory=list)
    sequential_scores: list = field(default_factory=list)
    profile: object = None


class BaselineBenchmarkRunner:
    METHODS = ("Static", "EMA", "EMA clipped", "EMA quarantine", "Protected")

    @staticmethod
    def evaluate_stream(training, calibration, observations, role_priors=None, scenario_start=None,
                        first_action=None, scenario_end=None, alpha=0.05, quarantine_days=14,
                        use_conformal=False):
        if len(training) < 2 or len(calibration) < 10 or not observations:
            raise ValueError("Need >=2 training, >=10 calibration and nonempty test observations")
        if max(o.timestamp for o in training) >= min(o.timestamp for o in calibration):
            raise ValueError("Training and calibration must be disjoint in time")
        if max(o.timestamp for o in calibration) >= min(o.timestamp for o in observations):
            raise ValueError("Calibration and test must be disjoint in time")
        if any(a.timestamp > b.timestamp for a, b in zip(observations, observations[1:])):
            raise ValueError("Test observations must be chronological")
        traces, rows = {}, []
        vectors = [o.markers for o in training]
        labels = np.asarray([o.label for o in observations])
        truth = {id(o): o.label for o in observations}
        observation_ids = {id(o): o.observation_id or f"test:{index}" for index, o in enumerate(observations)}
        if len(set(observation_ids.values())) != len(observations):
            raise ValueError("Test observation IDs must be unique")
        event_truth = {observation_ids[id(o)]: o.label for o in observations}
        normal_dates = {o.timestamp.date() for o in observations if o.label == 0 and
                        (scenario_start is None or o.timestamp.date() < scenario_start.date() or
                         (scenario_end is not None and o.timestamp.date() > scenario_end.date()))}
        normal_days = len(normal_dates)
        for name in BaselineBenchmarkRunner.METHODS:
            protected = name == "Protected"
            profile = ContaminationResistantProfile(alpha_ema=alpha, quarantine_delay_days=quarantine_days,
                                                   role_priors=role_priors if protected else None,
                                                   protect_anchor=protected, freeze_enabled=protected)
            profile.fit(vectors)
            trace = MethodTrace(profile=profile)
            traces[name] = trace
            detector = SequentialDecisionManager(use_conformal=use_conformal)
            calibration_buffer = QuarantineBuffer(quarantine_days, profile.base_h_quarantine)
            mature = []
            for observation in calibration:
                score = profile.compute_deviations(observation.markers, observation.workload if protected else 0.0)[1]
                calibration_buffer.enqueue(observation.timestamp, observation.markers,
                                           profile.marker_vector_to_dict(observation.markers), score)
                admitted, _ = calibration_buffer.extract_mature_clean_observations(observation.timestamp)
                mature.extend(admitted)
            admitted, _ = calibration_buffer.extract_mature_clean_observations(observations[0].timestamp - timedelta(microseconds=1))
            mature.extend(admitted)
            if len(mature) < 10:
                raise ValueError("Calibration requires >=10 observations matured before test start")
            calibration_scores = [o.risk_score for o in mature]
            threshold = max(float(np.quantile(calibration_scores, 0.99, method="higher")), 1e-6)
            profile.base_h_alert = threshold
            profile.base_h_quarantine = threshold * 2 / 3
            for score in calibration_scores:
                detector.martingale_detector.calibrate(score)
            for observation in observations:
                workload = observation.workload if protected else 0.0
                if name in ("EMA quarantine", "Protected"):
                    _, score, raw_alert = profile.register_observation(observation.markers, observation.timestamp, workload,
                                                                      defer_freeze=True, observation_id=observation_ids[id(observation)])
                    trace.admissions.extend(profile.last_admitted)
                else:
                    _, score, _ = profile.compute_deviations(observation.markers)
                    raw_alert = score >= threshold
                    if name == "EMA" or (name == "EMA clipped" and score < threshold * 2 / 3):
                        profile._update_trusted_baseline(profile.marker_vector_to_dict(observation.markers))
                        trace.admissions.append(observation)
                alarm, _ = detector.process_observation(score, raw_alert)
                if alarm:
                    trace.alerts.append(observation)
                    if protected:
                        profile.freeze_on_alert(observation.timestamp)
                trace.scores.append(score)
                trace.baselines.append(dict(profile.trusted_means))
                trace.sequential_scores.append(detector.martingale_detector.martingale_value)
            admitted_labels = []
            for admission in trace.admissions:
                if hasattr(admission, "raw_markers"):
                    admitted_labels.append(event_truth[admission.observation_id])
                else:
                    admitted_labels.append(truth[id(admission)])
            poisoned = sum(label == 1 for label in admitted_labels)
            positives = int(np.sum(labels == 1))
            first_detection = next((o.timestamp for o in trace.alerts if scenario_start is not None and
                                    o.timestamp >= scenario_start and (scenario_end is None or o.timestamp <= scenario_end)), None)
            false_alerts = sum(o.label == 0 and o.timestamp.date() in normal_dates for o in trace.alerts)
            pr_auc, recall = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr(labels, trace.scores)
            shift = math.sqrt(sum((profile.trusted_means[m] - profile.anchor_means[m]) ** 2 /
                                 profile.anchor_variances[m] for m in profile.TRACKED_MARKERS) / len(profile.TRACKED_MARKERS))
            delay = (first_detection - scenario_start).total_seconds() / 86400 if first_detection and scenario_start else float("nan")
            rows.append(BaselineComparisonRow(name, pr_auc, recall,
                        BenchmarkEvaluator.calculate_contamination_rate(positives, poisoned),
                        BenchmarkEvaluator.calculate_alarm_lead_time(first_detection, first_action),
                        false_alerts * 1000 / normal_days if normal_days else float("nan"),
                        delay, shift, poisoned, len(trace.admissions)))
        return rows, traces

    @staticmethod
    def format_markdown_table(rows):
        fields = ("method_name", "pr_auc", "recall_at_1pct_fpr", "contamination_rate", "lead_time_days",
                  "false_alarms_per_1000_days", "detection_delay_days", "baseline_shift",
                  "admitted_anomalies", "admitted_total")
        lines = ["| Method | PR-AP | Recall@1%FPR | Contamination % | Lead days | FA/1000 normal user-days | Delay days | Anchor shift | Anomalous updates | All updates |",
                 "|" + "|".join(["---"] * len(fields)) + "|"]
        for row in rows:
            values = [getattr(row, field) for field in fields]
            lines.append("| " + " | ".join("N/A" if isinstance(value, float) and not math.isfinite(value)
                                          else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
        return "\n".join(lines) + "\n"
