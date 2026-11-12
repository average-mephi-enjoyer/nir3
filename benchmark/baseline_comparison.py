from dataclasses import dataclass, field, replace
from datetime import timedelta
import math
import numpy as np
from benchmark.metrics import BenchmarkEvaluator
from profiling.profile import ContaminationResistantProfile
from profiling.sequential_decision import SequentialDecisionManager
from profiling.quarantine import QuarantineObservation


@dataclass(slots=True)
class FeatureObservation:
    timestamp: object
    markers: object
    label: int = -1
    workload: float = 0.0
    observation_id: str = ""
    role: str = "general"
    role_prior: dict | None = None
    role_confirmed: bool = False
    clean_markers: object = None  # Evaluation-only counterfactual, never used by detection.


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
    retained_contamination_rate: float = float("nan")
    revoked_updates: int = 0
    oracle_mean_error: float = float("nan")
    oracle_variance_error: float = float("nan")
    detected_scenario: float = float("nan")
    false_alert_episodes: int = 0
    alert_episodes: int = 0
    monitoring_days: float = 0.0
    normal_user_days: float = 0.0
    active_user_days: int = 0
    frozen_days: float = 0.0
    available_fraction: float = 1.0
    false_freezes: int = 0
    blocked_normal_update_fraction: float = float("nan")
    threshold: float = 0.0
    experiment: str = "A"
    na_reasons: str = ""


@dataclass
class MethodTrace:
    scores: list = field(default_factory=list)
    alerts: list = field(default_factory=list)
    baselines: list = field(default_factory=list)
    variances: list = field(default_factory=list)
    states: list = field(default_factory=list)
    admissions: list = field(default_factory=list)
    sequential_scores: list = field(default_factory=list)
    profile: object = None
    calibration_scores: list = field(default_factory=list)
    reviews: list = field(default_factory=list)


class BaselineBenchmarkRunner:
    METHODS = ("Static", "EMA", "EMA clipped", "EMA quarantine", "EMA quarantine + anchor", "Protected")
    ABLATIONS = ("Base", "Base + graph", "Base + graph + semantics", "Full no context",
                 "Full context", "Full role", "No quarantine", "No anchor", "No freeze")

    @staticmethod
    def evaluate_stream(training, calibration, observations, role_priors=None, scenario_start=None,
                        first_action=None, scenario_end=None, alpha=0.05, quarantine_days=14,
                        use_conformal=False, experiment="A", anchor_limit=2.0, anchor_variance_ratio=9.0,
                        shrinkage_tau=20.0, threshold_quantile=.99, calibration_window=1000,
                        methods=None, review_schedule=None, exposure_end=None, use_role=False,
                        use_context=False, mechanism_overrides=None, scenario_intervals=None):
        if len(training) < 2 or len(calibration) < 10 or not observations:
            raise ValueError("Need >=2 training, >=10 calibration and nonempty test observations")
        for stream in (training, calibration, observations):
            if any(a.timestamp > b.timestamp for a, b in zip(stream, stream[1:])):
                raise ValueError("All streams must be chronological (ties retain input order)")
        if training[-1].timestamp >= calibration[0].timestamp or calibration[-1].timestamp >= observations[0].timestamp:
            raise ValueError("Training, calibration and test must be disjoint in time")
        if experiment not in {"A", "B"} or not 0 < threshold_quantile <= 1 or calibration_window < 10:
            raise ValueError("Invalid experiment/calibration configuration")
        identifiers = [o.observation_id or f"test:{i}" for i, o in enumerate(observations)]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Test observation IDs must be unique")
        event_truth = dict(zip(identifiers, (o.label for o in observations)))
        labels = np.array([o.label for o in observations])
        end = exposure_end or observations[-1].timestamp + timedelta(days=1)
        if end <= observations[-1].timestamp:
            raise ValueError("Exposure end must follow last observation")
        monitoring_days = (end - observations[0].timestamp).total_seconds() / 86400
        intervals = scenario_intervals or ([(scenario_start, scenario_end)] if scenario_start is not None else [])
        normal_days = BenchmarkEvaluator.calendar_normal_exposure(observations[0].timestamp, end, intervals=intervals)
        traces, rows = {}, []
        for name in (methods or (BaselineBenchmarkRunner.METHODS if experiment == "A" else BaselineBenchmarkRunner.ABLATIONS)):
            if experiment == "A":
                if name not in BaselineBenchmarkRunner.METHODS:
                    raise ValueError("Unknown method")
                update = {"Static": "static", "EMA": "ema", "EMA clipped": "clipped"}.get(name, "quarantine")
                anchor, freeze = name in {"EMA quarantine + anchor", "Protected"}, name == "Protected"
                context, role, features = use_context, use_role, set(training[0].markers.feature_names())
            else:
                if name not in BaselineBenchmarkRunner.ABLATIONS:
                    raise ValueError("Technical ablation unavailable without independent validation")
                update = "ema" if name == "No quarantine" else "quarantine"
                anchor, freeze = name != "No anchor", name != "No freeze"
                context = name in {"Full context", "Full role", "No quarantine", "No anchor", "No freeze"}
                role = name in {"Full role", "No quarantine", "No anchor", "No freeze"}
                features = set(training[0].markers.feature_names())
                if name == "Base":
                    features -= {m for m in features if m.startswith("llm_")} | {"social_marker"}
                elif name == "Base + graph":
                    features -= {m for m in features if m.startswith("llm_")}
            overrides = (mechanism_overrides or {}).get(name, {})
            update, anchor, freeze = overrides.get("update", update), overrides.get("anchor", anchor), overrides.get("freeze", freeze)
            def masked(vector):
                return replace(vector, unavailable=set(vector.unavailable) | (set(vector.feature_names()) - features))
            profile = ContaminationResistantProfile(alpha_ema=alpha, quarantine_delay_days=quarantine_days,
                role_priors=role_priors if role else None, protect_anchor=anchor, freeze_enabled=freeze,
                anchor_limit=anchor_limit, anchor_variance_ratio=anchor_variance_ratio, shrinkage_tau=shrinkage_tau)
            profile.fit([masked(o.markers) for o in training])
            profile.role = training[0].role
            trace = MethodTrace(profile=profile)
            traces[name] = trace
            detector = SequentialDecisionManager(use_conformal=use_conformal)
            # Fixed independent calibration; no selection based on score value.
            calibration_scores = [profile.compute_deviations(masked(o.markers), o.workload if context else 0)[1]
                                  for o in calibration[-calibration_window:]]
            trace.calibration_scores = calibration_scores
            threshold = max(float(np.quantile(calibration_scores, threshold_quantile, method="higher")), 1e-6)
            profile.base_h_alert, profile.base_h_quarantine = threshold, threshold * 2 / 3
            for score in calibration_scores:
                detector.martingale_detector.calibrate(score)
            frozen_days, false_freezes = 0., 0
            review_events = sorted(review_schedule or [], key=lambda item: item["timestamp"])
            review_index = 0
            for i, observation in enumerate(observations):
                # External reviews fixed before the run, never inferred from labels.
                while review_index < len(review_events) and review_events[review_index]["timestamp"] <= observation.timestamp:
                    review = review_events[review_index]
                    if profile.state == "FROZEN":
                        detector.review_profile(profile, [masked(v) for v in review["training"]],
                            review["calibration_scores"], observation.timestamp, approved=review["approved"])
                        trace.reviews.append({"timestamp": str(observation.timestamp), "source": review["source"]})
                    review_index += 1
                if role and observation.role_confirmed and observation.role_prior and profile.role != observation.role:
                    profile.switch_role_prior(observation.role, observation.role_prior, observation.timestamp, confirmed=True)
                elif role and observation.role != profile.role:
                    profile.role, profile.role_priors = observation.role, {}
                    profile.role_history.append({"role": observation.role, "timestamp": str(observation.timestamp), "reason": "prior unavailable; individual-only"})
                vector = masked(observation.markers)
                workload = observation.workload if context else 0.
                if update == "quarantine":
                    _, score, raw_alert = profile.register_observation(vector, observation.timestamp, workload,
                        defer_freeze=True, observation_id=identifiers[i])
                    trace.admissions.extend(profile.last_admitted)
                else:
                    _, score, raw_alert = profile.compute_deviations(vector, workload)
                    if profile.state == "ACTIVE" and (update == "ema" or update == "clipped" and score < threshold * 2 / 3):
                        before = (dict(profile.trusted_means), dict(profile.trusted_variances), profile.observations_count, dict(profile.marker_counts))
                        if profile._update_trusted_baseline(profile.marker_vector_to_dict(vector)):
                            admission = QuarantineObservation(observation.timestamp, vector, profile.marker_vector_to_dict(vector),
                                risk_score=score, observation_id=identifiers[i], threshold=profile.base_h_quarantine,
                                normalized_risk=score / profile.base_h_quarantine)
                            admission.transition("ADMITTED", observation.timestamp, "direct_update", included=True)
                            profile.update_history.append((observation.timestamp, admission, before))
                            profile.admission_log.append(admission)
                            trace.admissions.append(admission)
                alarm, _ = detector.process_observation(score, raw_alert)
                if alarm:
                    trace.alerts.append(observation)
                    if freeze:
                        detector.freeze_profile(profile, observation.timestamp)
                        false_freezes += int(observation.label == 0)
                next_time = observations[i + 1].timestamp if i + 1 < len(observations) else end
                if profile.state == "FROZEN":
                    frozen_days += (next_time - observation.timestamp).total_seconds() / 86400
                trace.scores.append(score if not detector.frozen or alarm else float("nan"))
                trace.baselines.append(dict(profile.trusted_means))
                trace.variances.append(dict(profile.trusted_variances))
                trace.states.append(profile.state)
                trace.sequential_scores.append(detector.martingale_detector.martingale_value)
            poisoned, retained, revoked = 0, 0, 0
            for admission in trace.admissions:
                identifier = admission.observation_id
                if not identifier:
                    identifier = identifiers[next(i for i, o in enumerate(observations) if o is admission)]
                anomalous = event_truth[identifier] == 1
                was_revoked = getattr(admission, "status", "ADMITTED") == "REVOKED"
                poisoned += int(anomalous)
                retained += int(anomalous and not was_revoked)
                revoked += int(was_revoked)
            positives = int(np.sum(labels == 1))
            first_detection = next((o.timestamp for o in trace.alerts if scenario_start is not None and
                scenario_start <= o.timestamp and (scenario_end is None or o.timestamp <= scenario_end)), None)
            false_alerts = sum(o.label == 0 and not any(left <= o.timestamp and (right is None or o.timestamp <= right)
                                                      for left, right in intervals) for o in trace.alerts)
            pr_auc, recall = BenchmarkEvaluator.compute_pr_auc_and_recall_at_fpr(labels, trace.scores)
            shift = math.sqrt(np.mean([(profile.trusted_means[m] - profile.anchor_means[m]) ** 2 /
                                      profile.anchor_variances[m] for m in profile.trained_markers]))
            oracle_mean, oracle_var = BenchmarkEvaluator.oracle_errors(training, observations, profile, alpha)
            normal_decisions = [d for d in profile.quarantine.decisions if event_truth.get(d.observation_id) == 0]
            blocked = sum(d.reason == "cumulative_anchor_limit" for d in normal_decisions)
            reasons = []
            if not positives:
                reasons.append("AP/recall/contamination: no known positive windows")
            if not np.any(labels == 0):
                reasons.append("Recall@FPR: no known negative windows")
            if first_detection is None:
                reasons.append("delay/lead: no detected scenario")
            if first_action is None:
                reasons.append("lead: no annotated technical action")
            if not math.isfinite(oracle_mean):
                reasons.append("oracle: independent clean trajectory unavailable")
            if frozen_days:
                reasons.append("AP/recall conditional on available scores; see monitoring coverage")
            rows.append(BaselineComparisonRow(name, pr_auc, recall,
                BenchmarkEvaluator.calculate_contamination_rate(positives, poisoned),
                BenchmarkEvaluator.calculate_alarm_lead_time(first_detection, first_action),
                false_alerts * 1000 / normal_days if normal_days else float("nan"),
                (first_detection - scenario_start).total_seconds() / 86400 if first_detection else float("nan"),
                shift, poisoned, len(trace.admissions),
                BenchmarkEvaluator.calculate_contamination_rate(positives, retained), revoked, oracle_mean, oracle_var,
                float(first_detection is not None) if scenario_start is not None else float("nan"),
                false_alerts, len(trace.alerts), monitoring_days, normal_days,
                len({o.timestamp.date() for o in observations}), frozen_days, 1 - frozen_days / monitoring_days,
                false_freezes, blocked / len(normal_decisions) if normal_decisions else float("nan"), threshold,
                experiment, "; ".join(reasons)))
        return rows, traces

    @staticmethod
    def format_markdown_table(rows):
        fields = ("method_name", "pr_auc", "recall_at_1pct_fpr", "contamination_rate", "retained_contamination_rate",
                  "lead_time_days", "false_alarms_per_1000_days", "detection_delay_days", "available_fraction")
        lines = ["| Method | AP (available) | Recall@1%FPR | Ever contaminated % | Retained % | Lead days | FA/1000 normal days | Delay days | Availability |",
                 "|" + "|".join(["---"] * len(fields)) + "|"]
        for row in rows:
            values = [getattr(row, key) for key in fields]
            lines.append("| " + " | ".join("N/A" if isinstance(value, float) and not math.isfinite(value)
                else f"{value:.4f}" if isinstance(value, float) else str(value) for value in values) + " |")
        return "\n".join(lines) + "\n"
