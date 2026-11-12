from datetime import timedelta
import math
import numpy as np
from core.models import MarkerVector
from profiling.quarantine import QuarantineBuffer


class ContaminationResistantProfile:
    TRACKED_MARKERS = MarkerVector.feature_names()
    STRESS_SENSITIVE_MARKERS = {"llm_alienation", "llm_conflict", "syntactic_marker", "response_marker"}

    def __init__(self, alpha_ema=0.05, min_variance=1e-4, clip_limit=10.0, quarantine_delay_days=14,
                 base_h_quarantine=2.0, base_h_alert=3.0, shrinkage_tau=20.0, role_priors=None,
                 anchor_limit=2.0, protect_anchor=True, freeze_enabled=True):
        if not 0 < alpha_ema <= 1 or min_variance <= 0 or shrinkage_tau <= 0 or anchor_limit <= 0:
            raise ValueError("Invalid profile parameters")
        self.alpha_ema = alpha_ema
        self.min_variance = min_variance
        self.clip_limit = clip_limit
        self.base_h_quarantine = base_h_quarantine
        self.base_h_alert = base_h_alert
        self.shrinkage_tau = shrinkage_tau
        self.role_priors = role_priors or {}
        self.anchor_limit = anchor_limit
        self.protect_anchor = protect_anchor
        self.freeze_enabled = freeze_enabled
        self.quarantine = QuarantineBuffer(quarantine_delay_days, base_h_quarantine)
        self.trusted_means = {m: float(self.role_priors.get(m, {}).get("mean", 0.0)) for m in self.TRACKED_MARKERS}
        self.trusted_variances = {m: max(min_variance, float(self.role_priors.get(m, {}).get("variance", 1.0))) for m in self.TRACKED_MARKERS}
        self.anchor_means = dict(self.trusted_means)
        self.anchor_variances = dict(self.trusted_variances)
        self.observations_count = 0
        self.state = "COLD"
        self.last_admitted = []
        self.admission_log = []
        self.update_history = []
        self.bootstrap = []
        self.last_timestamp = None
        self.last_local_risk = 0.0
        self.last_anchor_risk = 0.0

    @classmethod
    def marker_vector_to_dict(cls, markers):
        values = {m: float(getattr(markers, m)) for m in cls.TRACKED_MARKERS}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Markers must be finite")
        return values

    def fit(self, observations):
        matrix = np.asarray([list(self.marker_vector_to_dict(v).values()) for v in observations], dtype=float)
        if len(matrix) < 2:
            raise ValueError("At least two independent training observations are required")
        self.trusted_means = dict(zip(self.TRACKED_MARKERS, matrix.mean(axis=0).tolist()))
        self.trusted_variances = dict(zip(self.TRACKED_MARKERS, np.maximum(matrix.var(axis=0, ddof=1), self.min_variance).tolist()))
        self.anchor_means = dict(self.trusted_means)
        self.anchor_variances = dict(self.trusted_variances)
        self.observations_count = len(matrix)
        self.state = "ACTIVE"

    def get_mixed_parameters(self, marker_name):
        mean = self.trusted_means[marker_name]
        variance = self.trusted_variances[marker_name]
        prior = self.role_priors.get(marker_name)
        if prior:
            eta = self.observations_count / (self.observations_count + self.shrinkage_tau)
            variance = eta * variance + (1 - eta) * prior["variance"] + eta * (1 - eta) * (mean - prior["mean"]) ** 2
            mean = eta * mean + (1 - eta) * prior["mean"]
        return mean, math.sqrt(max(variance, self.min_variance))

    def compute_dynamic_thresholds(self, team_workload_factor):
        factor = float(np.clip(team_workload_factor, 0.0, 1.0))
        return self.base_h_quarantine * (1 + 0.6 * factor), self.base_h_alert * (1 + 0.6 * factor)

    def compute_deviations(self, markers, team_workload_factor=0.0):
        if self.state == "COLD" and not self.role_priors:
            raise ValueError("Cold start requires training observations or independent role priors")
        values = self.marker_vector_to_dict(markers)
        z_scores, anchor_scores = {}, []
        factor = float(np.clip(team_workload_factor, 0.0, 1.0))
        for marker, value in values.items():
            mean, std = self.get_mixed_parameters(marker)
            adjustment = 1 + 0.5 * factor if marker in self.STRESS_SENSITIVE_MARKERS else 1.0
            z_scores[marker] = float(np.clip((value - mean) / (std * adjustment), -self.clip_limit, self.clip_limit))
            anchor_std = math.sqrt(self.anchor_variances[marker])
            anchor_scores.append(float(np.clip((value - self.anchor_means[marker]) / (anchor_std * adjustment), -self.clip_limit, self.clip_limit)))
        self.last_local_risk = math.sqrt(sum(v * v for v in z_scores.values()) / len(z_scores))
        self.last_anchor_risk = math.sqrt(sum(v * v for v in anchor_scores) / len(anchor_scores))
        risk = max(self.last_local_risk, self.last_anchor_risk) if self.protect_anchor else self.last_local_risk
        return z_scores, risk, risk >= self.compute_dynamic_thresholds(factor)[1]

    def register_observation(self, markers, timestamp, team_workload_factor=0.0, defer_freeze=False, observation_id=""):
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Observations must be chronological")
        self.last_timestamp = timestamp
        self.last_admitted = []
        z_scores, risk, alert = self.compute_deviations(markers, team_workload_factor)
        if self.state == "FROZEN":
            return z_scores, risk, alert
        if alert and self.state == "ACTIVE" and self.freeze_enabled and not defer_freeze:
            self.freeze_on_alert(timestamp)
            return z_scores, risk, alert
        self.quarantine.h_quarantine = self.compute_dynamic_thresholds(team_workload_factor)[0]
        self.quarantine.enqueue(timestamp, markers, self.marker_vector_to_dict(markers), risk, observation_id)
        approved, _ = self.quarantine.extract_mature_clean_observations(timestamp)
        for observation in approved:
            if self.state == "COLD":
                self.bootstrap.append(observation.raw_markers)
                observation.reason = "cold_start_candidate"
                if len(self.bootstrap) >= 30:
                    self.fit(self.bootstrap)
                continue
            before = (dict(self.trusted_means), dict(self.trusted_variances), self.observations_count)
            if self._update_trusted_baseline(observation.marker_dict):
                self.update_history.append((timestamp, observation, before))
                self.last_admitted.append(observation)
                self.admission_log.append(observation)
            else:
                observation.reason = "cumulative_anchor_limit"
        return z_scores, risk, alert

    def _update_trusted_baseline(self, clean_marker_dict):
        if self.state == "FROZEN":
            return False
        means, variances = {}, {}
        for marker, value in clean_marker_dict.items():
            difference = value - self.trusted_means[marker]
            means[marker] = self.trusted_means[marker] + self.alpha_ema * difference
            variances[marker] = max(self.min_variance, (1 - self.alpha_ema) * (self.trusted_variances[marker] + self.alpha_ema * difference ** 2))
        if self.protect_anchor:
            for marker in self.TRACKED_MARKERS:
                displacement = abs(means[marker] - self.anchor_means[marker]) / math.sqrt(self.anchor_variances[marker])
                if displacement > self.anchor_limit or variances[marker] > self.anchor_variances[marker] * (1 + self.anchor_limit) ** 2:
                    return False
        self.trusted_means, self.trusted_variances = means, variances
        self.observations_count += 1
        return True

    def freeze_on_alert(self, alert_time):
        if self.state == "FROZEN":
            return 0
        self.state = "FROZEN"
        start = alert_time - self.quarantine.delay
        removed = self.quarantine.purge_suspicious_horizon(start)
        while self.update_history and self.update_history[-1][0] >= start:
            _, observation, before = self.update_history.pop()
            self.trusted_means, self.trusted_variances, self.observations_count = before
            observation.reason = "revoked_after_alert"
        self.last_admitted = []
        return removed

    def resume_after_review(self, clean_observations, approved=False):
        if not approved:
            raise ValueError("Explicit review approval is required")
        self.fit(clean_observations)
        self.quarantine = QuarantineBuffer(self.quarantine.delay.days, self.base_h_quarantine)
        self.update_history.clear()
        self.last_admitted = []
