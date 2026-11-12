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
                 anchor_limit=2.0, protect_anchor=True, freeze_enabled=True, anchor_variance_ratio=9.0):
        if not 0 < alpha_ema <= 1 or min_variance <= 0 or shrinkage_tau <= 0 or anchor_limit <= 0:
            raise ValueError("Invalid profile parameters")
        if not all(math.isfinite(x) for x in (alpha_ema, min_variance, clip_limit, base_h_quarantine,
                                              base_h_alert, shrinkage_tau, anchor_limit, anchor_variance_ratio)):
            raise ValueError("Profile parameters must be finite")
        self.alpha_ema = alpha_ema
        self.min_variance = min_variance
        self.clip_limit = clip_limit
        self.base_h_quarantine = base_h_quarantine
        self.base_h_alert = base_h_alert
        self.shrinkage_tau = shrinkage_tau
        self.role_priors = role_priors or {}
        self.anchor_limit = anchor_limit
        if anchor_variance_ratio < 1:
            raise ValueError("Anchor variance ratio must be >= 1")
        self.anchor_variance_ratio = anchor_variance_ratio
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
        self.trained_markers = set(self.TRACKED_MARKERS) if role_priors else set()
        self.marker_counts = {m: 0 for m in self.TRACKED_MARKERS}
        self.state_history = []
        self.blocked_updates = 0
        self.attempted_updates = 0
        self.role = None
        self.role_history = []

    @classmethod
    def marker_vector_to_dict(cls, markers):
        values = {m: float(getattr(markers, m)) for m in markers.available_names()}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Markers must be finite")
        return values

    def fit(self, observations):
        if len(observations) < 2:
            raise ValueError("At least two independent training observations are required")
        self.trained_markers = set()
        for marker in self.TRACKED_MARKERS:
            values = [self.marker_vector_to_dict(v)[marker] for v in observations if marker not in v.unavailable]
            self.marker_counts[marker] = len(values)
            if len(values) >= 2:
                self.trusted_means[marker] = float(np.mean(values))
                self.trusted_variances[marker] = max(float(np.var(values, ddof=1)), self.min_variance)
                self.trained_markers.add(marker)
            elif marker in self.role_priors:
                self.trained_markers.add(marker)
        if not self.trained_markers:
            raise ValueError("No features with sufficient training history")
        self.anchor_means = dict(self.trusted_means)
        self.anchor_variances = dict(self.trusted_variances)
        self.observations_count = len(observations)
        self.state = "ACTIVE"

    def get_mixed_parameters(self, marker_name):
        mean = self.trusted_means[marker_name]
        variance = self.trusted_variances[marker_name]
        prior = self.role_priors.get(marker_name)
        if prior:
            count = self.marker_counts[marker_name]
            eta = count / (count + self.shrinkage_tau)
            variance = eta * variance + (1 - eta) * prior["variance"] + eta * (1 - eta) * (mean - prior["mean"]) ** 2
            mean = eta * mean + (1 - eta) * prior["mean"]
        return mean, math.sqrt(max(variance, self.min_variance))

    def compute_dynamic_thresholds(self, team_workload_factor):
        factor = max(0., min(1., float(team_workload_factor)))
        return self.base_h_quarantine, self.base_h_alert

    def compute_deviations(self, markers, team_workload_factor=0.0):
        if self.state == "COLD" and not self.role_priors:
            raise ValueError("Cold start requires training observations or independent role priors")
        values = self.marker_vector_to_dict(markers)
        z_scores, anchor_scores = {}, []
        factor = max(0., min(1., float(team_workload_factor)))
        for marker, value in values.items():
            if marker not in self.trained_markers:
                continue
            mean, std = self.get_mixed_parameters(marker)
            adjustment = 1 + 0.5 * factor if marker in self.STRESS_SENSITIVE_MARKERS else 1.0
            z_scores[marker] = max(-self.clip_limit, min(self.clip_limit, (value - mean) / (std * adjustment)))
            anchor_std = math.sqrt(self.anchor_variances[marker])
            anchor_scores.append(max(-self.clip_limit, min(self.clip_limit, (value - self.anchor_means[marker]) / (anchor_std * adjustment))))
        if not z_scores:
            raise ValueError("No available trained features for risk calculation")
        self.last_local_risk = math.sqrt(sum(v * v for v in z_scores.values()) / len(z_scores))
        self.last_anchor_risk = math.sqrt(sum(v * v for v in anchor_scores) / len(anchor_scores))
        risk = self.last_local_risk
        return z_scores, risk, risk >= self.compute_dynamic_thresholds(factor)[1]

    def register_observation(self, markers, timestamp, team_workload_factor=0.0, defer_freeze=False, observation_id=""):
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Observations must be chronological")
        self.last_timestamp = timestamp
        self.last_admitted = []
        z_scores, risk, alert = self.compute_deviations(markers, team_workload_factor)
        if self.state in {"FROZEN", "ALERT_PENDING_REVIEW", "RESUMING"}:
            return z_scores, risk, False
        if alert and self.state == "ACTIVE" and self.freeze_enabled and not defer_freeze:
            self.freeze_on_alert(timestamp)
            return z_scores, risk, alert
        self.quarantine.h_quarantine = self.compute_dynamic_thresholds(team_workload_factor)[0]
        self.quarantine.enqueue(timestamp, markers, self.marker_vector_to_dict(markers), risk, observation_id)
        approved, _ = self.quarantine.extract_mature_clean_observations(timestamp)
        for observation in approved:
            if self.state == "COLD":
                observation.transition("REJECTED", timestamp, "independent_training_required")
                continue
            before = (dict(self.trusted_means), dict(self.trusted_variances), self.observations_count, dict(self.marker_counts))
            if self._update_trusted_baseline(observation.marker_dict):
                observation.transition("ADMITTED", timestamp, "included_in_reference", included=True)
                self.update_history.append((timestamp, observation, before))
                self.last_admitted.append(observation)
                self.admission_log.append(observation)
            else:
                observation.transition("REJECTED", timestamp, "cumulative_anchor_limit")
        return z_scores, risk, alert

    def _update_trusted_baseline(self, clean_marker_dict):
        if self.state in {"FROZEN", "ALERT_PENDING_REVIEW", "RESUMING"}:
            return False
        self.attempted_updates += 1
        means, variances = dict(self.trusted_means), dict(self.trusted_variances)
        for marker, value in clean_marker_dict.items():
            if marker not in self.trained_markers:
                continue
            difference = value - self.trusted_means[marker]
            means[marker] = self.trusted_means[marker] + self.alpha_ema * difference
            variances[marker] = max(self.min_variance, (1 - self.alpha_ema) * (self.trusted_variances[marker] + self.alpha_ema * difference ** 2))
        if self.protect_anchor:
            for marker in self.TRACKED_MARKERS:
                displacement = abs(means[marker] - self.anchor_means[marker]) / math.sqrt(self.anchor_variances[marker])
                if displacement > self.anchor_limit or variances[marker] > self.anchor_variances[marker] * self.anchor_variance_ratio:
                    self.blocked_updates += 1
                    return False
        self.trusted_means, self.trusted_variances = means, variances
        self.observations_count += 1
        for marker in clean_marker_dict.keys() & self.trained_markers:
            self.marker_counts[marker] += 1
        return True

    def freeze_on_alert(self, alert_time):
        if self.state == "FROZEN":
            return 0
        self._set_state("ALERT_PENDING_REVIEW", alert_time, "sequential_alert")
        self._set_state("FROZEN", alert_time, "updates_and_monitoring_suspended")
        start = alert_time - self.quarantine.delay
        removed = self.quarantine.purge_suspicious_horizon(start)
        while self.update_history and self.update_history[-1][0] >= start:
            _, observation, before = self.update_history.pop()
            self.trusted_means, self.trusted_variances, self.observations_count, self.marker_counts = before
            observation.transition("REVOKED", alert_time, "revoked_after_alert")
        self.last_admitted = []
        return removed

    def _set_state(self, state, timestamp, reason):
        self.state = state
        self.state_history.append({"state": state, "timestamp": str(timestamp), "reason": reason})

    def switch_role_prior(self, role, prior, timestamp, confirmed=False):
        if not confirmed or not role or not prior:
            raise ValueError("Confirmed role and independently trained prior required")
        for parameters in prior.values():
            if not math.isfinite(parameters["mean"]) or not math.isfinite(parameters["variance"]) or parameters["variance"] <= 0:
                raise ValueError("Invalid role prior")
        self.role, self.role_priors = role, {key: dict(value) for key, value in prior.items()}
        self.role_history.append({"role": role, "timestamp": str(timestamp)})

    def resume_after_review(self, clean_observations, approved=False, timestamp=None):
        if not approved:
            raise ValueError("Explicit review approval is required")
        if len(clean_observations) < 2:
            raise ValueError("Independent reviewed observations required")
        # Validate and fit off to the side: a failed review must leave the frozen profile intact.
        candidate = ContaminationResistantProfile(min_variance=self.min_variance, role_priors=self.role_priors,
                                                  shrinkage_tau=self.shrinkage_tau)
        candidate.fit(clean_observations)
        timestamp = timestamp or self.last_timestamp
        self._set_state("RESUMING", timestamp, "explicit_review_approved")
        for observation in self.admission_log:
            if observation.included_in_reference:
                observation.transition("REVOKED", timestamp, "replaced_by_reviewed_reference")
        self.trusted_means, self.trusted_variances = candidate.trusted_means, candidate.trusted_variances
        self.observations_count, self.marker_counts = candidate.observations_count, candidate.marker_counts
        self.trained_markers = candidate.trained_markers
        self.quarantine.purge_suspicious_horizon(timestamp - self.quarantine.delay if timestamp else self.last_timestamp)
        self.update_history.clear()
        self.last_admitted = []
        self._set_state("ACTIVE", timestamp, "reviewed_restart_preserving_anchor")
