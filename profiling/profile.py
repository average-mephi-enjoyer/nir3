from datetime import datetime
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from core.models import MarkerVector
from profiling.quarantine import QuarantineBuffer


class ContaminationResistantProfile:
    TRACKED_MARKERS: List[str] = [
        "pronoun_marker",
        "syntactic_marker",
        "activity_marker",
        "response_marker",
        "social_marker",
        "llm_alienation",
        "llm_concealment",
        "llm_rationalization",
        "llm_access_focus",
        "llm_conflict"
    ]

    STRESS_SENSITIVE_MARKERS = {
        "llm_alienation",
        "llm_conflict",
        "syntactic_marker",
        "response_marker"
    }

    def __init__(
        self,
        alpha_ema: float = 0.05,
        min_variance: float = 1e-4,
        clip_limit: float = 10.0,
        quarantine_delay_days: int = 14,
        base_h_quarantine: float = 2.0,
        base_h_alert: float = 3.0,
        shrinkage_tau: float = 20.0,
        role_priors: Optional[Dict[str, Dict[str, float]]] = None
    ):
        self.alpha_ema = alpha_ema
        self.min_variance = min_variance
        self.clip_limit = clip_limit
        self.shrinkage_tau = shrinkage_tau
        self.base_h_quarantine = base_h_quarantine
        self.base_h_alert = base_h_alert
        self.role_priors = role_priors or {}
        self.trusted_means: Dict[str, float] = {}
        self.trusted_variances: Dict[str, float] = {}
        self.observations_count: int = 0
        self.quarantine = QuarantineBuffer(
            quarantine_delay_days=quarantine_delay_days,
            h_quarantine=base_h_quarantine
        )
        self._initialize_priors()

    def _initialize_priors(self) -> None:
        for marker in self.TRACKED_MARKERS:
            prior = self.role_priors.get(marker, {})
            self.trusted_means[marker] = float(prior.get("mean", 0.0))
            self.trusted_variances[marker] = float(prior.get("variance", 1.0))

    @classmethod
    def marker_vector_to_dict(cls, markers: MarkerVector) -> Dict[str, float]:
        raw_dict = {
            "pronoun_marker": markers.pronoun_marker,
            "syntactic_marker": markers.syntactic_marker,
            "activity_marker": markers.activity_marker,
            "response_marker": markers.response_marker,
            "social_marker": markers.social_marker,
            "llm_alienation": markers.llm_alienation,
            "llm_concealment": markers.llm_concealment,
            "llm_rationalization": markers.llm_rationalization,
            "llm_access_focus": markers.llm_access_focus,
            "llm_conflict": markers.llm_conflict
        }
        clean_dict: Dict[str, float] = {}
        for m in cls.TRACKED_MARKERS:
            val = raw_dict.get(m, 0.0)
            clean_dict[m] = 0.0 if (val is None or math.isnan(val)) else float(val)
        return clean_dict

    def get_mixed_parameters(self, marker_name: str) -> Tuple[float, float]:
        mu_ind = self.trusted_means.get(marker_name, 0.0)
        var_ind = max(self.trusted_variances.get(marker_name, self.min_variance), self.min_variance)
        role_prior = self.role_priors.get(marker_name, {})
        mu_role = float(role_prior.get("mean", mu_ind))
        var_role = float(role_prior.get("variance", var_ind))
        eta = self.observations_count / (self.observations_count + self.shrinkage_tau)
        mu_mixed = eta * mu_ind + (1.0 - eta) * mu_role
        var_mixed = (
            eta * var_ind
            + (1.0 - eta) * var_role
            + eta * (1.0 - eta) * ((mu_ind - mu_role) ** 2)
        )
        safe_std = math.sqrt(max(var_mixed, self.min_variance))
        return mu_mixed, safe_std

    def compute_dynamic_thresholds(self, team_workload_factor: float) -> Tuple[float, float]:
        clamped_factor = float(np.clip(team_workload_factor, 0.0, 1.0))
        threshold_multiplier = 1.0 + 0.6 * clamped_factor
        adapted_h_quarantine = self.base_h_quarantine * threshold_multiplier
        adapted_h_alert = self.base_h_alert * threshold_multiplier
        return adapted_h_quarantine, adapted_h_alert

    def compute_deviations(
        self,
        markers: MarkerVector,
        team_workload_factor: float = 0.0
    ) -> Tuple[Dict[str, float], float, bool]:
        mark_dict = self.marker_vector_to_dict(markers)
        z_scores: Dict[str, float] = {}
        total_sq_dev = 0.0
        _, dynamic_h_alert = self.compute_dynamic_thresholds(team_workload_factor)
        for m in self.TRACKED_MARKERS:
            val = mark_dict[m]
            mean_mix, std_mix = self.get_mixed_parameters(m)

            if m in self.STRESS_SENSITIVE_MARKERS and team_workload_factor > 0.0:
                std_mix *= (1.0 + 0.5 * team_workload_factor)
            raw_z = (val - mean_mix) / (std_mix + 1e-6)
            clipped_z = max(-self.clip_limit, min(self.clip_limit, raw_z))
            z_scores[m] = float(clipped_z)
            total_sq_dev += clipped_z ** 2
        risk_score = math.sqrt(total_sq_dev / len(self.TRACKED_MARKERS))
        is_alert = bool(risk_score >= dynamic_h_alert)
        return z_scores, risk_score, is_alert

    def register_observation(
        self,
        markers: MarkerVector,
        timestamp: datetime,
        team_workload_factor: float = 0.0
    ) -> Tuple[Dict[str, float], float, bool]:
        z_scores, risk_score, is_alert = self.compute_deviations(markers, team_workload_factor)
        dynamic_h_quarantine, _ = self.compute_dynamic_thresholds(team_workload_factor)
        self.quarantine.h_quarantine = dynamic_h_quarantine
        mark_dict = self.marker_vector_to_dict(markers)
        self.quarantine.enqueue(
            timestamp=timestamp,
            markers=markers,
            marker_dict=mark_dict,
            risk_score=risk_score
        )
        approved_obs, _ = self.quarantine.extract_mature_clean_observations(current_time=timestamp)
        for obs in approved_obs:
            self._update_trusted_baseline(obs.marker_dict)
        return z_scores, risk_score, is_alert

    def _update_trusted_baseline(self, clean_marker_dict: Dict[str, float]) -> None:
        if self.observations_count == 0 and not self.role_priors:
            for m in self.TRACKED_MARKERS:
                self.trusted_means[m] = clean_marker_dict[m]
                self.trusted_variances[m] = self.min_variance
            self.observations_count = 1
            return
        for m in self.TRACKED_MARKERS:
            val = clean_marker_dict[m]
            prev_mean = self.trusted_means[m]
            prev_var = self.trusted_variances[m]
            diff = val - prev_mean
            new_mean = prev_mean + self.alpha_ema * diff
            new_var = (1.0 - self.alpha_ema) * prev_var + self.alpha_ema * (diff ** 2)
            self.trusted_means[m] = float(new_mean)
            self.trusted_variances[m] = max(float(new_var), self.min_variance)
        self.observations_count += 1

    def freeze_on_alert(self, alert_time: datetime) -> int:
        return self.quarantine.purge_suspicious_horizon(start_time=alert_time)
