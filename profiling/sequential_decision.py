from collections import deque
import math
import numpy as np


class ConformalMartingaleDetector:
    def __init__(self, alpha_fpr=0.01, betting_power=0.5, calibration_window_size=1000):
        if not 0 < alpha_fpr < 1 or not 0 < betting_power < 1:
            raise ValueError("Invalid sequential detector parameters")
        self.alpha_fpr = alpha_fpr
        self.threshold = 1.0 / alpha_fpr
        self.betting_power = betting_power
        self.calibration_scores = deque(maxlen=calibration_window_size)
        self.martingale_value = 1.0
        self.log_value = 0.0
        self.history_e_values = []
        self.history_martingale = []
        self.guarantee = "empirical_only"

    def calibrate(self, baseline_score):
        if not math.isfinite(baseline_score):
            raise ValueError("Calibration scores must be finite")
        self.calibration_scores.append(float(baseline_score))

    def update(self, test_score):
        if not math.isfinite(test_score):
            raise ValueError("Test scores must be finite")
        if len(self.calibration_scores) < 10:
            p_value, e_value = 1.0, 1.0
        else:
            scores = np.asarray(self.calibration_scores)
            p_value = float((1 + np.sum(scores >= test_score)) / (len(scores) + 1))
            e_value = self.betting_power * p_value ** (self.betting_power - 1)
        self.log_value += math.log(e_value)
        self.martingale_value = math.exp(max(-700.0, min(700.0, self.log_value)))
        self.history_e_values.append(e_value)
        self.history_martingale.append(self.martingale_value)
        return p_value, self.martingale_value, self.log_value >= math.log(self.threshold)

    def reset_after_alert(self):
        self.log_value = 0.0
        self.martingale_value = 1.0


class KOutOfNFilter:
    def __init__(self, k=3, n=5):
        if not 1 <= k <= n:
            raise ValueError("Expected 1 <= k <= n")
        self.k, self.n = k, n
        self.window = deque(maxlen=n)

    def update(self, is_instant_alert):
        self.window.append(bool(is_instant_alert))
        return len(self.window) == self.n and sum(self.window) >= self.k

    def reset(self):
        self.window.clear()


class SequentialDecisionManager:
    def __init__(self, alpha_fpr=0.01, k=3, n=5, use_conformal=True):
        self.use_conformal = use_conformal
        self.martingale_detector = ConformalMartingaleDetector(alpha_fpr=alpha_fpr)
        self.k_of_n_filter = KOutOfNFilter(k=k, n=n)
        self.frozen = False
        self.active_alert = False

    def process_observation(self, risk_score, is_instant_alert, is_clean_mature=False):
        if self.frozen:
            return False, "Frozen pending reviewed restart"
        if is_clean_mature:
            raise ValueError("Calibrate explicitly with previously admitted observations")
        if self.use_conformal:
            p_value, value, stopped = self.martingale_detector.update(risk_score)
            rationale = f"Empirical sequential score={value:.4g}, p={p_value:.4g}; no FPR guarantee"
        else:
            stopped = self.k_of_n_filter.update(is_instant_alert)
            rationale = "Empirical k-of-n detector"
        triggered = stopped and not self.active_alert
        self.active_alert = stopped
        return triggered, rationale

    def calibrate_admitted(self, observations):
        if not self.frozen:
            for observation in observations:
                self.martingale_detector.calibrate(observation.risk_score)

    def freeze(self):
        self.frozen = True

    def resume_after_review(self):
        self.frozen = False
        self.active_alert = False
        self.martingale_detector.reset_after_alert()
        self.martingale_detector.calibration_scores.clear()
        self.k_of_n_filter.reset()
