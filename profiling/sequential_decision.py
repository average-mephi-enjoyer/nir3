from collections import deque
import math
from typing import Deque, List, Optional, Tuple
import numpy as np


class ConformalMartingaleDetector:
    """
    Online anomaly detector based on Conformal Martingales.
    Guarantees non-asymptotic family-wise false positive rate control:
    P_H0(sup_{t >= 1} E_t >= 1/alpha) <= alpha.
    """

    def __init__(
        self,
        alpha_fpr: float = 0.01,
        betting_power: float = 0.5,
        calibration_window_size: int = 60
    ):
        self.alpha_fpr = alpha_fpr
        self.threshold = 1.0 / alpha_fpr
        self.betting_power = betting_power
        self.calibration_scores: Deque[float] = deque(maxlen=calibration_window_size)
        self.martingale_value: float = 1.0
        self.history_e_values: List[float] = []
        self.history_martingale: List[float] = []

    def calibrate(self, baseline_score: float) -> None:
        """
        Store clean, mature observation risk scores to maintain empirical null distribution.
        """
        self.calibration_scores.append(float(baseline_score))

    def update(self, test_score: float) -> Tuple[float, float, bool]:
        """
        Compute conformal p-value, e-value, and sequential martingale test statistic.
        Returns: (p_value, martingale_value, is_stopped).
        """
        if len(self.calibration_scores) < 10:
            p_val = 0.5
            e_val = 1.0
        else:
            calib_arr = np.array(self.calibration_scores)
            count_ge = np.sum(calib_arr >= test_score)
            p_val = float((count_ge + 1.0) / (len(calib_arr) + 1.0))
            clamped_p = max(p_val, 1e-4)
            e_val = float(self.betting_power * (clamped_p ** (self.betting_power - 1.0)))
        self.martingale_value *= e_val
        self.history_e_values.append(e_val)
        self.history_martingale.append(self.martingale_value)
        is_stopped = bool(self.martingale_value >= self.threshold)
        return p_val, self.martingale_value, is_stopped

    def reset_after_alert(self) -> None:
        """
        Reset martingale accumulator following an incident freeze and investigation.
        """
        self.martingale_value = 1.0


class KOutOfNFilter:
    """
    Empirical multi-window persistent anomaly accumulator (k-out-of-n rule).
    Suppresses transient stochastic noise and single-message impulses.
    """

    def __init__(self, k: int = 3, n: int = 5):
        self.k = k
        self.n = n
        self.window: Deque[bool] = deque(maxlen=n)

    def update(self, is_instant_alert: bool) -> bool:
        """
        Record instantaneous alert flag and evaluate whether >= k alerts occurred in last n events.
        """
        self.window.append(bool(is_instant_alert))
        if len(self.window) < self.n:
            return False
        return sum(self.window) >= self.k

    def reset(self) -> None:
        self.window.clear()


class SequentialDecisionManager:
    def __init__(
        self,
        alpha_fpr: float = 0.01,
        k: int = 3,
        n: int = 5,
        use_conformal: bool = True
    ):
        self.use_conformal = use_conformal
        self.martingale_detector = ConformalMartingaleDetector(alpha_fpr=alpha_fpr)
        self.k_of_n_filter = KOutOfNFilter(k=k, n=n)

    def process_observation(
        self,
        risk_score: float,
        is_instant_alert: bool,
        is_clean_mature: bool = False
    ) -> Tuple[bool, str]:
        if is_clean_mature:
            self.martingale_detector.calibrate(risk_score)
        if self.use_conformal:
            p_val, mart_val, is_stopped = self.martingale_detector.update(risk_score)
            if is_stopped:
                rationale = f"Conformal Martingale stopping boundary reached (E_t={mart_val:.1e} >= {self.martingale_detector.threshold:.0f}, p={p_val:.4f})"
                return True, rationale
            return False, f"Martingale monitoring (E_t={mart_val:.2f}, p={p_val:.3f})"
        else:
            is_persistent = self.k_of_n_filter.update(is_instant_alert)
            if is_persistent:
                rationale = f"Persistent k-of-n criterion met ({self.k_of_n_filter.k} alerts in last {self.k_of_n_filter.n} messages)"
                return True, rationale
            return False, "Transient noise suppressed by k-of-n buffer"

    def freeze(self) -> None:
        self.martingale_detector.reset_after_alert()
        self.k_of_n_filter.reset()
