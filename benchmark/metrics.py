import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EvaluationMetricsResult:
    """
    Complex metrics for comparative evaluation of methods.
    """
    model_name: str
    pr_auc: float
    recall_at_fixed_fpr: float
    fixed_fpr_level: float
    alarm_lead_time_days: float
    contamination_rate: float  # C_contam [0.0..100.0 %]
    false_alarms_per_1000_days: float


class BenchmarkEvaluator:
    """
    Calculation of evidence metrics for detection and robustness.
    """

    @staticmethod
    def calculate_contamination_rate(
        total_drift_windows: int,
        windows_admitted_to_baseline: int
    ) -> float:
        """
        C_contam = (N_admitted_during_drift / N_total_drift_windows) * 100%
        """
        if total_drift_windows <= 0:
            return 0.0
        return float((windows_admitted_to_baseline / total_drift_windows) * 100.0)

    @staticmethod
    def calculate_alarm_lead_time(
        first_alert_time: Optional[datetime],
        first_technical_action_time: Optional[datetime]
    ) -> float:
        """
        Lead Time = t_first_technical_action - t_first_alert (в сутках).
        Positive value indicates proactive detection.
        """
        if not first_alert_time or not first_technical_action_time:
            return 0.0
        delta_seconds = (first_technical_action_time - first_alert_time).total_seconds()
        return float(delta_seconds / 86400.0)

    @staticmethod
    def compute_pr_auc_and_recall_at_fpr(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        target_fpr: float = 0.01
    ) -> Tuple[float, float]:
        """
        Calculation of PR-AUC and Recall at fixed FPR (1 false alarm per 100 observations / 1000 days).
        """
        if len(y_true) == 0 or np.sum(y_true) == 0:
            return 0.0, 0.0
        desc_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_indices]
        y_scores_sorted = y_scores[desc_indices]
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        if n_neg == 0:
            return 1.0, 1.0
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        recalls = tps / n_pos
        precisions = tps / (tps + fps)
        fprs = fps / n_neg
        try:
            if hasattr(np, "trapezoid"):
                pr_auc = float(np.trapezoid(precisions, recalls))
            else:
                pr_auc = float(getattr(np, "trapz")(precisions, recalls))
        except Exception:
            pr_auc = float(np.sum(precisions * (y_true_sorted == 1)) / n_pos)
        if math.isnan(pr_auc) or pr_auc < 0:
            pr_auc = float(np.sum(precisions * (y_true_sorted == 1)) / n_pos)
        valid_fpr_indices = np.where(fprs <= target_fpr)[0]
        if len(valid_fpr_indices) > 0:
            recall_at_fpr = float(recalls[valid_fpr_indices[-1]])
        else:
            recall_at_fpr = 0.0
        return abs(pr_auc), recall_at_fpr