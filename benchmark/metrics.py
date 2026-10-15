from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import numpy as np


@dataclass(slots=True)
class EvaluationMetricsResult:
    model_name: str
    pr_auc: float
    recall_at_fixed_fpr: float
    fixed_fpr_level: float
    alarm_lead_time_days: float
    contamination_rate: float
    false_alarms_per_1000_days: float


class BenchmarkEvaluator:
    @staticmethod
    def calculate_contamination_rate(
        total_drift_windows: int,
        admitted_poisoned_windows: int
    ) -> float:
        if total_drift_windows <= 0:
            return 0.0
        return float((admitted_poisoned_windows / total_drift_windows) * 100.0)

    @staticmethod
    def calculate_alarm_lead_time(
        first_alert_time: Optional[datetime],
        first_technical_action_time: Optional[datetime]
    ) -> float:
        if not first_alert_time or not first_technical_action_time:
            return 0.0
        delta_seconds = (first_technical_action_time - first_alert_time).total_seconds()
        return float(delta_seconds / 86400.0)

    @staticmethod
    def calculate_false_alarms_per_1000_days(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float,
        total_days: float
    ) -> float:
        if total_days <= 0:
            return 0.0
        predicted_positives = (y_scores >= threshold)
        false_alarms = np.sum((predicted_positives == 1) & (y_true == 0))
        return float((false_alarms / total_days) * 1000.0)

    @staticmethod
    def compute_pr_auc_and_recall_at_fpr(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        target_fpr: float = 0.01
    ) -> Tuple[float, float]:
        if len(y_true) == 0 or np.sum(y_true) == 0:
            return 0.0, 0.0
        desc_indices = np.argsort(y_scores)[::-1]
        y_true_sorted = y_true[desc_indices]
        n_pos = int(np.sum(y_true_sorted))
        n_neg = len(y_true_sorted) - n_pos
        if n_neg == 0:
            return 1.0, 1.0
        tps = np.cumsum(y_true_sorted)
        fps = np.cumsum(1 - y_true_sorted)
        recalls = tps / n_pos
        precisions = tps / (tps + fps)
        fprs = fps / n_neg
        recall_prev = np.insert(recalls[:-1], 0, 0.0)
        pr_auc = float(np.sum((recalls - recall_prev) * precisions))
        valid_fpr_indices = np.where(fprs <= target_fpr)[0]
        if len(valid_fpr_indices) > 0:
            recall_at_fpr = float(recalls[valid_fpr_indices[-1]])
        else:
            recall_at_fpr = 0.0
        return max(0.0, min(1.0, pr_auc)), max(0.0, min(1.0, recall_at_fpr))
