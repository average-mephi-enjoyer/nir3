from dataclasses import dataclass
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
    def calculate_contamination_rate(total_drift_windows, admitted_poisoned_windows):
        if total_drift_windows <= 0:
            return float("nan")
        return 100.0 * admitted_poisoned_windows / total_drift_windows

    @staticmethod
    def calculate_alarm_lead_time(first_alert_time, first_technical_action_time):
        if first_alert_time is None or first_technical_action_time is None:
            return float("nan")
        return (first_technical_action_time - first_alert_time).total_seconds() / 86400

    @staticmethod
    def calculate_false_alarms_per_1000_days(y_true, y_scores, threshold, total_days):
        if total_days <= 0:
            return float("nan")
        predicted = np.asarray(y_scores) >= threshold
        onsets = predicted & ~np.r_[False, predicted[:-1]]
        return float(np.sum(onsets & (np.asarray(y_true) == 0)) * 1000 / total_days)

    @staticmethod
    def compute_pr_auc_and_recall_at_fpr(y_true, y_scores, target_fpr=0.01):
        labels, scores = np.asarray(y_true), np.asarray(y_scores, dtype=float)
        if labels.shape != scores.shape or labels.ndim != 1 or not 0 <= target_fpr <= 1:
            raise ValueError("Invalid metric inputs")
        valid = np.isin(labels, [0, 1]) & np.isfinite(scores)
        labels, scores = labels[valid], scores[valid]
        positives, negatives = np.sum(labels == 1), np.sum(labels == 0)
        if not positives:
            return float("nan"), float("nan")
        order = np.argsort(-scores, kind="stable")
        labels, scores = labels[order], scores[order]
        boundaries = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
        tp = np.cumsum(labels)[boundaries]
        fp = np.cumsum(1 - labels)[boundaries]
        recall = tp / positives
        precision = tp / (tp + fp)
        area = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
        recall_at_fpr = float(np.max(np.r_[0.0, recall[fp / negatives <= target_fpr]])) if negatives else float("nan")
        return area, recall_at_fpr
