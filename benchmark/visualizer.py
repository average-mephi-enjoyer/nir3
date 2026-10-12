import os
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np


class BenchmarkVisualizer:
    """
    Graphical visualization of benchmark results and stress tests.
    """

    @staticmethod
    def plot_ramp_drift_comparison(
        time_steps: List[int],
        ground_truth_drift: List[float],
        ema_baseline_history: List[float],
        quarantine_baseline_history: List[float],
        output_filepath: str = "output/ramp_drift_comparison.png"
    ) -> None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        plt.figure(figsize=(10, 5), dpi=300)
        plt.plot(time_steps, ground_truth_drift, "k--", label="Истинный уровень аномалии (Ramp Drift)", linewidth=2.0)
        plt.plot(time_steps, ema_baseline_history, "r-", label="Стандартный профиль EMA (Самонормализация/Загрязнение)", linewidth=2.2)
        plt.plot(time_steps, quarantine_baseline_history, "g-", label="Защищенный профиль с карантином (Trusted Baseline)", linewidth=2.5)
        plt.title("Стресс-тестирование базовой линии при медленно развивающейся атаке", fontsize=12, pad=12)
        plt.xlabel("Шаги наблюдения (сутки)", fontsize=11)
        plt.ylabel("Значение базовой нормы признака", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_filepath, bbox_inches="tight")
        plt.close()
        print(f"[+] График сохранен: {output_filepath}")