import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import networkx as nx


class BenchmarkVisualizer:
    @staticmethod
    def plot_communication_graph(
        ego_graph: nx.DiGraph,
        target_user: str,
        output_filepath: str = "output/communication_ego_graph.png"
    ) -> None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        fig = plt.figure(figsize=(9, 8), dpi=300)
        if ego_graph.number_of_nodes() == 0:
            plt.close(fig)
            return
        pos = nx.spring_layout(ego_graph, seed=42, k=0.8)
        node_colors = [
            "crimson" if node == target_user.lower() else "skyblue"
            for node in ego_graph.nodes()
        ]
        node_sizes = [
            1200 if node == target_user.lower() else 500
            for node in ego_graph.nodes()
        ]
        nx.draw_networkx_nodes(ego_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
        nx.draw_networkx_edges(
            ego_graph, pos,
            arrowstyle="->",
            arrowsize=12,
            edge_color="gray",
            alpha=0.6,
            connectionstyle="arc3,rad=0.1"
        )
        nx.draw_networkx_labels(ego_graph, pos, font_size=8, font_family="sans-serif")
        plt.title(f"Dynamic Ego-Network for Subject: {target_user}", fontsize=12, fontweight="bold", pad=15)
        plt.axis("off")
        fig.tight_layout()
        fig.savefig(output_filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_top_markers_dynamics(
        time_steps: List[int],
        marker_histories: Dict[str, List[float]],
        baseline_histories: Dict[str, List[float]],
        top_markers: Optional[List[str]] = None,
        output_filepath: str = "output/top_markers_dynamics.png"
    ) -> None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        if not top_markers:
            top_markers = ["llm_access_focus", "llm_alienation", "pronoun_marker"]
        valid_markers = [m for m in top_markers if m in marker_histories]
        if not valid_markers:
            return
        fig, axes = plt.subplots(
            nrows=len(valid_markers),
            ncols=1,
            figsize=(10, 2.8 * len(valid_markers)),
            dpi=300,
            sharex=True
        )
        axes_list = [axes] if len(valid_markers) == 1 else list(axes)
        for ax, marker_name in zip(axes_list, valid_markers):
            obs_vals = marker_histories[marker_name]
            base_vals = baseline_histories.get(marker_name, [0.0] * len(obs_vals))
            ax.plot(time_steps, obs_vals, label="Observed Signal", color="navy", alpha=0.65, linewidth=1.2)
            ax.plot(time_steps, base_vals, label="Trusted Baseline Norm", color="darkgreen", linewidth=2.0)
            ax.set_title(f"Psycholinguistic Drift: {marker_name}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Intensity", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8)
        axes_list[-1].set_xlabel("Observation Timeline (Days)", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_multi_marker_dynamics(
        time_steps: List[int],
        marker_histories: Dict[str, List[float]],
        trusted_baseline_histories: Dict[str, List[float]],
        output_filepath: str = "output/multi_marker_dynamics.png"
    ) -> None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        tracked_keys = list(marker_histories.keys())
        num_markers = len(tracked_keys)
        if num_markers == 0:
            return
        nrows = (num_markers + 1) // 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(14, 3 * nrows),
            dpi=300,
            sharex=True
        )
        axes_flat = axes.flatten()
        for idx, key in enumerate(tracked_keys):
            ax = axes_flat[idx]
            observed_vals = marker_histories[key]
            baseline_vals = trusted_baseline_histories.get(key, [0.0] * len(observed_vals))
            ax.plot(time_steps, observed_vals, label="Observed Value", color="royalblue", alpha=0.7, linewidth=1.2)
            ax.plot(time_steps, baseline_vals, label="Trusted Baseline", color="forestgreen", linewidth=2.0)
            ax.set_title(f"Marker: {key}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Intensity", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8)
        for idx in range(num_markers, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
        fig.text(0.5, 0.01, "Observation Timeline (Days)", ha="center", fontsize=10)
        fig.tight_layout()
        fig.savefig(output_filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_all_markers_grid(
        time_steps: List[int],
        marker_histories: Dict[str, List[float]],
        baseline_histories: Dict[str, List[float]],
        output_filepath: str = "output/cert_all_markers.png"
    ) -> None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        tracked_keys = list(marker_histories.keys())
        num_markers = len(tracked_keys)
        if num_markers == 0:
            return
        nrows = (num_markers + 1) // 2
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=2,
            figsize=(15, 3.2 * nrows),
            dpi=300,
            sharex=True
        )
        axes_flat = axes.flatten()
        for idx, key in enumerate(tracked_keys):
            ax = axes_flat[idx]
            obs_vals = marker_histories[key]
            base_vals = baseline_histories.get(key, [0.0] * len(obs_vals))
            ax.plot(time_steps, obs_vals, label="Observed Signal (Individual Message)", color="royalblue", alpha=0.55, linewidth=1.0)
            ax.plot(time_steps, base_vals, label="Trusted Baseline Norm", color="darkgreen", linewidth=2.0)
            ax.set_title(f"Marker: {key}", fontsize=10, fontweight="bold")
            ax.set_ylabel("Intensity", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper left", fontsize=8)
        for idx in range(num_markers, len(axes_flat)):
            fig.delaxes(axes_flat[idx])
        fig.text(0.5, 0.01, "Observation Steps (Analyzed Sent Emails)", ha="center", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def plot_ramp_drift_comparison(
        time_steps: List[int],
        ground_truth_drift: List[float],
        ema_baseline_history: List[float],
        quarantine_baseline_history: List[float],
        risk_scores: List[float],
        alert_thresholds: List[float],
        alert_day: Optional[int] = None,
        technical_attack_day: Optional[int] = None,
        output_filepath: str = "output/ramp_drift_comparison.png"
    ) -> None:
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(11, 7), dpi=300, sharex=True)
        ax1.plot(time_steps, ground_truth_drift, "k--", label="True Anomaly Drift (Ground Truth)", linewidth=1.8)
        ax1.plot(time_steps, ema_baseline_history, "r-", label="Unprotected EMA (Baseline Poisoning)", linewidth=2.0)
        ax1.plot(time_steps, quarantine_baseline_history, "g-", label="Quarantine-Guarded Baseline", linewidth=2.2)
        ax1.set_title("Dynamic Baseline Protection under Ramp Drift Attack", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Marker Value", fontsize=10)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper left", fontsize=9)
        ax2.plot(time_steps, risk_scores, color="darkorange", label="Composite Risk R(t)", linewidth=2.0)
        ax2.plot(time_steps, alert_thresholds, color="crimson", linestyle="-.", label="Dynamic Alert Threshold", linewidth=1.5)
        if alert_day is not None:
            ax2.axvline(x=alert_day, color="red", linestyle=":", linewidth=1.8, label=f"First Warning (Day {alert_day})")
        if technical_attack_day is not None:
            ax2.axvline(x=technical_attack_day, color="black", linestyle="-", linewidth=2.0, label=f"Host Action (Day {technical_attack_day})")
        if alert_day is not None and technical_attack_day is not None and technical_attack_day > alert_day:
            ax2.axvspan(alert_day, technical_attack_day, color="lightgreen", alpha=0.3, label=f"Lead Time Window ({technical_attack_day - alert_day} days)")
        ax2.set_title("Early Warning Trajectory and Retrospective Verification Trigger", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Observation Steps (Days)", fontsize=10)
        ax2.set_ylabel("Risk Score / Threshold", fontsize=10)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc="upper left", fontsize=9)
        fig.tight_layout()
        fig.savefig(output_filepath, bbox_inches="tight")
        plt.close(fig)
