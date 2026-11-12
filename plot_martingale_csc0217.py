from pathlib import Path
import matplotlib.pyplot as plt
from main import run_cert_dataset_evaluation


def main():
    _, traces, observations = run_cert_dataset_evaluation(target_user="csc0217", use_conformal=True)
    trace = traces["Protected"]
    dates = [observation.timestamp for observation in observations]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    top.plot(dates, trace.sequential_scores, label="Empirical sequential score")
    top.axhline(100.0, color="crimson", linestyle="--", label="Empirical stopping threshold")
    top.set_yscale("log")
    top.set_title("CSC0217: empirical sequential monitoring; no anytime FPR guarantee")
    for marker in ("llm_alienation", "llm_conflict", "llm_access_focus"):
        bottom.plot(dates, [getattr(observation.markers, marker) for observation in observations], label=marker)
    for observation in trace.alerts:
        top.axvline(observation.timestamp, color="red", linestyle=":")
    for axis in (top, bottom):
        axis.legend()
        axis.grid(alpha=0.3)
    fig.tight_layout()
    destination = Path("output/csc0217_martingale.png")
    destination.parent.mkdir(exist_ok=True)
    fig.savefig(destination, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
