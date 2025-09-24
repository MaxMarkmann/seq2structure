import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
METRICS_FILE = RESULTS_DIR / "metrics.json"


def load_metrics():
    """Load metrics.json as a dict."""
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"{METRICS_FILE} not found. Run training first.")
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def metrics_to_dataframe(metrics_dict):
    """Convert metrics.json (dict) into a summary DataFrame."""
    rows = []
    for model, data in metrics_dict.items():
        rows.append({
            "Model": model,
            "Accuracy": round(data["accuracy"], 4),
        })
    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame):
    """Bar plot comparing model accuracies."""
    plt.figure(figsize=(6, 4))
    plt.bar(df["Model"], df["Accuracy"], color="skyblue")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_comparison.png")
    plt.close()
    print(f"Accuracy comparison plot saved to {RESULTS_DIR / 'accuracy_comparison.png'}")


def main():
    # Load metrics
    metrics = load_metrics()

    # Convert to DataFrame
    df = metrics_to_dataframe(metrics)
    print("\n=== Model Comparison ===")
    print(df)

    # Save as CSV
    summary_file = RESULTS_DIR / "metrics_summary.csv"
    df.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")

    # Plot accuracy comparison
    plot_accuracy_comparison(df)


if __name__ == "__main__":
    main()
