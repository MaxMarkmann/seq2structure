import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import RESULTS_DIR

METRICS_FILE = RESULTS_DIR / "metrics.json"


def load_metrics():
    """Load metrics.json as a dict."""
    if not METRICS_FILE.exists():
        raise FileNotFoundError(f"{METRICS_FILE} not found. Run training first.")
    with open(METRICS_FILE, "r") as f:
        return json.load(f)


def metrics_to_dataframe(metrics_dict):
    """Convert metrics.json (dict) into summary DataFrame (Accuracy only)."""
    rows = []
    for model, data in metrics_dict.items():
        rows.append({
            "Model": model,
            "Accuracy": round(data["accuracy"], 4),
        })
    return pd.DataFrame(rows)


def plot_accuracy_comparison(df: pd.DataFrame):
    """Bar plot comparing model accuracies."""
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="Model", y="Accuracy", palette="Blues_d")
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_file = RESULTS_DIR / "accuracy_comparison.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Accuracy comparison plot saved to {out_file}")


def plot_per_class_f1(metrics_dict):
    """Plot per-class F1 scores for each model (H/E/C)."""
    rows = []
    for model, vals in metrics_dict.items():
        report = vals["report"]
        if isinstance(report, str):
            lines = [l for l in report.split("\n") if l.strip()]
            for line in lines[2:5]:  # only classes H/E/C
                parts = line.split()
                cls = parts[0]
                precision, recall, f1 = map(float, parts[1:4])
                rows.append({"Model": model, "Class": cls, "F1": f1})

    df = pd.DataFrame(rows)
    if df.empty:
        print("No per-class F1 scores found.")
        return None

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y="F1", hue="Class", palette="Set2")
    plt.ylim(0, 1)
    plt.ylabel("F1-Score")
    plt.title("Per-Class F1-Scores (H/E/C)")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Class")
    plt.tight_layout()
    out_file = RESULTS_DIR / "per_class_f1_scores.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Per-class F1 plot saved to {out_file}")

    return df


def plot_confusion_matrices(metrics_dict):
    """Plot confusion matrices of all models in a grid (only those available)."""
    available = [(model, vals) for model, vals in metrics_dict.items()
                 if "confusion_matrix" in vals and isinstance(vals["confusion_matrix"], (list, np.ndarray))]
    if not available:
        print("No confusion matrices found in metrics.json.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (model, vals) in zip(axes, available):
        cm = np.array(vals["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["H", "E", "C"],
                    yticklabels=["H", "E", "C"],
                    ax=ax)
        ax.set_title(model)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    out_file = RESULTS_DIR / "all_confusion_matrices.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Confusion matrices plot saved to {out_file}")


def save_summary_csv(acc_df: pd.DataFrame, f1_df: pd.DataFrame):
    """Merge accuracy and per-class F1 into one summary table."""
    if f1_df is None:
        summary_file = RESULTS_DIR / "metrics_summary.csv"
        acc_df.to_csv(summary_file, index=False)
        print(f"Summary saved to {summary_file} (accuracy only)")
        return

    # Pivot F1: Model × Class
    f1_pivot = f1_df.pivot(index="Model", columns="Class", values="F1").reset_index()
    summary = pd.merge(acc_df, f1_pivot, on="Model", how="left")

    summary_file = RESULTS_DIR / "metrics_summary.csv"
    summary.to_csv(summary_file, index=False)
    print(f"Summary saved to {summary_file}")


def main():
    # Load metrics
    metrics = load_metrics()

    print("\nLoaded metrics from:", METRICS_FILE)
    for model, vals in metrics.items():
        print(f"{model}: {vals['accuracy']:.4f}")

    # Accuracy summary
    acc_df = metrics_to_dataframe(metrics)
    plot_accuracy_comparison(acc_df)

    # Per-class F1
    f1_df = plot_per_class_f1(metrics)

    # Confusion matrices
    plot_confusion_matrices(metrics)

    # Save combined summary CSV
    save_summary_csv(acc_df, f1_df)


if __name__ == "__main__":
    main()