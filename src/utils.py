import json
import csv
from pathlib import Path
import numpy as np

RESULTS_FILE = Path("results") / "metrics.json"


def save_metrics(model_name: str, metrics: dict):
    """
    Save metrics to results/metrics.json and confusion matrix to CSV.
    If metrics.json exists, append/update the entry.
    """
    # load existing metrics file
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # add/update current model
    all_metrics[model_name] = {
        "accuracy": float(metrics["accuracy"]),
        "report": metrics["report"],
    }

    # save metrics.json
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Metrics for {model_name} saved to {RESULTS_FILE}")

    # save confusion matrix separately as CSV
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        cm_file = RESULTS_FILE.parent / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.csv"
        with open(cm_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "Pred_H", "Pred_E", "Pred_C"])
            for i, row in enumerate(cm):
                writer.writerow([["True_H", "True_E", "True_C"][i]] + list(row))
        print(f"Confusion matrix for {model_name} saved to {cm_file}")
