import json
import csv
from pathlib import Path
import numpy as np

# always resolve project root (not src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "metrics.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_metrics(model_name: str, metrics: dict):
    """Save metrics JSON + confusion matrix CSV in results/ (project root)."""
    # load existing metrics file
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, "r") as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # kopieren, aber confusion_matrix nur für top-level Modelle speichern
    entry = {
        "accuracy": float(metrics["accuracy"]),
        "report": metrics["report"],
    }

    if "confusion_matrix" in metrics and isinstance(metrics["confusion_matrix"], (list, np.ndarray)):
        entry["confusion_matrix"] = metrics["confusion_matrix"]

    if "folds" in metrics:  # GroupKFold-Sonderfall
        entry["folds"] = metrics["folds"]

    all_metrics[model_name] = entry

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"Metrics for {model_name} saved to {RESULTS_FILE}")

    # confusion matrix als CSV nur wenn wirklich eine Matrix existiert
    cm = metrics.get("confusion_matrix")
    if isinstance(cm, (list, np.ndarray)):
        cm_file = RESULTS_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.csv"
        with open(cm_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["", "Pred_H", "Pred_E", "Pred_C"])
            for i, row in enumerate(cm):
                writer.writerow([["True_H", "True_E", "True_C"][i]] + list(row))
        print(f"Confusion matrix for {model_name} saved to {cm_file}")
