import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from preprocessing import preprocess_dataset
from dataloader import load_cb513
from embedding_utils import load_embeddings, load_residue_embeddings
from train import (
    train_baseline,
    train_random_forest,
    train_mlp,
    train_with_embeddings,
    train_residue_embeddings,
)
from config import PROCESSED_DATA_DIR, RESULTS_DIR


def evaluate_and_save(name, y_true, y_pred, labels):
    """Speichert Report + Confusion Matrix + CSV + Plot in einem Unterordner je Variante"""
    variant_dir = RESULTS_DIR / name
    os.makedirs(variant_dir, exist_ok=True)

    # === Classification Report ===
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(variant_dir / "report.csv")

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Confusion Matrix: {name}")
    plt.savefig(variant_dir / "confusion.png", bbox_inches="tight")
    plt.close()

    return report


def benchmark():
    results = []

    # === Sliding Window Encodings ===
    df = load_cb513()
    for encoding in ["onehot"]:  # ggf. erweitern
        X, y, groups = preprocess_dataset(df, encoding=encoding, window_size=17)

        for model_name, train_func in [
            ("logreg", train_baseline),
            ("rf", train_random_forest),
            ("mlp", lambda X, y, groups=None: train_mlp(X, y, groups=groups, epochs=5)),
        ]:
            name = f"{encoding}_{model_name}"
            start = time.time()
            y_true, y_pred, labels = train_func(X, y, groups=groups)  # jetzt korrekt
            duration = time.time() - start

            report = evaluate_and_save(name, y_true, y_pred, labels)
            results.append({"variant": name, "time": duration, "accuracy": report["accuracy"]})

    # === Protein-level Embeddings ===
    X_emb, y_emb, ids_emb = load_embeddings(PROCESSED_DATA_DIR / "protbert_full.npz")
    for model in ["logreg", "rf", "mlp"]:
        name = f"protbert_{model}"
        start = time.time()
        y_true, y_pred, labels = train_with_embeddings(
            X_emb, y_emb, ids_emb, model_type=model, use_groups=True
        )
        duration = time.time() - start

        report = evaluate_and_save(name, y_true, y_pred, labels)
        results.append({"variant": name, "time": duration, "accuracy": report["accuracy"]})

    # === Residue-level Embeddings ===
    X_res, y_res, ids_res = load_residue_embeddings(PROCESSED_DATA_DIR / "protbert_residues_full.npz")
    for model in ["logreg", "rf", "mlp"]:
        name = f"residues_{model}"
        start = time.time()
        y_true, y_pred, labels = train_residue_embeddings(
            X_res, y_res, model_type=model, use_groups=True
        )
        duration = time.time() - start

        report = evaluate_and_save(name, y_true, y_pred, labels)
        results.append({"variant": name, "time": duration, "accuracy": report["accuracy"]})

    # === Save summary in main results folder ===
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(RESULTS_DIR / "benchmark_summary.csv", index=False)

    # === Plot accuracy + runtime ===
    plt.figure(figsize=(10, 6))
    plt.bar(df_summary["variant"], df_summary["accuracy"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy")
    plt.title("Benchmark Accuracy Comparison")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_accuracy.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.bar(df_summary["variant"], df_summary["time"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Training Time (s)")
    plt.title("Benchmark Runtime Comparison")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "summary_runtime.png")
    plt.close()


if __name__ == "__main__":
    benchmark()
