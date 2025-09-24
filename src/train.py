from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def train_baseline(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Train a simple baseline classifier (Logistic Regression) 
    and evaluate with accuracy, classification report, and confusion matrix.

    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): label vector
        test_size (float): fraction of test split
        random_state (int): random seed

    Returns:
        model: trained LogisticRegression model
        metrics (dict): accuracy, classification report, confusion matrix
    """
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train Logistic Regression
    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["H", "E", "C"])
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)

    # Plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["H", "E", "C"], yticklabels=["H", "E", "C"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Baseline Logistic Regression)")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.close()

    return clf, {"accuracy": acc, "report": report, "confusion_matrix": cm}
