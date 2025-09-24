from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from utils import save_metrics


def _evaluate_model(clf, X_test, y_test, name: str):
    """Evaluate a trained classifier and save confusion matrix plot."""
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["H", "E", "C"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # plot confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["H", "E", "C"], yticklabels=["H", "E", "C"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    fig_path = f"results/confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


def train_baseline(X, y, test_size: float = 0.2, random_state: int = 42):
    """Train Logistic Regression baseline model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X_train, y_train)

    metrics = _evaluate_model(clf, X_test, y_test, name="Logistic Regression")
    save_metrics("Logistic Regression", metrics)
    return clf, metrics


def train_random_forest(X, y, test_size: float = 0.2, random_state: int = 42):
    """Train Random Forest baseline model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=random_state
    )
    clf.fit(X_train, y_train)

    metrics = _evaluate_model(clf, X_test, y_test, name="Random Forest")
    save_metrics("Random Forest", metrics)
    return clf, metrics
