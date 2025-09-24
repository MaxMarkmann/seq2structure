from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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



class MLP(nn.Module):
    """Simple feed-forward neural network for classification."""

    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(X, y, test_size=0.2, random_state=42, epochs=5, batch_size=512, lr=1e-3, device=None):
    """
    Train an MLP model on the dataset.

    Args:
        X (np.ndarray): feature matrix
        y (np.ndarray): labels
        epochs (int): number of training epochs
        batch_size (int): batch size
        lr (float): learning rate
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training MLP on {device}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # DataLoader
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # Model
    model = MLP(input_dim=X.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == yb).sum().item()
            total += yb.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    accuracy = correct / total
    print(f"MLP Accuracy: {accuracy:.4f}")

    # Classification report + confusion matrix (reuse sklearn)
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds, target_names=["H", "E", "C"])
    cm = confusion_matrix(all_labels, all_preds)

    print(report)

    # Save metrics
    metrics = {"accuracy": accuracy, "report": report, "confusion_matrix": cm.tolist()}
    save_metrics("MLP", metrics)

    return model, metrics


def _evaluate_model(clf, X_test, y_test, name: str):
    """Evaluate a trained classifier and save confusion matrix plot."""
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["H", "E", "C"])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(report)

    # ensure results directory exists
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

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
    fig_path = results_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
