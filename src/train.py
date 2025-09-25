import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import TensorDataset, DataLoader, random_split

from config import RESULTS_DIR
from utils import save_metrics


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------

def _plot_confusion_matrix(cm, name: str):
    """Save confusion matrix plot into results/."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["H", "E", "C"], yticklabels=["H", "E", "C"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    fig_path = RESULTS_DIR / f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()


def _evaluate_model(clf, X_test, y_test, name: str):
    """Evaluate sklearn model and save confusion matrix."""
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["H", "E", "C"])
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])  # fix

    print(f"\n=== {name} Results ===")
    print(f"Accuracy: {acc:.4f}\n{report}")

    _plot_confusion_matrix(cm, name)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm.tolist()}


# ----------------------------------------------------------------------
# Logistic Regression (Baseline)
# ----------------------------------------------------------------------

def train_baseline(X, y, test_size: float = 0.2, random_state: int = 42):
    """Train baseline Logistic Regression."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = LogisticRegression(max_iter=200, n_jobs=-1)
    clf.fit(X_train, y_train)

    metrics = _evaluate_model(clf, X_test, y_test, name="Logistic Regression")
    save_metrics("Logistic Regression", metrics)
    return clf, metrics


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------

def train_random_forest(X, y, test_size: float = 0.2, random_state: int = 42):
    """Train Random Forest model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=20, n_jobs=-1, random_state=random_state
    )
    clf.fit(X_train, y_train)

    metrics = _evaluate_model(clf, X_test, y_test, name="Random Forest")
    save_metrics("Random Forest", metrics)
    return clf, metrics


# ----------------------------------------------------------------------
# MLP (PyTorch)
# ----------------------------------------------------------------------

class MLP(nn.Module):
    """Simple feed-forward neural network."""

    def __init__(self, input_dim, hidden_dim=512, num_classes=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X, y,
    test_size=0.2, random_state=42,
    epochs=10, batch_size=1024, lr=3e-4,
    hidden_dim=512, dropout=0.4,
    early_stopping_patience=3,
    device=None
):
    """Train an MLP with early stopping, validation split, and AMP."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training MLP on {device}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long)

    # Validation split (10%)
    val_size = max(1, int(0.1 * len(X_train)))
    train_size = len(X_train) - val_size
    train_ds, val_ds = random_split(
        TensorDataset(X_train, y_train),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size)

    # Model
    model = MLP(input_dim=X.shape[1], hidden_dim=hidden_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.1)
    scaler = GradScaler("cuda" if device == "cuda" else "cpu", enabled=(device == "cuda"))

    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    patience = early_stopping_patience

    # Training loop
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        total_v = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                total_v += loss.item()
        avg_val = total_v / len(val_loader)
        scheduler.step()

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        print(f"Epoch {epoch}/{epochs} | train loss {avg_train:.4f} | val loss {avg_val:.4f}")

        # Early stopping
        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience = early_stopping_patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping triggered.")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Test evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            with autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    acc = (np.array(all_preds) == np.array(all_labels)).mean()
    report = classification_report(all_labels, all_preds, target_names=["H","E","C"])
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])  # fix

    print(f"MLP Accuracy: {acc:.4f}\n{report}")
    _plot_confusion_matrix(cm, "MLP")

    # Learning curve plot
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Learning Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mlp_learning_curves.png")
    plt.close()

    metrics = {"accuracy": float(acc), "report": report, "confusion_matrix": cm.tolist()}
    save_metrics("MLP", metrics)
    return model, metrics


# ----------------------------------------------------------------------
# Logistic Regression mit GroupKFold
# ----------------------------------------------------------------------

def train_baseline_groupkfold(X, y, groups, n_splits: int = 5):
    """Train Logistic Regression with GroupKFold (protein-level split)."""
    gkf = GroupKFold(n_splits=n_splits)

    all_metrics = []
    fold = 1
    for train_idx, test_idx in gkf.split(X, y, groups):
        print(f"\n=== Fold {fold}/{n_splits} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(X_train, y_train)

        metrics = _evaluate_model(clf, X_test, y_test, name=f"LogReg (Fold {fold})")
        all_metrics.append(metrics)
        fold += 1

    # Mittelwert über alle Folds
    avg_acc = np.mean([m["accuracy"] for m in all_metrics])
    print(f"\n=== GroupKFold Average Accuracy: {avg_acc:.4f} ===")

    # Speichern
    save_metrics("Logistic Regression (GroupKFold)", {
        "accuracy": avg_acc,
        "report": "Cross-validation results saved per fold",
        "confusion_matrix": "see fold plots"
    })

    return clf, all_metrics