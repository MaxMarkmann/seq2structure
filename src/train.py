import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ----------------------------------------------------------------------
# Logistic Regression (Baseline)
# ----------------------------------------------------------------------
def train_baseline(X, y, groups=None, test_size: float = 0.2, random_state: int = 42):
    """Train Logistic Regression, optional GroupKFold."""
    if groups is not None:
        gkf = GroupKFold(n_splits=5)
        all_preds, all_labels = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            print(f"\n=== LogReg Fold {fold} ===")
            clf = LogisticRegression(max_iter=200, n_jobs=-1)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            all_preds.extend(preds)
            all_labels.extend(y[test_idx])
        return np.array(all_labels), np.array(all_preds), np.unique(y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        clf = LogisticRegression(max_iter=200, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_test, y_pred, np.unique(y)


# ----------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------
def train_random_forest(X, y, groups=None, test_size: float = 0.2, random_state: int = 42):
    """Train Random Forest, optional GroupKFold."""
    if groups is not None:
        gkf = GroupKFold(n_splits=5)
        all_preds, all_labels = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            print(f"\n=== RF Fold {fold} ===")
            clf = RandomForestClassifier(
                n_estimators=200, max_depth=20, n_jobs=-1, random_state=random_state
            )
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            all_preds.extend(preds)
            all_labels.extend(y[test_idx])
        return np.array(all_labels), np.array(all_preds), np.unique(y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=20, n_jobs=-1, random_state=random_state
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_test, y_pred, np.unique(y)


# ----------------------------------------------------------------------
# MLP (sklearn für Benchmark)
# ----------------------------------------------------------------------
def train_mlp(X, y, groups=None, test_size=0.2, random_state=42, epochs=20, batch_size=512):
    """Train sklearn MLPClassifier, optional GroupKFold."""
    if groups is not None:
        gkf = GroupKFold(n_splits=5)
        all_preds, all_labels = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            print(f"\n=== MLP Fold {fold} ===")
            clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=epochs, verbose=True)
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            all_preds.extend(preds)
            all_labels.extend(y[test_idx])
        return np.array(all_labels), np.array(all_preds), np.unique(y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=epochs, verbose=True)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        return y_test, y_pred, np.unique(y)


# ----------------------------------------------------------------------
# Helper for embeddings
# ----------------------------------------------------------------------
def _train_and_eval(X_train, y_train, X_test, y_test, model_type):
    if model_type == "logreg":
        model = LogisticRegression(max_iter=200, n_jobs=-1)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    elif model_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=50, verbose=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"Fitting {model_type.upper()}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return y_test, preds


# ----------------------------------------------------------------------
# Protein-level Embeddings
# ----------------------------------------------------------------------
def train_with_embeddings(X, y, ids, model_type="logreg", use_groups=True):
    """Train models on protein-level embeddings."""
    models = ["logreg", "rf", "mlp"] if model_type == "all" else [model_type]

    for m in models:
        print(f"\n=== Training {m.upper()} ===")
        if use_groups:
            groups = np.array([seq_id.split("_")[0] for seq_id in ids])
            gkf = GroupKFold(n_splits=5)
            all_preds, all_labels = [], []
            for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
                print(f"--- Fold {fold} ---")
                y_test, preds = _train_and_eval(X[train_idx], y[train_idx], X[test_idx], y[test_idx], m)
                all_preds.extend(preds)
                all_labels.extend(y_test)
            return np.array(all_labels), np.array(all_preds), np.unique(y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            y_test, preds = _train_and_eval(X_train, y_train, X_test, y_test, m)
            return y_test, preds, np.unique(y)


# ----------------------------------------------------------------------
# Residue-level Embeddings
# ----------------------------------------------------------------------
def train_residue_embeddings(X, y, model_type="mlp", use_groups=False):
    """Train models on residue-level embeddings (optional GroupKFold)."""
    if use_groups:
        groups = np.arange(len(y))  # placeholder falls echte Gruppen vorhanden
        gkf = GroupKFold(n_splits=5)
        all_preds, all_labels = [], []
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
            print(f"=== Residue {model_type.upper()} Fold {fold} ===")
            if model_type == "logreg":
                clf = LogisticRegression(max_iter=500, n_jobs=-1)
            elif model_type == "rf":
                clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
            elif model_type == "mlp":
                clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=20, verbose=True)
            else:
                raise ValueError(f"Unknown model_type={model_type}")
            clf.fit(X[train_idx], y[train_idx])
            preds = clf.predict(X[test_idx])
            all_preds.extend(preds)
            all_labels.extend(y[test_idx])
        return np.array(all_labels), np.array(all_preds), np.unique(y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        if model_type == "logreg":
            clf = LogisticRegression(max_iter=500, n_jobs=-1)
        elif model_type == "rf":
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
        elif model_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=20, verbose=True)
        else:
            raise ValueError(f"Unknown model_type={model_type}")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return y_test, preds, np.unique(y)
