import numpy as np
from config import PROCESSED_DATA_DIR
from pathlib import Path


def load_embeddings(filename: str):
    """
    Load embeddings from .npz file.
    If filename is not an absolute path, look in data/processed/.
    """
    path = Path(filename)
    if not path.is_absolute():
        path = PROCESSED_DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")

    data = np.load(path, allow_pickle=True)
    return data["embeddings"], data["labels"], data["ids"]


def summarize_embeddings(path):
    """
    Gibt eine kurze Zusammenfassung der gespeicherten Embeddings aus.
    """
    X, y, ids = load_embeddings(path)
    print(f"Embeddings loaded from {path}")
    print(f" - Shape: {X.shape}")
    print(f" - Labels: {len(y)}")
    print(f" - IDs: {len(ids)}")
    print(f" - First ID: {ids[0] if len(ids) > 0 else 'None'}")
    return X, y, ids

def load_residue_embeddings(filename="protbert_residue_embeddings.npz"):
    path = PROCESSED_DATA_DIR / filename
    data = np.load(path, allow_pickle=True)

    embeddings = data["embeddings"]   # Liste von Arrays [L_i, 1024]
    labels = data["labels"]           # Liste von Listen [L_i]
    ids = data["ids"]

    # Flatten
    X = np.vstack(embeddings)  # shape (sum(L_i), 1024)
    y = np.concatenate(labels) # shape (sum(L_i),)
    return X, y, ids