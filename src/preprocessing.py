import numpy as np
from typing import Tuple

# Standard amino acids (20)
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Map 3-class secondary structure
SS3_TO_INT = {"H": 0, "E": 1, "C": 2}


def one_hot_encode_aa(aa: str) -> np.ndarray:
    """One-hot encode a single amino acid (size 20)."""
    vec = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    if aa in AA_TO_INT:
        vec[AA_TO_INT[aa]] = 1.0
    return vec


def preprocess_dataset(df, window_size: int = 17) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert sequences and labels into X, y arrays using a sliding window.
    Also returns groups (protein IDs) for GroupKFold.

    Args:
        df (pd.DataFrame): DataFrame with 'seq', 'sst3' and 'pdb_id'
        window_size (int): size of sliding window (must be odd)

    Returns:
        X (np.ndarray): shape (N, window_size*20) → features
        y (np.ndarray): shape (N,) → labels (0=H, 1=E, 2=C)
        groups (np.ndarray): shape (N,) → protein IDs (for GroupKFold)
    """
    half_win = window_size // 2
    X, y, groups = [], [], []

    for pdb_id, seq, labels in zip(df["pdb_id"], df["seq"], df["sst3"]):
        seq = seq.upper()
        labels = labels.upper()

        # pad sequence with "X" (unknown AA) for windowing
        padded_seq = "X" * half_win + seq + "X" * half_win

        for i in range(len(seq)):
            window = padded_seq[i : i + window_size]

            # encode window (concat one-hot of each AA)
            window_vec = np.concatenate([one_hot_encode_aa(aa) for aa in window])
            X.append(window_vec)

            # encode label of the central AA
            y.append(SS3_TO_INT[labels[i]])

            # group = protein id for this residue
            groups.append(pdb_id)

    return np.array(X), np.array(y), np.array(groups)
