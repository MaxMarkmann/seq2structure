from pathlib import Path
import pandas as pd
from config import RAW_DATA_DIR


def load_cb513(n_proteins=None):
    """
    Load the CB513 dataset as DataFrame.
    Will look for cb513.csv in data/raw/ and data/raw/CB513/.
    """
    # mögliche Pfade
    candidates = [
        RAW_DATA_DIR / "cb513.csv",
        RAW_DATA_DIR / "CB513" / "cb513.csv",
    ]

    path = None
    for cand in candidates:
        if cand.exists():
            path = cand
            break

    if path is None:
        raise FileNotFoundError(
            f"CB513 dataset not found in any of: {[str(c) for c in candidates]}"
        )

    df = pd.read_csv(path)
    if n_proteins:
        df = df.iloc[:n_proteins]
    return df
