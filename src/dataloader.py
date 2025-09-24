import pandas as pd
from config import RAW_DATA_DIR


def load_cb513(filename: str = "cb513.csv") -> pd.DataFrame:
    """
    Load the CB513 dataset as a Pandas DataFrame.

    Args:
        filename (str): Name of the file inside RAW_DATA_DIR/CB513/

    Returns:
        pd.DataFrame: Contains protein sequences and secondary structure labels.
    """
    file_path = RAW_DATA_DIR / "CB513" / filename
    df = pd.read_csv(file_path)
    return df
