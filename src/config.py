from pathlib import Path

# Project root = one level above src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results"

# (Optional) Notebooks directory
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they do not exist
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, NOTEBOOKS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Convenience: print project root once when imported (debugging)
if __name__ == "__main__":
    print("Project root:", PROJECT_ROOT)
    print("Data dir:", DATA_DIR)
    print("Results dir:", RESULTS_DIR)
    print("Notebooks dir:", NOTEBOOKS_DIR)
