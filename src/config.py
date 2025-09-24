from pathlib import Path

# Project root = parent of this file's parent (src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create dirs if missing
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)
