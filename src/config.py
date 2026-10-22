from pathlib import Path

# Base Paths
ROOT_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Large artifact paths — resolved via LOCAL-FIRST / Hugging Face fallback.
# Import from artifacts.py; do not hard-code .pt paths directly here.
# ---------------------------------------------------------------------------
from src.artifacts import (  # noqa: E402  (import after constant ROOT_DIR)
    get_model_path,
    get_reference_embeddings_path,
)

MODEL_PATH                = get_model_path()
REFERENCE_EMBEDDINGS_PATH = get_reference_embeddings_path()

# Small data files that are always local (tracked in git)
METADATA_PATH     = ROOT_DIR / "data" / "final_metadata.csv"
GEOCELL_TREE_PATH = ROOT_DIR / "data" / "geocell_tree.json"

# Locked ML Default Parameters
TOP_GEOCELLS = 10
RETRIEVAL_K = 100
DBSCAN_EPS_KM = 100.0
DBSCAN_MIN_SAMPLES = 3
SIMILARITY_TEMPERATURE = 0.05
USE_CLASSIFIER_PRIOR = False
ALPHA = 1.0
BETA = 1.0
METHOD = "dbscan_v2"
