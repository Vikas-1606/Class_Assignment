"""Centralized configuration settings for the LSTM Log Anomaly Detection system.

Uses pathlib.Path for cross-platform filesystem management and establishes
production-grade hyperparameters for sequence modeling and anomaly detection.
"""

from pathlib import Path
from typing import Dict, Any

# =====================================================================
# Directory Structure & File Paths (Cookiecutter Data Science v2)
# =====================================================================
SRC_DIR: Path = Path(__file__).resolve().parent
PROJ_ROOT: Path = SRC_DIR.parent

DATA_DIR: Path = PROJ_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DATA_DIR: Path = DATA_DIR / "external"

MODELS_DIR: Path = PROJ_ROOT / "models"
REPORTS_DIR: Path = PROJ_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
NOTEBOOKS_DIR: Path = PROJ_ROOT / "notebooks"
REFERENCES_DIR: Path = PROJ_ROOT / "references"

# File paths
RAW_LOG_FILE: Path = RAW_DATA_DIR / "hdfs_raw.log"
LABEL_FILE: Path = RAW_DATA_DIR / "anomaly_label.csv"
PARSED_EVENTS_FILE: Path = INTERIM_DATA_DIR / "parsed_log_events.csv"
SESSION_SEQUENCES_FILE: Path = INTERIM_DATA_DIR / "session_sequences.json"
TEMPLATES_FILE: Path = INTERIM_DATA_DIR / "event_templates.json"

VOCAB_FILE: Path = PROCESSED_DATA_DIR / "vocab.json"
TRAIN_X_FILE: Path = PROCESSED_DATA_DIR / "X_train.npy"
TRAIN_Y_FILE: Path = PROCESSED_DATA_DIR / "y_train.npy"
VAL_X_FILE: Path = PROCESSED_DATA_DIR / "X_val.npy"
VAL_Y_FILE: Path = PROCESSED_DATA_DIR / "y_val.npy"
TEST_X_FILE: Path = PROCESSED_DATA_DIR / "X_test.npy"
TEST_Y_FILE: Path = PROCESSED_DATA_DIR / "y_test.npy"
TEST_SESSIONS_FILE: Path = PROCESSED_DATA_DIR / "test_sessions.json"

MODEL_SAVE_PATH: Path = MODELS_DIR / "lstm_log_anomaly_model.keras"
METADATA_SAVE_PATH: Path = MODELS_DIR / "model_metadata.json"
EVALUATION_METRICS_FILE: Path = REPORTS_DIR / "evaluation_metrics.json"

# =====================================================================
# Data Ingestion & Preprocessing Parameters
# =====================================================================
RANDOM_SEED: int = 42

# Special tokens
PAD_TOKEN: str = "<PAD>"
UNK_TOKEN: str = "<UNK>"
PAD_INDEX: int = 0
UNK_INDEX: int = 1

# Sequence Modeling Parameters
WINDOW_SIZE: int = 10        # History window length (h in DeepLog paper)
TOP_K: int = 9               # Top-K candidate events considered normal
MIN_SEQ_LEN: int = 3         # Minimum sequence length to process

# Dataset splitting
TRAIN_SPLIT: float = 0.80
VAL_SPLIT: float = 0.10
TEST_SPLIT: float = 0.10

# Anomaly thresholding
# In DeepLog: If actual event is NOT in top-k predicted candidates, flagged as anomaly
ANOMALY_PERCENT_THRESHOLD: float = 0.0  # Session flagged if >= 1 window anomalous

# =====================================================================
# Neural Network Hyperparameters (DeepLog LSTM Architecture)
# =====================================================================
EMBEDDING_DIM: int = 64
LSTM_UNITS: int = 128
NUM_LSTM_LAYERS: int = 2
DENSE_UNITS: int = 64
DROPOUT_RATE: float = 0.20
RECURRENT_DROPOUT: float = 0.0

LEARNING_RATE: float = 0.001
BATCH_SIZE: int = 64
EPOCHS: int = 15
EARLY_STOPPING_PATIENCE: int = 3
REDUCE_LR_PATIENCE: int = 2


def ensure_directories() -> None:
    """Ensure all standard project directories exist on the filesystem."""
    dirs_to_create = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        NOTEBOOKS_DIR,
        REFERENCES_DIR,
    ]
    for directory in dirs_to_create:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_dict() -> Dict[str, Any]:
    """Return a dictionary of the configuration values for logging/metadata."""
    return {
        "WINDOW_SIZE": WINDOW_SIZE,
        "TOP_K": TOP_K,
        "RANDOM_SEED": RANDOM_SEED,
        "TRAIN_SPLIT": TRAIN_SPLIT,
        "VAL_SPLIT": VAL_SPLIT,
        "TEST_SPLIT": TEST_SPLIT,
        "EMBEDDING_DIM": EMBEDDING_DIM,
        "LSTM_UNITS": LSTM_UNITS,
        "NUM_LSTM_LAYERS": NUM_LSTM_LAYERS,
        "DENSE_UNITS": DENSE_UNITS,
        "DROPOUT_RATE": DROPOUT_RATE,
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
    }


if __name__ == "__main__":
    ensure_directories()
    print("Project configuration directories validated and initialized successfully.")
