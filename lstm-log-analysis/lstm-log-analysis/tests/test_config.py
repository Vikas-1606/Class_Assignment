"""Unit tests for configuration parameters and path management."""

from pathlib import Path
from src.config import (
    PROJ_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    WINDOW_SIZE,
    TOP_K,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    EMBEDDING_DIM,
    LSTM_UNITS,
    NUM_LSTM_LAYERS,
    LEARNING_RATE,
    ensure_directories,
    get_config_dict,
)


def test_paths_are_pathlib_instances():
    """Verify all defined directory paths are Path instances."""
    assert isinstance(PROJ_ROOT, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(RAW_DATA_DIR, Path)
    assert isinstance(INTERIM_DATA_DIR, Path)
    assert isinstance(PROCESSED_DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    assert isinstance(REPORTS_DIR, Path)
    assert isinstance(FIGURES_DIR, Path)


def test_ensure_directories_creates_all_dirs():
    """Verify ensure_directories creates required directories."""
    ensure_directories()
    assert RAW_DATA_DIR.exists()
    assert INTERIM_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert MODELS_DIR.exists()
    assert REPORTS_DIR.exists()
    assert FIGURES_DIR.exists()


def test_hyperparameter_validity():
    """Verify parameters are within valid numerical bounds."""
    assert WINDOW_SIZE > 0
    assert TOP_K > 0
    assert 0.0 < TRAIN_SPLIT < 1.0
    assert 0.0 < VAL_SPLIT < 1.0
    assert 0.0 < TEST_SPLIT < 1.0
    assert abs((TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT) - 1.0) < 1e-5
    assert EMBEDDING_DIM > 0
    assert LSTM_UNITS > 0
    assert NUM_LSTM_LAYERS >= 1
    assert LEARNING_RATE > 0


def test_config_dictionary():
    """Verify get_config_dict returns required metadata keys."""
    cfg = get_config_dict()
    assert "WINDOW_SIZE" in cfg
    assert "TOP_K" in cfg
    assert "EMBEDDING_DIM" in cfg
    assert "LSTM_UNITS" in cfg
    assert cfg["WINDOW_SIZE"] == WINDOW_SIZE
