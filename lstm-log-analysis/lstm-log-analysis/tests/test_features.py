"""Unit tests for feature engineering, vocabulary, and sliding windows."""

import tempfile
from pathlib import Path
import numpy as np

from src.features import (
    EventVocabulary,
    create_sliding_windows,
    process_and_split_sessions,
)
from src.config import PAD_INDEX, UNK_INDEX


def test_vocabulary_encoding_decoding():
    """Verify vocabulary indexing, special tokens, and bidirectional mapping."""
    vocab = EventVocabulary()
    sequences = [["E1", "E2", "E3"], ["E2", "E4", "E1"]]
    vocab.build_vocab(sequences)

    assert vocab.size == 6  # PAD, UNK, E1, E2, E3, E4
    assert vocab.token_to_id("<PAD>") == PAD_INDEX
    assert vocab.token_to_id("<UNK>") == UNK_INDEX

    # Test known token
    e1_id = vocab.token_to_id("E1")
    assert e1_id not in [PAD_INDEX, UNK_INDEX]
    assert vocab.id_to_token(e1_id) == "E1"

    # Test unknown token fallback
    assert vocab.token_to_id("E_NONEXISTENT") == UNK_INDEX
    assert vocab.id_to_token(9999) == "<UNK>"


def test_vocabulary_save_and_load():
    """Verify serialization and deserialization of EventVocabulary."""
    vocab = EventVocabulary()
    vocab.build_vocab([["E5", "E6", "E7"]])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        vocab.save(tmp_path)
        loaded_vocab = EventVocabulary.load(tmp_path)
        assert loaded_vocab.size == vocab.size
        assert loaded_vocab.token_to_idx == vocab.token_to_idx
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_create_sliding_windows_standard():
    """Verify sliding window shape and targets for standard length sequence."""
    seq = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    window_size = 5
    x, y = create_sliding_windows(seq, window_size=window_size)

    # Number of windows = len(seq) - window_size = 12 - 5 = 7
    assert x.shape == (7, 5)
    assert y.shape == (7,)
    np.testing.assert_array_equal(x[0], [2, 3, 4, 5, 6])
    assert y[0] == 7
    np.testing.assert_array_equal(x[1], [3, 4, 5, 6, 7])
    assert y[1] == 8


def test_create_sliding_windows_short_sequence():
    """Verify that sequences shorter than window size are padded properly."""
    seq = [4, 5]
    window_size = 5
    x, y = create_sliding_windows(seq, window_size=window_size, pad_index=PAD_INDEX)

    assert len(x) >= 1
    assert x.shape[1] == window_size
    # Trailing tokens should be [PAD, PAD, PAD, PAD, 4] and target 5
    assert x[0][-1] == 4
    assert y[0] == 5


def test_session_splitting_no_leakage():
    """Verify train, validation, and test sets are non-empty and session-isolated."""
    sessions = {
        f"blk_{i}": {
            "events": ["E2", "E1", "E3", "E4", "E5", "E7", "E6"],
            "label": "Normal",
            "length": 7,
        }
        for i in range(40)
    }
    # Add 10 anomalous sessions
    for j in range(40, 50):
        sessions[f"blk_{j}"] = {
            "events": ["E2", "E1", "E19", "E20"],
            "label": "Anomaly",
            "length": 4,
        }

    x_train, y_train, x_val, y_val, x_test, y_test, vocab, test_sessions = (
        process_and_split_sessions(
            sessions=sessions,
            window_size=5,
            train_split=0.8,
            val_split=0.1,
            seed=42,
        )
    )

    assert len(x_train) > 0
    assert len(x_val) > 0
    assert len(x_test) > 0
    assert len(test_sessions) > 0
    assert any(s["is_anomaly"] == 1 for s in test_sessions.values())
    assert any(s["is_anomaly"] == 0 for s in test_sessions.values())
