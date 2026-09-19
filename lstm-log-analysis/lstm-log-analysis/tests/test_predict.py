"""Unit tests for inference, top-k candidate logic, and metric calculations."""

import numpy as np
from src.features import EventVocabulary
from src.modeling.train import build_deeplog_lstm_model
from src.modeling.predict import (
    predict_window_anomaly,
    predict_single_sequence,
    evaluate_precomputed_sessions,
)


def test_predict_window_anomaly_top_k():
    """Verify top-k classification rule."""
    vocab_size = 10
    window_size = 5
    model = build_deeplog_lstm_model(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=16,
        lstm_units=16,
        num_lstm_layers=1,
    )

    x_window = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    # Predict probabilities directly
    prob_dist = model.predict(x_window[np.newaxis, :], verbose=0)[0]
    top_3_candidates = np.argsort(prob_dist)[::-1][:3]

        # Candidate in top 3 should not be an anomaly
    known_good = top_3_candidates[0]
    is_anom, score, candidates, prob = predict_window_anomaly(
        model, x_window, actual_next_event=known_good, top_k=3
    )
    assert is_anom is False
    assert score <= 1.0

    # Candidate at the bottom should be flagged as anomaly
    bottom_candidate = np.argsort(prob_dist)[0]
    is_anom_bad, score_bad, _, _ = predict_window_anomaly(
        model, x_window, actual_next_event=bottom_candidate, top_k=3
    )
    assert is_anom_bad is True


def test_predict_single_sequence():
    """Verify end-to-end trace evaluation on an ad-hoc sequence."""
    vocab = EventVocabulary()
    vocab.build_vocab([["E1", "E2", "E3", "E4", "E5"]])
    model = build_deeplog_lstm_model(
        vocab_size=vocab.size,
        window_size=5,
        embedding_dim=16,
        lstm_units=16,
        num_lstm_layers=1,
    )

    trace = ["E1", "E2", "E3", "E4", "E5", "E1"]
    res = predict_single_sequence(model, vocab, trace, window_size=5, top_k=vocab.size)
    assert "is_anomaly" in res
    assert "session_anomaly_score" in res
    assert "total_steps" in res
    # With top_k == vocab_size, all observed events must be allowed
    assert res["is_anomaly"] is False


def test_evaluate_precomputed_sessions():
    """Verify metric computation over synthetic precomputed sessions."""
    precomputed = [
        {
            "block_id": "blk_1",
            "targets": np.array([2, 3]),
            "probs": np.array([
                [0.0, 0.0, 0.9, 0.1],  # 2 is in top-1
                [0.0, 0.0, 0.1, 0.9],  # 3 is in top-1
            ]),
            "ground_truth": 0,
        },
        {
            "block_id": "blk_2",
            "targets": np.array([1]),
            "probs": np.array([
                [0.0, 0.0, 0.9, 0.1],  # 1 has prob 0.0 (not in top-1)
            ]),
            "ground_truth": 1,
        },
    ]

    metrics = evaluate_precomputed_sessions(precomputed, top_k=1)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["confusion_matrix"]["true_positives"] == 1
    assert metrics["confusion_matrix"]["true_negatives"] == 1
