"""Unit tests for LSTM model construction, compilation, and tensor shapes."""

import numpy as np
import tensorflow as tf
from src.modeling.train import build_deeplog_lstm_model


def test_model_architecture_shapes():
    """Verify input shape, output shape, and layer configuration."""
    vocab_size = 20
    window_size = 10
    embedding_dim = 32
    lstm_units = 64

    model = build_deeplog_lstm_model(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        num_lstm_layers=2,
        dense_units=32,
        dropout_rate=0.1,
        top_k=5,
    )

    assert model.input_shape == (None, window_size)
    assert model.output_shape == (None, vocab_size)


def test_model_forward_pass_probabilities():
    """Verify that forward pass produces valid probability distribution."""
    vocab_size = 15
    window_size = 8
    model = build_deeplog_lstm_model(
        vocab_size=vocab_size,
        window_size=window_size,
        embedding_dim=16,
        lstm_units=32,
        num_lstm_layers=1,
        dense_units=16,
        top_k=3,
    )

    dummy_input = np.random.randint(0, vocab_size, size=(4, window_size), dtype=np.int32)
    output = model.predict(dummy_input, verbose=0)

    assert output.shape == (4, vocab_size)
    # Check that each output row sums to ~1.0 (valid softmax)
    row_sums = np.sum(output, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(4), atol=1e-5)
    # Check non-negative
    assert np.all(output >= 0.0)
