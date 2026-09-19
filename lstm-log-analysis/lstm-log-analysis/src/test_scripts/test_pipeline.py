"""End-to-End Integration Test for the complete DeepLog LSTM pipeline.

Executes a lightweight end-to-end run:
1. Synthesizes a mini benchmark dataset
2. Parses raw telemetry logs
3. Extracts sliding window tensors and vocabulary
4. Trains an LSTM model for 1 epoch
5. Runs anomaly detection inference and validates metrics
"""

import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import pandas as pd
import numpy as np

from src.dataset import (
    generate_synthetic_benchmark_dataset,
    parse_raw_logs,
)
from src.features import (
    process_and_split_sessions,
)
from src.modeling.train import (
    build_deeplog_lstm_model,
)
from src.modeling.predict import (
    precompute_test_session_predictions,
    evaluate_precomputed_sessions,
    predict_single_sequence,
)


def test_full_pipeline_end_to_end():
    """Verify that all pipeline components interlock seamlessly without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_log_file = tmp_path / "test_raw.log"
        label_file = tmp_path / "test_labels.csv"

        # 1. Dataset Generation
        raw_lines, labels = generate_synthetic_benchmark_dataset(
            num_sessions=60,
            anomaly_ratio=0.20,
            seed=42,
        )
        with open(raw_log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(raw_lines) + "\n")
        pd.DataFrame(labels).to_csv(label_file, index=False)

        assert raw_log_file.exists()
        assert label_file.exists()

        # 2. Parsing
        df_events, sessions = parse_raw_logs(raw_log_file, label_file)
        assert len(df_events) > 0
        assert len(sessions) == 60

        # 3. Feature Engineering & Windowing
        window_size = 5
        x_train, y_train, x_val, y_val, x_test, y_test, vocab, test_sessions = (
            process_and_split_sessions(
                sessions=sessions,
                window_size=window_size,
                train_split=0.8,
                val_split=0.1,
                seed=42,
            )
        )
        assert len(x_train) > 0
        assert x_train.shape[1] == window_size
        assert vocab.size > 2

        # 4. Model Training (1 epoch fast run)
        model = build_deeplog_lstm_model(
            vocab_size=vocab.size,
            window_size=window_size,
            embedding_dim=16,
            lstm_units=32,
            num_lstm_layers=1,
            dense_units=16,
            dropout_rate=0.0,
            top_k=3,
            learning_rate=0.01,
        )
        history = model.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_val, y_val),
            epochs=1,
            batch_size=32,
            verbose=0,
        )
        assert "loss" in history.history

        # 5. Prediction & Evaluation
        precomputed = precompute_test_session_predictions(
            model=model,
            test_sessions=test_sessions,
            window_size=window_size,
        )
        metrics = evaluate_precomputed_sessions(precomputed=precomputed, top_k=3)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0

        # 6. Single sequence diagnosis
        sample_trace = ["E2", "E1", "E3", "E4", "E5"]
        diagnosis = predict_single_sequence(model, vocab, sample_trace, window_size=window_size, top_k=3)
        assert "is_anomaly" in diagnosis
        assert "session_anomaly_score" in diagnosis


if __name__ == "__main__":
    print("Running end-to-end integration test...")
    test_full_pipeline_end_to_end()
    print("End-to-end integration pipeline test PASSED successfully!")
