"""DeepLog LSTM Network Training Pipeline with TensorFlow and Keras.

Builds, compiles, trains, and serializes a sequential recurrent language model
for next-token log event prediction on enterprise telemetry streams.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

# Suppress excessive TensorFlow runtime logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is on sys.path
PROJ_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    TRAIN_X_FILE,
    TRAIN_Y_FILE,
    VAL_X_FILE,
    VAL_Y_FILE,
    VOCAB_FILE,
    MODELS_DIR,
    MODEL_SAVE_PATH,
    METADATA_SAVE_PATH,
    FIGURES_DIR,
    WINDOW_SIZE,
    TOP_K,
    EMBEDDING_DIM,
    LSTM_UNITS,
    NUM_LSTM_LAYERS,
    DENSE_UNITS,
    DROPOUT_RATE,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    RANDOM_SEED,
    ensure_directories,
)
from src.features import EventVocabulary
from src.plots import plot_training_history


def build_deeplog_lstm_model(
    vocab_size: int,
    window_size: int = WINDOW_SIZE,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    num_lstm_layers: int = NUM_LSTM_LAYERS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    top_k: int = TOP_K,
    learning_rate: float = LEARNING_RATE,
) -> keras.Model:
    """Construct and compile the DeepLog recurrent neural architecture.

    Args:
        vocab_size: Total number of discrete log event classes in vocabulary.
        window_size: Length of input event history sequence (h).
        embedding_dim: Dimensionality of event embedding representation.
        lstm_units: Hidden units per LSTM layer.
        num_lstm_layers: Number of stacked LSTM layers (typically 2).
        dense_units: Hidden units for dense projection layer.
        dropout_rate: Dropout rate for regularization.
        top_k: K candidates for top-k accuracy metric.
        learning_rate: Adam optimizer initial learning rate.

    Returns:
        Compiled tf.keras.Model ready for training.
    """
    inputs = layers.Input(shape=(window_size,), dtype=tf.int32, name="log_sequence_input")

    # Dense Embedding layer maps discrete event IDs to continuous space
    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        name="event_embedding",
    )(inputs)

    # Stacked LSTM layers
    for i in range(num_lstm_layers):
        is_last = i == (num_lstm_layers - 1)
        x = layers.LSTM(
            units=lstm_units,
            return_sequences=not is_last,
            dropout=dropout_rate,
            name=f"lstm_layer_{i+1}",
        )(x)

    # Dense bottleneck representation
    x = layers.Dense(units=dense_units, activation="relu", name="dense_projection")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)

    # Softmax output layer predicts probability distribution across all classes
    outputs = layers.Dense(units=vocab_size, activation="softmax", name="next_event_softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="DeepLog_LSTM_Anomaly_Detector")

    # Safe top_k metric bounded by vocab size
    effective_top_k = min(top_k, max(1, vocab_size - 1))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "sparse_categorical_accuracy",
            keras.metrics.SparseTopKCategoricalAccuracy(
                k=effective_top_k,
                name=f"top_{effective_top_k}_accuracy",
            ),
        ],
    )

    return model


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    vocab_size: int,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    lstm_units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    top_k: int = TOP_K,
) -> Tuple[keras.Model, keras.callbacks.History]:
    """Execute model training with early stopping, checkpointing, and LR decay."""
    ensure_directories()
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = build_deeplog_lstm_model(
        vocab_size=vocab_size,
        window_size=x_train.shape[1],
        embedding_dim=EMBEDDING_DIM,
        lstm_units=lstm_units,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dense_units=DENSE_UNITS,
        dropout_rate=dropout_rate,
        top_k=top_k,
        learning_rate=learning_rate,
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_SAVE_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(
            filename=str(MODELS_DIR / "training_history.csv"),
            separator=",",
            append=False,
        ),
    ]

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


@click.command()
@click.option("--epochs", default=EPOCHS, type=int, help="Maximum training epochs.")
@click.option("--batch-size", default=BATCH_SIZE, type=int, help="Training batch size.")
@click.option("--lr", default=LEARNING_RATE, type=float, help="Initial learning rate.")
@click.option("--lstm-units", default=LSTM_UNITS, type=int, help="Hidden units in LSTM.")
@click.option("--top-k", default=TOP_K, type=int, help="Top-K candidate accuracy parameter.")
def main(epochs: int, batch_size: int, lr: float, lstm_units: int, top_k: int) -> None:
    """CLI Entrypoint for Model Training."""
    ensure_directories()
    click.echo("=" * 70)
    click.echo("Step 3: DeepLog LSTM Model Training")
    click.echo("=" * 70)

    if not TRAIN_X_FILE.exists() or not VOCAB_FILE.exists():
        raise FileNotFoundError(
            "Processed training files not found. Run src/features.py first."
        )

    vocab = EventVocabulary.load(VOCAB_FILE)
    x_train = np.load(TRAIN_X_FILE)
    y_train = np.load(TRAIN_Y_FILE)
    x_val = np.load(VAL_X_FILE)
    y_val = np.load(VAL_Y_FILE)

    click.echo(f"Loaded {len(x_train)} training sequences, {len(x_val)} validation sequences.")
    click.echo(f"Vocabulary size: {vocab.size}")

    model, history = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        vocab_size=vocab.size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lstm_units=lstm_units,
        dropout_rate=DROPOUT_RATE,
        top_k=top_k,
    )

    # Save model explicitly (also checkpointed)
    model.save(str(MODEL_SAVE_PATH))
    click.echo(f"Model saved to: {MODEL_SAVE_PATH}")

    # Generate training progression plots
    hist_dict = history.history
    curve_path = plot_training_history(hist_dict, FIGURES_DIR / "training_curves.png")
    click.echo(f"Training progression curves saved to: {curve_path}")

    # Save training metadata
    metadata: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat(),
        "vocab_size": vocab.size,
        "window_size": int(x_train.shape[1]),
        "epochs_trained": len(hist_dict["loss"]),
        "final_train_loss": float(hist_dict["loss"][-1]),
        "final_val_loss": float(hist_dict["val_loss"][-1]),
        "final_train_accuracy": float(hist_dict.get("sparse_categorical_accuracy", [0])[-1]),
        "final_val_accuracy": float(hist_dict.get("val_sparse_categorical_accuracy", [0])[-1]),
        "hyperparameters": {
            "embedding_dim": EMBEDDING_DIM,
            "lstm_units": lstm_units,
            "num_lstm_layers": NUM_LSTM_LAYERS,
            "dense_units": DENSE_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "learning_rate": lr,
            "batch_size": batch_size,
            "top_k": top_k,
        },
    }
    with open(METADATA_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    click.echo(f"Model metadata saved to: {METADATA_SAVE_PATH}")
    click.echo("DeepLog LSTM training completed successfully.\n")


if __name__ == "__main__":
    main()
