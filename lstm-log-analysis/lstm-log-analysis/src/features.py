"""Feature engineering and sliding-window sequence generation for DeepLog LSTM.

Transforms structured event log sequences into sliding window training pairs
and session-isolated evaluation sets, enforcing zero data leakage.
"""

import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure project root is on sys.path
PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

import click
import numpy as np

from src.config import (
    SESSION_SEQUENCES_FILE,
    PROCESSED_DATA_DIR,
    VOCAB_FILE,
    TRAIN_X_FILE,
    TRAIN_Y_FILE,
    VAL_X_FILE,
    VAL_Y_FILE,
    TEST_X_FILE,
    TEST_Y_FILE,
    TEST_SESSIONS_FILE,
    WINDOW_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
    PAD_TOKEN,
    UNK_TOKEN,
    PAD_INDEX,
    UNK_INDEX,
    ensure_directories,
)


class EventVocabulary:
    """Vocabulary mapping event IDs to integer tokens with special tokens."""

    def __init__(self, pad_token: str = PAD_TOKEN, unk_token: str = UNK_TOKEN) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.token_to_idx: Dict[str, int] = {pad_token: PAD_INDEX, unk_token: UNK_INDEX}
        self.idx_to_token: Dict[int, str] = {PAD_INDEX: pad_token, UNK_INDEX: unk_token}

    def build_vocab(self, event_sequences: List[List[str]]) -> None:
        """Construct vocabulary from unique event tokens across all sequences."""
        unique_tokens = sorted(list({token for seq in event_sequences for token in seq}))
        for token in unique_tokens:
            if token not in self.token_to_idx:
                new_idx = len(self.token_to_idx)
                self.token_to_idx[token] = new_idx
                self.idx_to_token[new_idx] = token

    def token_to_id(self, token: str) -> int:
        """Map token string to integer ID, falling back to UNK_INDEX."""
        return self.token_to_idx.get(token, UNK_INDEX)

    def id_to_token(self, idx: int) -> str:
        """Map integer ID to token string, falling back to UNK_TOKEN."""
        return self.idx_to_token.get(idx, self.unk_token)

    @property
    def size(self) -> int:
        """Return the total number of distinct tokens."""
        return len(self.token_to_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize vocabulary to dictionary."""
        return {
            "token_to_idx": self.token_to_idx,
            "idx_to_token": {str(k): v for k, v in self.idx_to_token.items()},
            "vocab_size": self.size,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "pad_index": PAD_INDEX,
            "unk_index": UNK_INDEX,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventVocabulary":
        """Instantiate vocabulary from serialized dictionary."""
        vocab = cls(pad_token=data["pad_token"], unk_token=data["unk_token"])
        vocab.token_to_idx = {k: int(v) for k, v in data["token_to_idx"].items()}
        vocab.idx_to_token = {int(k): v for k, v in data["idx_to_token"].items()}
        return vocab

    def save(self, filepath: Path) -> None:
        """Persist vocabulary to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "EventVocabulary":
        """Load vocabulary from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_sliding_windows(
    token_sequence: List[int],
    window_size: int = WINDOW_SIZE,
    pad_index: int = PAD_INDEX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sliding window input/output pairs (x, y) from an integer sequence.

    Args:
        token_sequence: List of integer tokens representing the log event stream.
        window_size: History length (h).
        pad_index: Index used to pad short sequences.

    Returns:
        X: array of shape (N, window_size)
        y: array of shape (N,)
    """
    if len(token_sequence) <= window_size:
        # Pad sequence so that it can form at least one window
        pad_needed = window_size - len(token_sequence) + 1
        seq = [pad_index] * pad_needed + token_sequence
    else:
        seq = token_sequence

    x_list: List[List[int]] = []
    y_list: List[int] = []

    for i in range(len(seq) - window_size):
        x_window = seq[i : i + window_size]
        y_target = seq[i + window_size]
        x_list.append(x_window)
        y_list.append(y_target)

    return np.array(x_list, dtype=np.int32), np.array(y_list, dtype=np.int32)


def process_and_split_sessions(
    sessions: Dict[str, Dict[str, Any]],
    window_size: int = WINDOW_SIZE,
    train_split: float = TRAIN_SPLIT,
    val_split: float = VAL_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    EventVocabulary,
    Dict[str, Any],
]:
    """Execute session-isolated splitting and window creation.

    In DeepLog, the model is trained exclusively on normal execution traces
    to model normal system behavior. Validation tracks generalization on normal
    sequences. The test set evaluates anomaly detection on held-out normal
    and anomalous sessions.
    """
    random.seed(seed)
    np.random.seed(seed)

    normal_sessions: List[Tuple[str, List[str]]] = []
    anomaly_sessions: List[Tuple[str, List[str]]] = []

    for block_id, s_data in sessions.items():
        if s_data["label"] == "Normal":
            normal_sessions.append((block_id, s_data["events"]))
        else:
            anomaly_sessions.append((block_id, s_data["events"]))

    # Build vocabulary over all sessions
    all_event_sequences = [s[1] for s in normal_sessions + anomaly_sessions]
    vocab = EventVocabulary()
    vocab.build_vocab(all_event_sequences)

    # Shuffle normal sessions
    random.shuffle(normal_sessions)
    n_normal = len(normal_sessions)
    n_train = int(n_normal * train_split)
    n_val = int(n_normal * val_split)

    train_normal = normal_sessions[:n_train]
    val_normal = normal_sessions[n_train : n_train + n_val]
    test_normal = normal_sessions[n_train + n_val :]

    # Build sliding windows for Train
    x_train_list, y_train_list = [], []
    for _, events in train_normal:
        token_ids = [vocab.token_to_id(e) for e in events]
        x_w, y_w = create_sliding_windows(token_ids, window_size=window_size)
        if len(x_w) > 0:
            x_train_list.append(x_w)
            y_train_list.append(y_w)

    # Build sliding windows for Val
    x_val_list, y_val_list = [], []
    for _, events in val_normal:
        token_ids = [vocab.token_to_id(e) for e in events]
        x_w, y_w = create_sliding_windows(token_ids, window_size=window_size)
        if len(x_w) > 0:
            x_val_list.append(x_w)
            y_val_list.append(y_w)

    # Build sliding windows for Test (evaluation)
    x_test_list, y_test_list = [], []
    for _, events in test_normal:
        token_ids = [vocab.token_to_id(e) for e in events]
        x_w, y_w = create_sliding_windows(token_ids, window_size=window_size)
        if len(x_w) > 0:
            x_test_list.append(x_w)
            y_test_list.append(y_w)

    x_train = np.vstack(x_train_list) if x_train_list else np.empty((0, window_size), dtype=np.int32)
    y_train = np.concatenate(y_train_list) if y_train_list else np.empty((0,), dtype=np.int32)
    x_val = np.vstack(x_val_list) if x_val_list else np.empty((0, window_size), dtype=np.int32)
    y_val = np.concatenate(y_val_list) if y_val_list else np.empty((0,), dtype=np.int32)
    x_test = np.vstack(x_test_list) if x_test_list else np.empty((0, window_size), dtype=np.int32)
    y_test = np.concatenate(y_test_list) if y_test_list else np.empty((0,), dtype=np.int32)

    # Package test sessions for end-to-end evaluation in predict.py
    test_sessions_dict: Dict[str, Any] = {}
    for block_id, events in test_normal:
        test_sessions_dict[block_id] = {
            "events": events,
            "token_ids": [vocab.token_to_id(e) for e in events],
            "label": "Normal",
            "is_anomaly": 0,
        }
    for block_id, events in anomaly_sessions:
        test_sessions_dict[block_id] = {
            "events": events,
            "token_ids": [vocab.token_to_id(e) for e in events],
            "label": "Anomaly",
            "is_anomaly": 1,
        }

    return x_train, y_train, x_val, y_val, x_test, y_test, vocab, test_sessions_dict


@click.command()
@click.option(
    "--window-size",
    default=WINDOW_SIZE,
    type=int,
    help="Sliding window size (history length h).",
)
@click.option(
    "--train-split",
    default=TRAIN_SPLIT,
    type=float,
    help="Proportion of normal sessions for training.",
)
def main(window_size: int, train_split: float) -> None:
    """CLI Entrypoint for Feature Engineering."""
    ensure_directories()
    click.echo("=" * 70)
    click.echo("Step 2: Feature Engineering & Sliding Window Sequence Construction")
    click.echo("=" * 70)

    if not SESSION_SEQUENCES_FILE.exists():
        raise FileNotFoundError(
            f"Interim session file not found at {SESSION_SEQUENCES_FILE}. Run src/dataset.py first."
        )

    with open(SESSION_SEQUENCES_FILE, "r", encoding="utf-8") as f:
        sessions = json.load(f)

    click.echo(f"Loaded {len(sessions)} sessions from {SESSION_SEQUENCES_FILE}")
    click.echo(f"Window Size: {window_size}, Train Split: {train_split:.2f}")

    x_train, y_train, x_val, y_val, x_test, y_test, vocab, test_sessions = (
        process_and_split_sessions(
            sessions=sessions,
            window_size=window_size,
            train_split=train_split,
        )
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Persist vocabulary and test sessions
    vocab.save(VOCAB_FILE)
    with open(TEST_SESSIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(test_sessions, f, indent=2)

    # Persist numpy tensors
    np.save(TRAIN_X_FILE, x_train)
    np.save(TRAIN_Y_FILE, y_train)
    np.save(VAL_X_FILE, x_val)
    np.save(VAL_Y_FILE, y_val)
    np.save(TEST_X_FILE, x_test)
    np.save(TEST_Y_FILE, y_test)

    click.echo(f"Vocabulary Size: {vocab.size} distinct tokens")
    click.echo(f"Training windows:   X={x_train.shape}, y={y_train.shape}")
    click.echo(f"Validation windows: X={x_val.shape}, y={y_val.shape}")
    click.echo(f"Test windows:       X={x_test.shape}, y={y_test.shape}")
    click.echo(f"Test sessions:      {len(test_sessions)} (Normal={sum(1 for s in test_sessions.values() if s['is_anomaly'] == 0)}, Anomaly={sum(1 for s in test_sessions.values() if s['is_anomaly'] == 1)})")
    click.echo(f"Artifacts saved in: {PROCESSED_DATA_DIR}")
    click.echo("Feature engineering completed successfully.\n")


if __name__ == "__main__":
    main()
