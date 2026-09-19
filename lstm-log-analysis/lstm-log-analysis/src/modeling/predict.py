"""Anomaly detection inference and evaluation engine for DeepLog LSTM.

Evaluates sequential log telemetry against top-k predictive candidate boundaries,
computes production diagnostic metrics (Precision, Recall, F1, FPR, ROC-AUC),
and generates diagnostic visualizations.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Suppress verbose TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Ensure project root is on sys.path
PROJ_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

import click
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

from src.config import (
    MODEL_SAVE_PATH,
    VOCAB_FILE,
    TEST_SESSIONS_FILE,
    EVALUATION_METRICS_FILE,
    FIGURES_DIR,
    WINDOW_SIZE,
    TOP_K,
    ensure_directories,
)
from src.features import EventVocabulary, create_sliding_windows
from src.plots import (
    plot_confusion_matrix_heatmap,
    plot_roc_and_pr_curves,
    plot_top_k_sensitivity,
    plot_anomaly_score_distribution,
)


def load_trained_model(model_path: Path = MODEL_SAVE_PATH) -> keras.Model:
    """Load serialized Keras model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train model first.")
    return keras.models.load_model(str(model_path), compile=False)


def predict_window_anomaly(
    model: keras.Model,
    x_window: np.ndarray,
    actual_next_event: int,
    top_k: int = TOP_K,
) -> Tuple[bool, float, List[int], float]:
    """Evaluate a single history window against top-k next event candidate predictions.

    Args:
        model: Trained DeepLog LSTM model.
        x_window: Input history array of shape (1, window_size) or (window_size,).
        actual_next_event: The ground truth observed next event token ID.
        top_k: Number of top candidate events permitted under normal grammar.

    Returns:
        is_anomaly: True if actual next event is NOT within top-k candidates.
        anomaly_score: Continuous probability-based penalty (1.0 - P(actual)).
        top_k_candidates: List of the top-k predicted candidate token IDs.
        actual_prob: Probability assigned to actual event.
    """
    if x_window.ndim == 1:
        x_window = np.expand_dims(x_window, axis=0)

    prob_dist = model.predict(x_window, verbose=0)[0]
    top_k_indices = np.argsort(prob_dist)[::-1][:top_k].tolist()
    actual_prob = float(prob_dist[actual_next_event]) if actual_next_event < len(prob_dist) else 0.0
    anomaly_score = float(1.0 - actual_prob)
    is_anomaly = bool(actual_next_event not in top_k_indices)

    return is_anomaly, anomaly_score, top_k_indices, actual_prob


def predict_single_sequence(
    model: keras.Model,
    vocab: EventVocabulary,
    event_sequence: List[str],
    window_size: int = WINDOW_SIZE,
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Perform real-time anomaly diagnosis on an ad-hoc sequence of event strings.

    Args:
        model: Loaded DeepLog model.
        vocab: EventVocabulary mapping.
        event_sequence: List of event string IDs, e.g. ['E2', 'E1', 'E3', 'E4', 'E20'].
        window_size: History length.
        top_k: Top candidates considered normal.

    Returns:
        Structured diagnosis with overall decision and step-by-step predictions.
    """
    token_ids = [vocab.token_to_id(e) for e in event_sequence]
    x_windows, y_targets = create_sliding_windows(token_ids, window_size=window_size)

    if len(x_windows) == 0:
        return {
            "is_anomaly": False,
            "session_anomaly_score": 0.0,
            "anomalous_steps": [],
            "total_steps": 0,
        }

    probs = model.predict(x_windows, batch_size=len(x_windows), verbose=0)
    anomalous_steps = []
    max_score = 0.0

    for i, (actual, prob_dist) in enumerate(zip(y_targets, probs)):
        top_k_ids = np.argsort(prob_dist)[::-1][:top_k].tolist()
        actual_prob = float(prob_dist[actual]) if actual < len(prob_dist) else 0.0
        score = float(1.0 - actual_prob)
        if score > max_score:
            max_score = score

        is_step_anom = actual not in top_k_ids
        if is_step_anom:
            anomalous_steps.append({
                "step": i + 1,
                "history": [vocab.id_to_token(tid) for tid in x_windows[i]],
                "observed_event": vocab.id_to_token(actual),
                "observed_probability": actual_prob,
                "top_candidates": [vocab.id_to_token(tid) for tid in top_k_ids],
                "penalty_score": score,
            })

    return {
        "is_anomaly": len(anomalous_steps) > 0,
        "session_anomaly_score": max_score,
        "anomalous_steps_count": len(anomalous_steps),
        "total_steps": len(x_windows),
        "anomalous_steps": anomalous_steps,
    }


def precompute_test_session_predictions(
    model: keras.Model,
    test_sessions: Dict[str, Dict[str, Any]],
    window_size: int = WINDOW_SIZE,
) -> List[Dict[str, Any]]:
    """Precompute model probabilities for all test sessions in a single batched pass.

    This enables instant evaluation and threshold sweeps across any number
    of top-k candidate values without re-invoking the neural network.
    """
    session_items = list(test_sessions.items())
    all_windows_list: List[np.ndarray] = []
    session_slices: List[Tuple[str, int, int, np.ndarray, int]] = []

    current_idx = 0
    for block_id, s_data in session_items:
        tokens = s_data["token_ids"]
        x_w, y_w = create_sliding_windows(tokens, window_size=window_size)
        n_windows = len(x_w)
        if n_windows > 0:
            all_windows_list.append(x_w)
            session_slices.append((block_id, current_idx, current_idx + n_windows, y_w, int(s_data["is_anomaly"])))
            current_idx += n_windows
        else:
            session_slices.append((block_id, -1, -1, np.array([]), int(s_data["is_anomaly"])))

    if all_windows_list:
        x_all = np.vstack(all_windows_list)
        prob_all = model.predict(x_all, batch_size=256, verbose=0)
    else:
        prob_all = np.empty((0, 0))

    precomputed_sessions: List[Dict[str, Any]] = []
    for block_id, start_idx, end_idx, targets, is_anom in session_slices:
        if start_idx != -1 and end_idx > start_idx:
            probs = prob_all[start_idx:end_idx]
        else:
            probs = np.empty((0, 0))

        precomputed_sessions.append({
            "block_id": block_id,
            "targets": targets,
            "probs": probs,
            "ground_truth": is_anom,
        })

    return precomputed_sessions


def evaluate_precomputed_sessions(
    precomputed: List[Dict[str, Any]],
    top_k: int = TOP_K,
) -> Dict[str, Any]:
    """Vectorized session-level anomaly evaluation using precomputed probabilities."""
    y_true: List[int] = []
    y_pred: List[int] = []
    anomaly_scores: List[float] = []
    normal_scores: List[float] = []
    anom_scores: List[float] = []

    for item in precomputed:
        ground_truth = item["ground_truth"]
        probs = item["probs"]
        targets = item["targets"]

        if len(probs) == 0:
            is_anomaly = False
            score = 0.0
        else:
            # For each window, check if target is in top-k
            # argsort returns indices in ascending order, so [:, -top_k:] are the top-k
            sorted_indices = np.argsort(probs, axis=1)[:, -top_k:]
            target_col = targets[:, np.newaxis]
            is_in_top_k = np.any(sorted_indices == target_col, axis=1)
            is_anomaly = bool(np.any(~is_in_top_k))

            # Continuous anomaly score = 1.0 - target_prob
            row_indices = np.arange(len(targets))
            target_probs = probs[row_indices, targets]
            scores = 1.0 - target_probs
            score = float(np.max(scores)) if len(scores) > 0 else 0.0

        pred_label = 1 if is_anomaly else 0
        y_true.append(ground_truth)
        y_pred.append(pred_label)
        anomaly_scores.append(score)

        if ground_truth == 0:
            normal_scores.append(score)
        else:
            anom_scores.append(score)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
        pr_auc = float(average_precision_score(y_true, anomaly_scores))
    except ValueError:
        roc_auc = 1.0
        pr_auc = 1.0

    return {
        "top_k": top_k,
        "total_test_sessions": len(y_true),
        "normal_sessions": int(tn + fp),
        "anomalous_sessions": int(tp + fn),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "anomaly_scores": anomaly_scores,
        "normal_scores": normal_scores,
        "anom_scores": anom_scores,
    }


def run_top_k_sensitivity_sweep_fast(
    precomputed: List[Dict[str, Any]],
    k_candidates: List[int] = [1, 2, 3, 5, 9, 15],
) -> Dict[str, List[float]]:
    """Instant sensitivity sweep over precomputed probability tensors."""
    precisions, recalls, f1s = [], [], []
    for k in k_candidates:
        res = evaluate_precomputed_sessions(precomputed, top_k=k)
        precisions.append(res["precision"])
        recalls.append(res["recall"])
        f1s.append(res["f1_score"])

    return {
        "k_candidates": k_candidates,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1s,
    }


@click.command()
@click.option("--top-k", default=TOP_K, type=int, help="Number of predicted candidates considered normal.")
@click.option("--model-path", default=None, type=click.Path(), help="Path to trained model .keras file.")
def main(top_k: int, model_path: Optional[str]) -> None:
    """CLI Entrypoint for Anomaly Detection Evaluation."""
    ensure_directories()
    click.echo("=" * 70)
    click.echo(f"Step 4: Anomaly Detection Inference & Evaluation (Top-K = {top_k})")
    click.echo("=" * 70)

    target_model_path = Path(model_path) if model_path else MODEL_SAVE_PATH
    if not target_model_path.exists():
        raise FileNotFoundError(f"Model file not found at {target_model_path}. Run src/modeling/train.py first.")

    if not TEST_SESSIONS_FILE.exists() or not VOCAB_FILE.exists():
        raise FileNotFoundError("Test sessions or vocab not found. Run src/features.py first.")

    with open(TEST_SESSIONS_FILE, "r", encoding="utf-8") as f:
        test_sessions = json.load(f)

    vocab = EventVocabulary.load(VOCAB_FILE)
    click.echo(f"Loaded {len(test_sessions)} test sessions and {vocab.size} vocab tokens.")

    click.echo(f"Loading trained neural model from: {target_model_path}...")
    model = load_trained_model(target_model_path)

    click.echo("Batched precomputation of predictive probability distributions...")
    precomputed = precompute_test_session_predictions(
        model=model,
        test_sessions=test_sessions,
        window_size=WINDOW_SIZE,
    )

    click.echo("Evaluating session anomaly classification across test set...")
    metrics = evaluate_precomputed_sessions(precomputed=precomputed, top_k=top_k)

    cm = metrics["confusion_matrix"]
    click.echo("\n--- Anomaly Detection Performance Results ---")
    click.echo(f"Total Test Sessions: {metrics['total_test_sessions']}")
    click.echo(f"Confusion Matrix:    TP={cm['true_positives']} | FP={cm['false_positives']} | TN={cm['true_negatives']} | FN={cm['false_negatives']}")
    click.echo(f"Accuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    click.echo(f"Precision:           {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    click.echo(f"Recall (TPR):        {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    click.echo(f"F1-Score:            {metrics['f1_score']:.4f}")
    click.echo(f"False Positive Rate: {metrics['false_positive_rate']:.4f} ({metrics['false_positive_rate'] * 100:.2f}%)")
    click.echo(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    click.echo(f"PR-AUC:              {metrics['pr_auc']:.4f}")

    # Generate diagnostic plots
    cm_path = plot_confusion_matrix_heatmap(
        y_true=metrics["y_true"],
        y_pred=metrics["y_pred"],
        save_path=FIGURES_DIR / "confusion_matrix.png",
    )
    click.echo(f"\nSaved Confusion Matrix Heatmap: {cm_path}")

    roc_path = plot_roc_and_pr_curves(
        y_true=metrics["y_true"],
        anomaly_scores=metrics["anomaly_scores"],
        save_path=FIGURES_DIR / "roc_pr_curves.png",
    )
    click.echo(f"Saved ROC & PR Curves:          {roc_path}")

    score_dist_path = plot_anomaly_score_distribution(
        normal_scores=metrics["normal_scores"],
        anomaly_scores=metrics["anom_scores"],
        threshold=0.5,
        save_path=FIGURES_DIR / "anomaly_score_distribution.png",
    )
    click.echo(f"Saved Score Distribution Plot:  {score_dist_path}")

    # Top-K sensitivity sweep
    click.echo("\nRunning Top-K sensitivity sweep across [1, 2, 3, 5, 9, 15]...")
    sweep_results = run_top_k_sensitivity_sweep_fast(
        precomputed=precomputed,
        k_candidates=[1, 2, 3, 5, 9, 15],
    )
    top_k_path = plot_top_k_sensitivity(
        k_values=sweep_results["k_candidates"],
        precisions=sweep_results["precisions"],
        recalls=sweep_results["recalls"],
        f1_scores=sweep_results["f1_scores"],
        save_path=FIGURES_DIR / "top_k_sensitivity.png",
    )
    click.echo(f"Saved Top-K Sensitivity Plot:   {top_k_path}")

    # Persist metrics summary
    metrics_to_save = {k: v for k, v in metrics.items() if not isinstance(v, list)}
    metrics_to_save["sensitivity_sweep"] = sweep_results
    with open(EVALUATION_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, indent=2)

    click.echo(f"Saved Evaluation Metrics to:    {EVALUATION_METRICS_FILE}")

    # Ad-hoc demonstration of single sequence anomaly diagnosis
    click.echo("\n--- Live Telemetry Trace Diagnosis Demonstration ---")
    normal_sample = ["E2", "E1", "E3", "E4", "E5", "E5", "E5", "E7", "E6"]
    anom_sample = ["E2", "E1", "E19", "E20"]

    diag_normal = predict_single_sequence(model, vocab, normal_sample, top_k=top_k)
    diag_anom = predict_single_sequence(model, vocab, anom_sample, top_k=top_k)

    click.echo(f"Sample Normal Trace {normal_sample} -> Anomaly Detected: {diag_normal['is_anomaly']} (Score: {diag_normal['session_anomaly_score']:.4f})")
    click.echo(f"Sample Anomaly Trace {anom_sample} -> Anomaly Detected: {diag_anom['is_anomaly']} (Score: {diag_anom['session_anomaly_score']:.4f})")
    if diag_anom["anomalous_steps"]:
        first_fault = diag_anom["anomalous_steps"][0]
        click.echo(f"  Fault Trigger: Observed '{first_fault['observed_event']}' with prob {first_fault['observed_probability']:.4f} (expected top candidates: {first_fault['top_candidates']})")

    click.echo("\nInference and evaluation completed successfully.\n")


if __name__ == "__main__":
    main()
