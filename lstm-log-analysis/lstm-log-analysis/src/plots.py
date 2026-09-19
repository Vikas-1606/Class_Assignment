"""Visualization suite for LSTM log anomaly detection training and evaluation.

Generates publication-quality charts for training dynamics, ROC/PR curves,
confusion matrices, top-k sensitivity, and anomaly score separation.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# Ensure project root is on sys.path
PROJ_DIR = Path(__file__).resolve().parent.parent
if str(PROJ_DIR) not in sys.path:
    sys.path.insert(0, str(PROJ_DIR))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from src.config import FIGURES_DIR, ensure_directories

# Set global aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
})


def plot_training_history(
    history_dict: Dict[str, List[float]],
    save_path: Optional[Path] = None,
) -> Path:
    """Plot training and validation loss and accuracy trajectories.

    Args:
        history_dict: Dictionary containing 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        save_path: Target path to save the image. Defaults to figures/training_curves.png.

    Returns:
        The Path where the figure was saved.
    """
    ensure_directories()
    if save_path is None:
        save_path = FIGURES_DIR / "training_curves.png"

    epochs = range(1, len(history_dict.get("loss", [])) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history_dict.get("loss", []), "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history_dict:
        ax1.plot(epochs, history_dict["val_loss"], "s--", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.set_title("Cross-Entropy Loss Progression")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Accuracy subplot
    acc_key = "sparse_categorical_accuracy" if "sparse_categorical_accuracy" in history_dict else "accuracy"
    val_acc_key = f"val_{acc_key}"
    if acc_key in history_dict:
        ax2.plot(epochs, history_dict[acc_key], "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    if val_acc_key in history_dict:
        ax2.plot(epochs, history_dict[val_acc_key], "s--", label="Val Accuracy", color="#d62728", linewidth=2)

    ax2.set_title("Next-Event Prediction Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_confusion_matrix_heatmap(
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
) -> Path:
    """Plot an annotated confusion matrix heatmap.

    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Anomaly).
        y_pred: Predicted binary labels.
        save_path: Destination path for the figure.
        class_names: Display labels for classes.

    Returns:
        The Path where the figure was saved.
    """
    ensure_directories()
    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrix.png"
    if class_names is None:
        class_names = ["Normal (0)", "Anomaly (1)"]

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]:,d}\n({cm_norm[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        square=True,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )
    ax.set_title("Session-Level Anomaly Detection Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_roc_and_pr_curves(
    y_true: List[int],
    anomaly_scores: List[float],
    save_path: Optional[Path] = None,
) -> Path:
    """Plot both ROC and Precision-Recall curves side by side.

    Args:
        y_true: Ground truth binary labels (0 = Normal, 1 = Anomaly).
        anomaly_scores: Continuous anomaly probability or likelihood scores.
        save_path: Destination figure path.

    Returns:
        The Path where the figure was saved.
    """
    ensure_directories()
    if save_path is None:
        save_path = FIGURES_DIR / "roc_pr_curves.png"

    fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ROC Subplot
    ax1.plot(fpr, tpr, color="#2b5c8f", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax1.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Precision-Recall Subplot
    ax2.plot(recall, precision, color="#d95f02", lw=2.5, label=f"PR curve (AP = {pr_auc:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_top_k_sensitivity(
    k_values: List[int],
    precisions: List[float],
    recalls: List[float],
    f1_scores: List[float],
    save_path: Optional[Path] = None,
) -> Path:
    """Plot metric sensitivity as top-k candidate parameter varies.

    Args:
        k_values: Evaluated top-k values (e.g. [1, 2, 3, 5, 9, 15]).
        precisions: Corresponding precision scores.
        recalls: Corresponding recall scores.
        f1_scores: Corresponding F1 scores.
        save_path: Destination figure path.

    Returns:
        The Path where the figure was saved.
    """
    ensure_directories()
    if save_path is None:
        save_path = FIGURES_DIR / "top_k_sensitivity.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, precisions, "o-", label="Precision", color="#1b9e77", lw=2.5)
    ax.plot(k_values, recalls, "s-", label="Recall", color="#d95f02", lw=2.5)
    ax.plot(k_values, f1_scores, "^-", label="F1-Score", color="#7570b3", lw=2.5)

    ax.set_title("Top-K Candidate Parameter Sensitivity Analysis")
    ax.set_xlabel("Top-K Candidates Allowed (k)")
    ax.set_ylabel("Score (0.0 to 1.0)")
    ax.set_xticks(k_values)
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_anomaly_score_distribution(
    normal_scores: List[float],
    anomaly_scores: List[float],
    threshold: float = 0.5,
    save_path: Optional[Path] = None,
) -> Path:
    """Plot separation between normal and anomalous session score distributions.

    Args:
        normal_scores: Anomaly scores for normal sessions.
        anomaly_scores: Anomaly scores for anomalous sessions.
        threshold: Decision boundary line to plot.
        save_path: Destination figure path.

    Returns:
        The Path where the figure was saved.
    """
    ensure_directories()
    if save_path is None:
        save_path = FIGURES_DIR / "anomaly_score_distribution.png"

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(
        normal_scores,
        bins=30,
        color="#2ca02c",
        label="Normal Sessions",
        kde=True,
        stat="density",
        alpha=0.5,
        ax=ax,
    )
    sns.histplot(
        anomaly_scores,
        bins=30,
        color="#d62728",
        label="Anomalous Sessions",
        kde=True,
        stat="density",
        alpha=0.5,
        ax=ax,
    )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Decision Boundary ({threshold:.2f})",
    )

    ax.set_title("Anomaly Score Probability Density Distribution")
    ax.set_xlabel("Session Anomaly Score")
    ax.set_ylabel("Density")
    ax.legend(loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path
