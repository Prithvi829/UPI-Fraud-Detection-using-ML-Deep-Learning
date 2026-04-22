"""
pipeline/evaluate.py — Shared model evaluation utilities.

Provides standardised metrics computation and comparison across
Isolation Forest and Autoencoder models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LOG_DATE_FORMAT, LOG_FORMAT

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_model(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    model_name: str,
) -> Dict[str, Any]:
    """Compute classification metrics at a given anomaly score threshold.

    Args:
        y_true: Ground-truth binary labels (0=normal, 1=fraud).
        y_scores: Continuous anomaly scores (higher = more anomalous).
        threshold: Decision boundary for binary classification.
        model_name: Label used in log messages.

    Returns:
        Dict with precision, recall, f1, roc_auc, pr_auc, threshold.
    """
    y_pred = (y_scores >= threshold).astype(int)

    metrics: Dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_scores)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "threshold": float(threshold),
        "n_test_samples": int(len(y_true)),
        "fraud_detected": int(y_pred.sum()),
        "total_fraud": int(y_true.sum()),
    }

    logger.info(
        "%s evaluation | precision=%.4f | recall=%.4f | f1=%.4f | "
        "roc_auc=%.4f | pr_auc=%.4f",
        model_name,
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["roc_auc"],
        metrics["pr_auc"],
    )
    return metrics


def generate_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, int]:
    """Compute confusion matrix components.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dict with tn, fp, fn, tp counts.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_pr_curve(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Dict[str, List[float]]:
    """Compute precision-recall curve data for plotting.

    Args:
        y_true: Ground-truth binary labels.
        y_scores: Continuous anomaly scores.

    Returns:
        Dict with lists 'precision', 'recall', 'thresholds'.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compare_models(
    if_metrics: Dict[str, Any], ae_metrics: Dict[str, Any]
) -> None:
    """Print a formatted comparison table of IF vs AE metrics.

    Args:
        if_metrics: Metrics dict for Isolation Forest.
        ae_metrics: Metrics dict for Autoencoder.
    """
    header = f"{'Metric':<20} {'Isolation Forest':>18} {'Autoencoder':>14}"
    separator = "─" * len(header)
    rows = []
    for key in ["precision", "recall", "f1", "roc_auc", "pr_auc"]:
        iv = if_metrics.get(key, 0.0)
        av = ae_metrics.get(key, 0.0)
        winner = "◄ IF" if iv > av else "◄ AE"
        rows.append(f"{key:<20} {iv:>18.4f} {av:>14.4f}  {winner}")

    print("\n" + separator)
    print("  MODEL COMPARISON — UPI Anomaly Detection")
    print(separator)
    print(header)
    print(separator)
    for row in rows:
        print(row)
    print(separator + "\n")
