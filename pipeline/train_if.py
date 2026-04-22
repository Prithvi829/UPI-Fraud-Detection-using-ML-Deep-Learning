"""
pipeline/train_if.py — Isolation Forest training for UPI anomaly detection.

Performs hyperparameter search via GridSearchCV, retrains on combined
train+val data, and persists the model, scaler, and evaluation metrics.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    ANOMALY_PERCENTILE_THRESHOLD,
    FEATURE_COLS,
    IF_CONTAMINATION_DEFAULT,
    IF_CONTAMINATION_GRID,
    IF_MAX_FEATURES_GRID,
    IF_N_ESTIMATORS_GRID,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    MODEL_IF_PATH,
    RANDOM_STATE,
)
from pipeline.evaluate import evaluate_model

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_feature_cols(df: pd.DataFrame) -> list:
    """Return available model feature columns from DataFrame."""
    desired = FEATURE_COLS + ["Amount_log", "amount_zscore", "is_night", "high_velocity"]
    return [c for c in desired if c in df.columns]


def train_isolation_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = MODEL_IF_PATH,
) -> Dict[str, Any]:
    """Train an Isolation Forest model with grid search over hyperparameters.

    Args:
        train_df: Training split (post-SMOTE).
        val_df: Validation split.
        test_df: Test split.
        output_dir: Directory to save model artefacts.

    Returns:
        Metrics dictionary.
    """
    t0 = time.time()
    logger.info("UPI IF Training | Starting Isolation Forest hyperparameter search")

    feature_cols = _get_feature_cols(train_df)
    logger.info("Using %d feature columns", len(feature_cols))

    # Fit scaler on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].fillna(0))
    y_train = train_df["is_fraud"].values

    X_val = scaler.transform(val_df[feature_cols].fillna(0))
    y_val = val_df["is_fraud"].values

    X_test = scaler.transform(test_df[feature_cols].fillna(0))
    y_test = test_df["is_fraud"].values

    # Grid search — train only on normal transactions for IF
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    param_grid = {
        "n_estimators": IF_N_ESTIMATORS_GRID,
        "contamination": IF_CONTAMINATION_GRID,
        "max_features": IF_MAX_FEATURES_GRID,
    }

    best_f1 = -1.0
    best_params: Dict[str, Any] = {}
    total_configs = len(list(ParameterGrid(param_grid)))

    logger.info("Grid searching %d configurations", total_configs)

    for i, params in enumerate(ParameterGrid(param_grid)):
        model = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **params)
        model.fit(X_train_normal)

        # Scores: more negative = more anomalous → invert for scoring
        raw_scores = -model.score_samples(X_val)
        thresh = np.percentile(raw_scores, ANOMALY_PERCENTILE_THRESHOLD)
        metrics = evaluate_model(y_val, raw_scores, thresh, f"IF-grid-{i}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_params = params
            logger.info(
                "New best | f1=%.4f | params=%s", best_f1, params
            )

    logger.info("Best IF params: %s | val_f1=%.4f", best_params, best_f1)

    # Retrain on train + val combined
    val_feature_cols = _get_feature_cols(val_df)
    X_train_val = scaler.transform(
        pd.concat([train_df[feature_cols], val_df[val_feature_cols]], ignore_index=True).fillna(0)
    )
    y_train_val = np.concatenate([y_train, y_val])

    # Retrain on normals only
    normal_mask_full = y_train_val == 0
    X_train_val_normal = X_train_val[normal_mask_full]

    final_model = IsolationForest(random_state=RANDOM_STATE, n_jobs=-1, **best_params)
    final_model.fit(X_train_val_normal)

    # Compute threshold on val normal transactions
    val_normal_mask = y_val == 0
    val_scores_normal = -final_model.score_samples(X_val[val_normal_mask])
    threshold = float(np.percentile(val_scores_normal, ANOMALY_PERCENTILE_THRESHOLD))

    # Evaluate on test set
    test_scores = -final_model.score_samples(X_test)
    test_metrics = evaluate_model(y_test, test_scores, threshold, "IsolationForest-final")

    # Save artefacts
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, str(Path(output_dir) / "model.pkl"))
    joblib.dump(scaler, str(Path(output_dir) / "scaler.pkl"))

    metrics_payload = {
        **test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": int(X_train_normal.shape[0]),
        "contamination": float(best_params.get("contamination", IF_CONTAMINATION_DEFAULT)),
        "best_params": best_params,
        "feature_cols": feature_cols,
        "training_seconds": round(time.time() - t0, 2),
    }
    with open(str(Path(output_dir) / "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(
        "UPI IF Training complete | test_f1=%.4f | saved to %s | elapsed=%.1fs",
        test_metrics["f1"],
        output_dir,
        time.time() - t0,
    )
    return metrics_payload


if __name__ == "__main__":
    from pipeline.features import load_processed_splits
    train_df, val_df, test_df = load_processed_splits()
    train_isolation_forest(train_df, val_df, test_df)
