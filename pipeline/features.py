"""
pipeline/features.py — Feature engineering for UPI anomaly detection.

Transforms raw transaction data into model-ready features,
applies SMOTE for class balancing, and persists processed splits.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED_PATH,
    FEATURE_COLS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    RANDOM_STATE,
    TEST_SIZE,
    VAL_SIZE,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations.

    Adds:
        - Amount_log: log1p of Amount
        - amount_zscore: z-score of Amount within merchant_category
        - is_night: 1 if hour_of_day in [0-5] or [22-23]
        - high_velocity: 1 if txn_velocity_1h > 10

    Args:
        df: DataFrame with raw + UPI columns.

    Returns:
        DataFrame with engineered features appended.
    """
    logger.info("Feature engineering: processing %d rows", len(df))

    df = df.copy()

    # Log-transform Amount
    df["Amount_log"] = np.log1p(df["Amount"])

    # Per-category z-score of Amount
    df["amount_zscore"] = df.groupby("merchant_category")["Amount"].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )

    # Night transaction flag
    night_hours = set(range(0, 6)) | {22, 23}
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h in night_hours else 0)

    # High velocity flag
    df["high_velocity"] = (df["txn_velocity_1h"] > 10).astype(int)

    logger.info(
        "Feature engineering complete | is_night_rate=%.2f%% | high_velocity_rate=%.2f%%",
        df["is_night"].mean() * 100,
        df["high_velocity"].mean() * 100,
    )
    return df


def _get_model_feature_cols(df: pd.DataFrame) -> list:
    """Return available model feature columns."""
    desired = FEATURE_COLS + ["Amount_log", "amount_zscore", "is_night", "high_velocity"]
    return [c for c in desired if c in df.columns]


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets.

    Applies SMOTE only to the training split to handle class imbalance.

    Args:
        df: Fully engineered DataFrame.
        test_size: Fraction for test set.
        val_size: Fraction for validation set (taken from remaining after test).
        random_state: RNG seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    logger.info(
        "Splitting data | total=%d | test_size=%.0f%% | val_size=%.0f%%",
        len(df),
        test_size * 100,
        val_size * 100,
    )

    # First split off test set
    train_val, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["is_fraud"]
    )

    # Then split val from train_val
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_fraction,
        random_state=random_state,
        stratify=train_val["is_fraud"],
    )

    logger.info(
        "Pre-SMOTE class distribution | train fraud rate=%.4f%% | val=%.4f%% | test=%.4f%%",
        train_df["is_fraud"].mean() * 100,
        val_df["is_fraud"].mean() * 100,
        test_df["is_fraud"].mean() * 100,
    )

    # Apply SMOTE to training set only
    train_df = _apply_smote(train_df, random_state)

    logger.info(
        "Post-SMOTE class distribution | train fraud rate=%.2f%%",
        train_df["is_fraud"].mean() * 100,
    )

    return train_df, val_df, test_df


def _apply_smote(train_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    """Apply SMOTE oversampling to the training split.

    Args:
        train_df: Training DataFrame.
        random_state: RNG seed.

    Returns:
        Balanced training DataFrame.
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        logger.warning("imbalanced-learn not installed — skipping SMOTE")
        return train_df

    feature_cols = _get_model_feature_cols(train_df)
    X = train_df[feature_cols].fillna(0).values
    y = train_df["is_fraud"].values

    logger.info(
        "Applying SMOTE | before: normal=%d fraud=%d",
        (y == 0).sum(),
        (y == 1).sum(),
    )

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)

    logger.info(
        "SMOTE complete | after: normal=%d fraud=%d | total=%d",
        (y_res == 0).sum(),
        (y_res == 1).sum(),
        len(y_res),
    )

    resampled = pd.DataFrame(X_res, columns=feature_cols)
    resampled["is_fraud"] = y_res.astype(int)
    return resampled


def save_processed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = DATA_PROCESSED_PATH,
) -> None:
    """Persist processed splits to parquet files.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        output_dir: Directory to write parquet files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = str(Path(output_dir) / f"{name}.parquet")
        split.to_parquet(path, index=False)
        logger.info("Saved %s split | rows=%d | path=%s", name, len(split), path)


def load_processed_splits(
    output_dir: str = DATA_PROCESSED_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load persisted parquet splits.

    Args:
        output_dir: Directory containing parquet files.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    dfs = []
    for name in ["train", "val", "test"]:
        path = str(Path(output_dir) / f"{name}.parquet")
        df = pd.read_parquet(path)
        logger.info("Loaded %s split | rows=%d", name, len(df))
        dfs.append(df)
    return tuple(dfs)  # type: ignore[return-value]
