"""
pipeline/ingest.py — Data ingestion and synthetic UPI column generation.

Loads the Credit Card Fraud Detection dataset (or generates a synthetic
equivalent) and enriches it with UPI-style transaction metadata.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_RAW_PATH,
    LOG_FORMAT,
    LOG_DATE_FORMAT,
    MERCHANT_CATEGORIES,
    RANDOM_STATE,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REQUIRED_COLUMNS: List[str] = (
    [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "Class"]
)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load raw CSV data from *path*.

    If the file does not exist, falls back to generating a synthetic dataset
    that mirrors the ULB Credit Card Fraud Detection statistics.

    Args:
        path: Filesystem path to creditcard.csv.

    Returns:
        DataFrame with original schema (Time, V1–V28, Amount, Class).
    """
    if os.path.exists(path):
        logger.info("UPI Ingest | Loading real dataset from %s", path)
        df = pd.read_csv(path)
        logger.info(
            "UPI Ingest | Loaded %d rows, columns: %s", len(df), list(df.columns)
        )
    else:
        logger.warning(
            "UPI Ingest | creditcard.csv not found at %s — generating synthetic dataset",
            path,
        )
        df = _generate_synthetic_dataset()

    # Rename Class → is_fraud
    df = df.rename(columns={"Class": "is_fraud"})
    _log_dataset_stats(df)
    return df


def _generate_synthetic_dataset() -> pd.DataFrame:
    """Generate a synthetic dataset mirroring ULB Credit Card Fraud stats.

    Returns:
        DataFrame with columns: Time, V1–V28, Amount, Class.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    n_total = 284_807
    n_fraud = 492
    n_normal = n_total - n_fraud

    logger.info(
        "Generating synthetic dataset: %d normal + %d fraud = %d total",
        n_normal,
        n_fraud,
        n_total,
    )

    # Time column — seconds elapsed (0 to 172792)
    time_normal = np.sort(rng.uniform(0, 172_792, n_normal))
    time_fraud = rng.uniform(0, 172_792, n_fraud)
    time_col = np.concatenate([time_normal, time_fraud])

    # V1–V28 — PCA features, normal ~N(0,1), fraud slightly shifted
    v_normal = rng.standard_normal((n_normal, 28))
    v_fraud = rng.standard_normal((n_fraud, 28)) * 1.5 + rng.uniform(-2, 2, 28)
    v_all = np.vstack([v_normal, v_fraud])

    # Amount — lognormal; fraud amounts skewed lower
    amount_normal = rng.lognormal(mean=3.0, sigma=1.5, size=n_normal)
    amount_fraud = rng.lognormal(mean=2.0, sigma=1.2, size=n_fraud)
    amount_col = np.concatenate([amount_normal, amount_fraud])
    amount_col = np.clip(amount_col, 0.01, 25_000)

    # Class labels
    class_col = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])

    df = pd.DataFrame(v_all, columns=[f"V{i}" for i in range(1, 29)])
    df["Time"] = time_col
    df["Amount"] = amount_col
    df["Class"] = class_col.astype(int)

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    logger.info("Synthetic dataset generated: %d rows", len(df))
    return df


def add_upi_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich DataFrame with synthetic UPI-style metadata columns.

    Adds:
        - upi_id: random UPI handle string
        - merchant_category: one of grocery/fuel/ecommerce/utilities/P2P
        - hour_of_day: 0–23 derived from Time column
        - day_of_week: 0–6
        - txn_velocity_1h: simulated transaction count in last hour

    Args:
        df: DataFrame with at least Time and is_fraud columns.

    Returns:
        Enriched DataFrame.
    """
    rng = np.random.default_rng(RANDOM_STATE + 1)
    n = len(df)

    logger.info("UPI Ingest | Adding UPI metadata columns to %d rows", n)

    # UPI IDs — pool of 5 000 handles
    bank_suffixes = ["okicici", "oksbi", "okaxis", "okhdfcbank", "ybl", "paytm"]
    upi_pool = [
        f"user{rng.integers(1000, 99999)}@{rng.choice(bank_suffixes)}"
        for _ in range(5_000)
    ]
    df["upi_id"] = [upi_pool[i % 5_000] for i in rng.integers(0, 5_000, n)]

    # Merchant category
    df["merchant_category"] = rng.choice(MERCHANT_CATEGORIES, size=n)

    # Hour derived from Time (seconds elapsed, mod 86400 → seconds in day)
    df["hour_of_day"] = (df["Time"] % 86_400 // 3_600).astype(int)

    # Day of week
    df["day_of_week"] = (df["Time"] // 86_400 % 7).astype(int)

    # Transaction velocity — fraud transactions biased toward higher velocity
    velocity_normal = rng.integers(1, 8, size=n)
    fraud_mask = df["is_fraud"] == 1
    velocity_normal[fraud_mask] = rng.integers(8, 16, size=fraud_mask.sum())
    df["txn_velocity_1h"] = velocity_normal

    logger.info(
        "UPI Ingest | UPI columns added. Sample upi_id: %s, categories: %s",
        df["upi_id"].iloc[0],
        df["merchant_category"].value_counts().to_dict(),
    )
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that all required columns are present.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If any required column is missing.
    """
    required = (
        [f"V{i}" for i in range(1, 29)]
        + ["Time", "Amount", "is_fraud", "upi_id", "merchant_category",
           "hour_of_day", "day_of_week", "txn_velocity_1h"]
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed — missing columns: {missing}")
    logger.info("Schema validation passed — all %d required columns present", len(required))


def _log_dataset_stats(df: pd.DataFrame) -> None:
    """Log dataset statistics."""
    total = len(df)
    fraud_col = "is_fraud" if "is_fraud" in df.columns else "Class"
    fraud_count = int(df[fraud_col].sum())
    fraud_rate = fraud_count / total * 100
    logger.info(
        "Dataset stats | total_rows=%d | fraud_count=%d | fraud_rate=%.4f%%",
        total,
        fraud_count,
        fraud_rate,
    )


if __name__ == "__main__":
    df = load_raw_data(DATA_RAW_PATH)
    df = add_upi_columns(df)
    validate_schema(df)
    print(df.head())
