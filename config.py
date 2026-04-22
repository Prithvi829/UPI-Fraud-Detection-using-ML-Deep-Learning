"""
config.py — Central configuration for UPI Anomaly Detector.

All constants and hyperparameters live here.
No magic numbers anywhere else in the codebase.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

DATA_RAW_PATH = str(BASE_DIR / "data" / "raw" / "creditcard.csv")
DATA_PROCESSED_PATH = str(BASE_DIR / "data" / "processed")
MODEL_IF_PATH = str(BASE_DIR / "models" / "isolation_forest")
MODEL_AE_PATH = str(BASE_DIR / "models" / "autoencoder")

# ── Feature columns ──────────────────────────────────────────────────────────
FEATURE_COLS = (
    [f"V{i}" for i in range(1, 29)]
    + ["Amount", "hour_of_day", "day_of_week", "txn_velocity_1h"]
)

ENGINEERED_COLS = FEATURE_COLS + ["Amount_log", "amount_zscore", "is_night", "high_velocity"]

MERCHANT_CATEGORIES = ["grocery", "fuel", "ecommerce", "utilities", "P2P"]

# ── Data split ────────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42

# ── Isolation Forest ──────────────────────────────────────────────────────────
IF_CONTAMINATION_DEFAULT = 0.017
IF_N_ESTIMATORS_GRID = [100, 200, 300]
IF_CONTAMINATION_GRID = [0.001, 0.005, 0.01, 0.017]
IF_MAX_FEATURES_GRID = [0.8, 1.0]

# ── Autoencoder ───────────────────────────────────────────────────────────────
AE_EPOCHS = 50
AE_BATCH_SIZE = 256
AE_LR = 1e-3
AE_PATIENCE = 5
AE_BOTTLENECK = 8
AE_HIDDEN_DIMS = [32, 16]  # encoder dims before bottleneck

# ── Anomaly thresholds ────────────────────────────────────────────────────────
ANOMALY_PERCENTILE_THRESHOLD = 95

# ── Drift watchdog ────────────────────────────────────────────────────────────
DRIFT_CHECK_INTERVAL = 300        # seconds between checks
DRIFT_PRECISION_THRESHOLD = 0.70  # retrain if precision drops below this
DRIFT_WINDOW_SIZE = 200           # sliding window size

# ── API / transaction log ─────────────────────────────────────────────────────
TRANSACTION_LOG_MAXLEN = 10000
DRIFT_ALERT_MAXLEN = 100

# ── Risk thresholds ───────────────────────────────────────────────────────────
RISK_VELOCITY_CRITICAL_THRESHOLD = 10

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_MS = 5000

# ── Flask ─────────────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001

# ── Logging format ────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
