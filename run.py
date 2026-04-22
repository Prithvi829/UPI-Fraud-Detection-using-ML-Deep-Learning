"""
run.py — UPI Anomaly Detector entrypoint.

Orchestrates the full startup sequence:
  1. Data ingestion + feature engineering (if processed splits absent)
  2. Model training (Isolation Forest + Autoencoder, if models absent)
  3. Model comparison table
  4. DriftWatchdog daemon
  5. Flask server
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PROCESSED_PATH,
    DATA_RAW_PATH,
    FLASK_HOST,
    FLASK_PORT,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    MODEL_AE_PATH,
    MODEL_IF_PATH,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT, level=logging.INFO)
logger = logging.getLogger("run")


# ── Step helpers ──────────────────────────────────────────────────────────────

def _processed_splits_exist() -> bool:
    return all(
        (Path(DATA_PROCESSED_PATH) / f"{s}.parquet").exists()
        for s in ["train", "val", "test"]
    )


def _models_exist() -> bool:
    return (
        (Path(MODEL_IF_PATH) / "model.pkl").exists()
        and (Path(MODEL_AE_PATH) / "model.pt").exists()
    )


def run_data_pipeline() -> None:
    print("━" * 60)
    print("  [1/5] DATA PIPELINE")
    print("━" * 60)
    from pipeline.ingest import load_raw_data, add_upi_columns, validate_schema
    from pipeline.features import engineer_features, split_data, save_processed_splits

    df = load_raw_data(DATA_RAW_PATH)
    df = add_upi_columns(df)
    validate_schema(df)
    df = engineer_features(df)
    train_df, val_df, test_df = split_data(df)
    save_processed_splits(train_df, val_df, test_df)
    print(f"  ✓ Processed splits saved to {DATA_PROCESSED_PATH}\n")


def run_training() -> None:
    print("━" * 60)
    print("  [2/5] MODEL TRAINING")
    print("━" * 60)
    from pipeline.features import load_processed_splits
    train_df, val_df, test_df = load_processed_splits()

    # Train Isolation Forest
    print("\n  Training Isolation Forest…")
    t0 = time.time()
    from pipeline.train_if import train_isolation_forest
    if_metrics = train_isolation_forest(train_df, val_df, test_df)
    print(f"  ✓ IF trained in {time.time()-t0:.1f}s | F1={if_metrics['f1']:.4f}\n")

    # Train Autoencoder
    print("  Training Autoencoder…")
    t0 = time.time()
    from pipeline.train_ae import train_autoencoder
    ae_metrics = train_autoencoder(train_df, val_df, test_df)
    print(f"  ✓ AE trained in {time.time()-t0:.1f}s | F1={ae_metrics['f1']:.4f}\n")


def print_comparison_table() -> None:
    print("━" * 60)
    print("  [3/5] MODEL COMPARISON")
    print("━" * 60)

    if_path = Path(MODEL_IF_PATH) / "metrics.json"
    ae_path = Path(MODEL_AE_PATH) / "metrics.json"

    if not if_path.exists() or not ae_path.exists():
        print("  ⚠ metrics.json not found — skipping comparison\n")
        return

    with open(if_path) as f:
        if_m = json.load(f)
    with open(ae_path) as f:
        ae_m = json.load(f)

    from pipeline.evaluate import compare_models
    compare_models(if_m, ae_m)


def start_watchdog(app) -> None:
    print("━" * 60)
    print("  [4/5] DRIFT WATCHDOG")
    print("━" * 60)
    from pipeline.drift_watchdog import DriftWatchdog
    registry = app.config.get("MODEL_REGISTRY")
    watchdog = DriftWatchdog(model_registry=registry)
    watchdog.start()
    app.config["DRIFT_WATCHDOG"] = watchdog
    print("  ✓ DriftWatchdog daemon started\n")


def main() -> None:
    print("\n" + "═" * 60)
    print("  UPI ANOMALY DETECTOR — Payment Operations Intelligence")
    print("═" * 60 + "\n")

    # ── Step 1: Data pipeline ──────────────────────────────────────────────
    if not _processed_splits_exist():
        run_data_pipeline()
    else:
        print("  [1/5] Processed splits found — skipping ingestion ✓\n")

    # ── Step 2: Training ───────────────────────────────────────────────────
    if not _models_exist():
        run_training()
    else:
        print("  [2/5] Trained models found — skipping training ✓\n")

    # ── Step 3: Comparison table ───────────────────────────────────────────
    print_comparison_table()

    # ── Step 4: Flask app + Watchdog ───────────────────────────────────────
    print("━" * 60)
    print("  [5/5] STARTING FLASK SERVER")
    print("━" * 60)

    from api.app import create_app
    app = create_app()
    start_watchdog(app)

    print(f"\n  ✓ Dashboard running at http://localhost:{FLASK_PORT}")
    print(f"  ✓ API base:       http://localhost:{FLASK_PORT}/api/v1")
    print(f"  ✓ Health check:   http://localhost:{FLASK_PORT}/api/v1/health")
    print("\n" + "═" * 60 + "\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
