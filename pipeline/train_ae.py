"""
pipeline/train_ae.py — Autoencoder training for UPI anomaly detection.

Trains a PyTorch autoencoder on normal transactions only.
Anomaly threshold is set at the 95th percentile of reconstruction errors
on validation normal transactions.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    AE_BATCH_SIZE,
    AE_BOTTLENECK,
    AE_EPOCHS,
    AE_HIDDEN_DIMS,
    AE_LR,
    AE_PATIENCE,
    ANOMALY_PERCENTILE_THRESHOLD,
    FEATURE_COLS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    MODEL_AE_PATH,
    RANDOM_STATE,
)
from pipeline.evaluate import evaluate_model

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_feature_cols(df: pd.DataFrame) -> list:
    desired = FEATURE_COLS + ["Amount_log", "amount_zscore", "is_night", "high_velocity"]
    return [c for c in desired if c in df.columns]


def _build_autoencoder(input_dim: int):
    """Build PyTorch Autoencoder with Encoder: input→32→16→8, Decoder: 8→16→32→input."""
    try:
        import torch
        import torch.nn as nn

        class Autoencoder(nn.Module):
            def __init__(self, input_dim: int):
                super().__init__()
                hidden = AE_HIDDEN_DIMS  # [32, 16]
                bottleneck = AE_BOTTLENECK  # 8

                # Encoder
                enc_layers = []
                prev = input_dim
                for h in hidden:
                    enc_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                    prev = h
                enc_layers.append(nn.Linear(prev, bottleneck))
                enc_layers.append(nn.ReLU())
                self.encoder = nn.Sequential(*enc_layers)

                # Decoder
                dec_layers = []
                prev = bottleneck
                for h in reversed(hidden):
                    dec_layers.extend([nn.Linear(prev, h), nn.ReLU()])
                    prev = h
                dec_layers.append(nn.Linear(prev, input_dim))  # Linear output
                self.decoder = nn.Sequential(*dec_layers)

            def forward(self, x):
                return self.decoder(self.encoder(x))

        return Autoencoder(input_dim)

    except ImportError:
        logger.error("PyTorch not available — cannot build autoencoder")
        raise


def train_autoencoder(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = MODEL_AE_PATH,
) -> Dict[str, Any]:
    """Train the Autoencoder model on normal transactions.

    Args:
        train_df: Training split.
        val_df: Validation split.
        test_df: Test split.
        output_dir: Directory to save model artefacts.

    Returns:
        Metrics dictionary.
    """
    t0 = time.time()
    logger.info("UPI AE Training | Starting Autoencoder training")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        torch.manual_seed(RANDOM_STATE)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
    except ImportError:
        logger.error("PyTorch not installed — cannot train Autoencoder")
        raise

    feature_cols = _get_feature_cols(train_df)
    logger.info("Using %d feature columns", len(feature_cols))

    # Fit scaler on normal training transactions only
    normal_train = train_df[train_df["is_fraud"] == 0]
    scaler = StandardScaler()
    X_train_normal = scaler.fit_transform(normal_train[feature_cols].fillna(0))

    # Validation normals
    normal_val = val_df[val_df["is_fraud"] == 0]
    X_val_normal = scaler.transform(normal_val[feature_cols].fillna(0))

    # Full validation + test for threshold-based evaluation
    X_val_all = scaler.transform(val_df[feature_cols].fillna(0))
    y_val_all = val_df["is_fraud"].values

    X_test_all = scaler.transform(test_df[feature_cols].fillna(0))
    y_test_all = test_df["is_fraud"].values

    input_dim = X_train_normal.shape[1]
    model = _build_autoencoder(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR)
    criterion = nn.MSELoss()

    # DataLoaders
    train_tensor = torch.FloatTensor(X_train_normal).to(device)
    val_tensor = torch.FloatTensor(X_val_normal).to(device)
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=AE_BATCH_SIZE,
        shuffle=True,
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    train_losses: List[float] = []
    val_losses: List[float] = []

    logger.info("Training for up to %d epochs | batch_size=%d | lr=%s", AE_EPOCHS, AE_BATCH_SIZE, AE_LR)

    for epoch in range(1, AE_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, _ in train_loader:
            optimizer.zero_grad()
            recon = model(x_batch)
            loss = criterion(recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(x_batch)
        epoch_loss /= len(X_train_normal)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = criterion(val_recon, val_tensor).item()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                "Epoch %02d/%d | train_loss=%.6f | val_loss=%.6f",
                epoch, AE_EPOCHS, epoch_loss, val_loss,
            )

        # Early stopping
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= AE_PATIENCE:
                logger.info("Early stopping at epoch %d | best_val_loss=%.6f", epoch, best_val_loss)
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute anomaly threshold: 95th percentile of val normal reconstruction errors
    model.eval()
    with torch.no_grad():
        val_normal_tensor = torch.FloatTensor(X_val_normal).to(device)
        val_recon = model(val_normal_tensor)
        val_errors = ((val_recon - val_normal_tensor) ** 2).mean(dim=1).cpu().numpy()

    threshold = float(np.percentile(val_errors, ANOMALY_PERCENTILE_THRESHOLD))
    logger.info("Anomaly threshold (p%d): %.6f", ANOMALY_PERCENTILE_THRESHOLD, threshold)

    # Evaluate on test set
    with torch.no_grad():
        test_tensor = torch.FloatTensor(X_test_all).to(device)
        test_recon = model(test_tensor)
        test_errors = ((test_recon - test_tensor) ** 2).mean(dim=1).cpu().numpy()

    test_metrics = evaluate_model(y_test_all, test_errors, threshold, "Autoencoder-final")

    # Save artefacts
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(Path(output_dir) / "model.pt"))
    # Save model architecture info for loading
    arch_info = {"input_dim": input_dim, "hidden_dims": AE_HIDDEN_DIMS, "bottleneck": AE_BOTTLENECK}
    with open(str(Path(output_dir) / "arch.json"), "w") as f:
        json.dump(arch_info, f)
    joblib.dump(scaler, str(Path(output_dir) / "scaler.pkl"))

    metrics_payload = {
        **test_metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train_samples": int(X_train_normal.shape[0]),
        "contamination": float(ANOMALY_PERCENTILE_THRESHOLD / 100),
        "feature_cols": feature_cols,
        "training_seconds": round(time.time() - t0, 2),
        "train_losses": train_losses[-5:],
        "val_losses": val_losses[-5:],
    }
    with open(str(Path(output_dir) / "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info(
        "UPI AE Training complete | test_f1=%.4f | saved to %s | elapsed=%.1fs",
        test_metrics["f1"],
        output_dir,
        time.time() - t0,
    )
    return metrics_payload


if __name__ == "__main__":
    from pipeline.features import load_processed_splits
    train_df, val_df, test_df = load_processed_splits()
    train_autoencoder(train_df, val_df, test_df)
