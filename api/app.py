"""
api/app.py — Flask application factory for UPI Anomaly Detector.

Implements the ModelRegistry for thread-safe model access and hot-swapping,
and the create_app() factory that wires together blueprints, middleware,
and shared state.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from flask import Flask
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    AE_BOTTLENECK,
    AE_HIDDEN_DIMS,
    DATA_PROCESSED_PATH,
    FEATURE_COLS,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    MODEL_AE_PATH,
    MODEL_IF_PATH,
    TRANSACTION_LOG_MAXLEN,
)

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe registry for Isolation Forest and Autoencoder models.

    Supports hot-swapping models on retrain without restarting the service.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._if_model = None
        self._if_scaler = None
        self._ae_model = None
        self._ae_scaler = None
        self._ae_arch = None
        self.threshold_if: float = 0.5
        self.threshold_ae: float = 0.5
        self.version: str = "unloaded"
        self._feature_cols: list = []
        self._ready = False

    def load_all(self) -> None:
        """Load both models from disk."""
        self._load_isolation_forest()
        self._load_autoencoder()
        self._ready = self._if_model is not None or self._ae_model is not None
        logger.info("ModelRegistry | loaded | IF=%s | AE=%s",
                    "✓" if self._if_model else "✗",
                    "✓" if self._ae_model else "✗")

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready

    def _load_isolation_forest(self) -> None:
        model_path = Path(MODEL_IF_PATH) / "model.pkl"
        scaler_path = Path(MODEL_IF_PATH) / "scaler.pkl"
        metrics_path = Path(MODEL_IF_PATH) / "metrics.json"

        if not model_path.exists():
            logger.warning("IF model not found at %s", model_path)
            return

        with self._lock:
            self._if_model = joblib.load(str(model_path))
            self._if_scaler = joblib.load(str(scaler_path))

            if metrics_path.exists():
                with open(metrics_path) as f:
                    metrics = json.load(f)
                self.threshold_if = float(metrics.get("threshold", 0.5))
                self._feature_cols = metrics.get("feature_cols", FEATURE_COLS)
                self.version = metrics.get("trained_at", "v1")[:10]

        logger.info("IF model loaded | threshold=%.6f", self.threshold_if)

    def _load_autoencoder(self) -> None:
        arch_path = Path(MODEL_AE_PATH) / "arch.json"
        model_path = Path(MODEL_AE_PATH) / "model.pt"
        scaler_path = Path(MODEL_AE_PATH) / "scaler.pkl"
        metrics_path = Path(MODEL_AE_PATH) / "metrics.json"

        if not model_path.exists():
            logger.warning("AE model not found at %s", model_path)
            return

        try:
            import torch
            import torch.nn as nn

            with open(arch_path) as f:
                arch = json.load(f)

            input_dim = arch["input_dim"]
            hidden_dims = arch.get("hidden_dims", AE_HIDDEN_DIMS)
            bottleneck = arch.get("bottleneck", AE_BOTTLENECK)

            class Autoencoder(nn.Module):
                def __init__(self):
                    super().__init__()
                    enc = []
                    prev = input_dim
                    for h in hidden_dims:
                        enc.extend([nn.Linear(prev, h), nn.ReLU()])
                        prev = h
                    enc.extend([nn.Linear(prev, bottleneck), nn.ReLU()])
                    self.encoder = nn.Sequential(*enc)
                    dec = []
                    prev = bottleneck
                    for h in reversed(hidden_dims):
                        dec.extend([nn.Linear(prev, h), nn.ReLU()])
                        prev = h
                    dec.append(nn.Linear(prev, input_dim))
                    self.decoder = nn.Sequential(*dec)

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            model = Autoencoder()
            model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
            model.eval()

            with self._lock:
                self._ae_model = model
                self._ae_scaler = joblib.load(str(scaler_path))
                self._ae_arch = arch

                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    self.threshold_ae = float(metrics.get("threshold", 0.5))

            logger.info("AE model loaded | threshold=%.6f", self.threshold_ae)

        except ImportError:
            logger.warning("PyTorch not available — AE model not loaded")
        except Exception as exc:
            logger.error("Failed to load AE model: %s", exc)

    def reload_model(self, model_name: str) -> None:
        """Hot-swap a model after retraining.

        Args:
            model_name: 'isolation_forest' or 'autoencoder'.
        """
        logger.info("Hot-swapping model: %s", model_name)
        if model_name == "isolation_forest":
            self._load_isolation_forest()
        elif model_name == "autoencoder":
            self._load_autoencoder()

    def _build_feature_vector(
        self,
        v_features: list,
        amount: float,
        hour_of_day: int,
        day_of_week: int,
        merchant_category: str,
        txn_velocity_1h: int,
    ) -> np.ndarray:
        """Assemble feature vector matching training schema.

        Returns:
            1D numpy array.
        """
        import math

        v = [float(x) for x in v_features[:28]] + [0.0] * max(0, 28 - len(v_features))
        amount_log = math.log1p(amount)
        is_night = 1 if hour_of_day in set(range(6)) | {22, 23} else 0
        high_velocity = 1 if txn_velocity_1h > 10 else 0
        amount_zscore = (amount - 200.0) / 300.0  # rough global estimate for inference

        # Build ordered feature vector matching FEATURE_COLS + engineered
        vec = v + [amount, hour_of_day, day_of_week, txn_velocity_1h,
                   amount_log, amount_zscore, is_night, high_velocity]
        return np.array(vec, dtype=np.float32)

    def score(
        self,
        v_features: list,
        amount: float,
        hour_of_day: int,
        day_of_week: int,
        merchant_category: str,
        txn_velocity_1h: int,
    ) -> Tuple[float, float]:
        """Score a transaction with both models.

        Returns:
            Tuple of (anomaly_score_if, reconstruction_error_ae).
        """
        raw_vec = self._build_feature_vector(
            v_features, amount, hour_of_day, day_of_week, merchant_category, txn_velocity_1h
        )

        score_if = 0.5
        recon_ae = 0.0

        with self._lock:
            # Isolation Forest
            if self._if_model is not None and self._if_scaler is not None:
                try:
                    n_if_features = self._if_scaler.n_features_in_
                    x_if = raw_vec[:n_if_features].reshape(1, -1)
                    x_if_scaled = self._if_scaler.transform(x_if)
                    score_if = float(-self._if_model.score_samples(x_if_scaled)[0])
                except Exception as exc:
                    logger.error("IF scoring error: %s", exc)

            # Autoencoder
            if self._ae_model is not None and self._ae_scaler is not None:
                try:
                    import torch
                    n_ae_features = self._ae_scaler.n_features_in_
                    x_ae = raw_vec[:n_ae_features].reshape(1, -1)
                    x_ae_scaled = self._ae_scaler.transform(x_ae)
                    tensor = torch.FloatTensor(x_ae_scaled)
                    with torch.no_grad():
                        recon = self._ae_model(tensor)
                    recon_ae = float(((recon - tensor) ** 2).mean().item())
                except Exception as exc:
                    logger.error("AE scoring error: %s", exc)

        return score_if, recon_ae


def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """Flask application factory.

    Args:
        config: Optional dict of Flask config overrides.

    Returns:
        Configured Flask application.
    """
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent.parent / "dashboard" / "templates"),
        static_folder=str(Path(__file__).parent.parent / "dashboard" / "static"),
    )

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    if config:
        app.config.update(config)

    # ── Shared state ──────────────────────────────────────────────────────────
    txn_log = deque(maxlen=TRANSACTION_LOG_MAXLEN)
    txn_lock = threading.Lock()
    app.config["TXN_LOG"] = txn_log
    app.config["TXN_LOCK"] = txn_lock

    # ── Model registry ────────────────────────────────────────────────────────
    registry = ModelRegistry()
    registry.load_all()
    app.config["MODEL_REGISTRY"] = registry

    # ── Register blueprints ───────────────────────────────────────────────────
    from api.routes.score import score_bp
    from api.routes.transactions import transactions_bp
    from api.routes.analytics import analytics_bp
    from api.routes.health import health_bp

    app.register_blueprint(score_bp)
    app.register_blueprint(transactions_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(health_bp)

    # ── Dashboard route ───────────────────────────────────────────────────────
    from flask import render_template

    @app.route("/")
    def dashboard():
        return render_template("index.html")

    # ── Middleware hooks ──────────────────────────────────────────────────────
    from api.middleware import log_request, set_start_time
    app.before_request(set_start_time)
    app.after_request(log_request)

    # ── Error handlers ────────────────────────────────────────────────────────
    from flask import jsonify as _jsonify

    @app.errorhandler(404)
    def not_found(e):
        return _jsonify({"error": "not_found", "path": str(e)}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return _jsonify({"error": "method_not_allowed"}), 405

    @app.errorhandler(500)
    def internal_error(e):
        logger.error("Internal server error: %s", e)
        return _jsonify({"error": "internal_server_error", "message": str(e)}), 500

    logger.info("Flask app created | blueprints registered | models_ready=%s", registry.is_ready())
    return app
