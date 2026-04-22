"""
api/routes/score.py — POST /api/v1/score endpoint.

Accepts a transaction payload, runs it through both the Isolation Forest
and Autoencoder models, and returns a structured risk assessment.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Blueprint, jsonify, request, current_app

from config import LOG_DATE_FORMAT, LOG_FORMAT, RISK_VELOCITY_CRITICAL_THRESHOLD

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

score_bp = Blueprint("score", __name__)


def _compute_risk_level(
    consensus: bool, is_fraud_if: bool, is_fraud_ae: bool, txn_velocity: int
) -> str:
    """Determine risk level from model outputs and transaction velocity.

    Args:
        consensus: Both models agree it's fraud.
        is_fraud_if: Isolation Forest flag.
        is_fraud_ae: Autoencoder flag.
        txn_velocity: Number of transactions in last hour.

    Returns:
        Risk level string: LOW | MEDIUM | HIGH | CRITICAL.
    """
    if consensus and txn_velocity > RISK_VELOCITY_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if consensus:
        return "HIGH"
    if is_fraud_if or is_fraud_ae:
        return "MEDIUM"
    return "LOW"


def _routing_action(risk_level: str) -> str:
    """Derive payment routing action from risk level.

    Args:
        risk_level: LOW | MEDIUM | HIGH | CRITICAL.

    Returns:
        ALLOW | FLAG | BLOCK.
    """
    return {"LOW": "ALLOW", "MEDIUM": "FLAG", "HIGH": "BLOCK", "CRITICAL": "BLOCK"}.get(
        risk_level, "FLAG"
    )


@score_bp.route("/api/v1/score", methods=["POST"])
def score_transaction():
    """Score a UPI transaction for anomaly detection.

    Request body (JSON):
        amount (float): Transaction amount in INR.
        hour_of_day (int): 0–23.
        day_of_week (int): 0–6.
        merchant_category (str): grocery | fuel | ecommerce | utilities | P2P.
        txn_velocity_1h (int): Transactions from same UPI ID in last hour.
        upi_id (str): Sending UPI handle.
        v_features (list[float]): 28 PCA features (optional, zeros if absent).

    Returns:
        JSON response with risk assessment.
    """
    t_start = time.time()

    try:
        body: Dict[str, Any] = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid_json", "message": "Request body must be valid JSON"}), 400

    # Validate required fields
    required = ["amount", "hour_of_day", "day_of_week", "merchant_category",
                "txn_velocity_1h", "upi_id"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": "missing_fields", "fields": missing}), 422

    try:
        amount = float(body["amount"])
        hour_of_day = int(body["hour_of_day"])
        day_of_week = int(body["day_of_week"])
        merchant_category = str(body["merchant_category"])
        txn_velocity = int(body["txn_velocity_1h"])
        upi_id = str(body["upi_id"])
        v_features = body.get("v_features", [0.0] * 28)
        if len(v_features) != 28:
            v_features = ([float(x) for x in v_features] + [0.0] * 28)[:28]
    except (ValueError, TypeError) as exc:
        return jsonify({"error": "type_error", "message": str(exc)}), 422

    # Score with both models
    registry = current_app.config.get("MODEL_REGISTRY")
    if registry is None:
        return jsonify({"error": "models_not_loaded", "message": "Model registry unavailable"}), 503

    try:
        score_if, recon_ae = registry.score(
            v_features=v_features,
            amount=amount,
            hour_of_day=hour_of_day,
            day_of_week=day_of_week,
            merchant_category=merchant_category,
            txn_velocity_1h=txn_velocity,
        )
    except Exception as exc:
        logger.error("Scoring error: %s", exc)
        return jsonify({"error": "scoring_error", "message": str(exc)}), 500

    is_fraud_if = bool(score_if >= registry.threshold_if)
    is_fraud_ae = bool(recon_ae >= registry.threshold_ae)
    consensus = is_fraud_if and is_fraud_ae

    risk_level = _compute_risk_level(consensus, is_fraud_if, is_fraud_ae, txn_velocity)
    routing_action = _routing_action(risk_level)
    latency_ms = round((time.time() - t_start) * 1000, 2)
    txn_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    result = {
        "transaction_id": txn_id,
        "upi_id": upi_id,
        "amount": amount,
        "merchant_category": merchant_category,
        "hour_of_day": hour_of_day,
        "is_fraud_if": is_fraud_if,
        "is_fraud_ae": is_fraud_ae,
        "consensus": consensus,
        "anomaly_score_if": round(float(score_if), 6),
        "reconstruction_error_ae": round(float(recon_ae), 6),
        "risk_level": risk_level,
        "routing_action": routing_action,
        "model_version": registry.version,
        "latency_ms": latency_ms,
        "timestamp": timestamp,
        "txn_velocity_1h": txn_velocity,
    }

    # Append to transaction log
    txn_log = current_app.config.get("TXN_LOG")
    txn_lock = current_app.config.get("TXN_LOCK")
    if txn_log is not None and txn_lock is not None:
        with txn_lock:
            txn_log.appendleft(result)

    # Record in drift watchdog
    watchdog = current_app.config.get("DRIFT_WATCHDOG")
    if watchdog:
        watchdog.record_transaction(result)

    logger.info(
        "UPI Score | txn_id=%s | upi_id=%s | amount=%.2f | risk=%s | action=%s | latency=%.1fms",
        txn_id, upi_id[:8], amount, risk_level, routing_action, latency_ms,
    )

    return jsonify(result), 200
