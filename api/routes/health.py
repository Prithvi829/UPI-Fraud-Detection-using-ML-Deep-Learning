"""
api/routes/health.py — Health check and drift status endpoints.
"""

from __future__ import annotations

import logging
import time

from flask import Blueprint, jsonify, current_app

from config import LOG_DATE_FORMAT, LOG_FORMAT

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

health_bp = Blueprint("health", __name__)
_server_start = time.time()


@health_bp.route("/api/v1/health", methods=["GET"])
def health_check():
    """Return service health and UPI model drift status.

    Returns:
        JSON health payload.
    """
    registry = current_app.config.get("MODEL_REGISTRY")
    watchdog = current_app.config.get("DRIFT_WATCHDOG")
    txn_log = current_app.config.get("TXN_LOG")
    txn_lock = current_app.config.get("TXN_LOCK")

    models_loaded = registry is not None and registry.is_ready()
    drift_detected = False
    watchdog_status = {}

    if watchdog:
        ws = watchdog.get_status()
        drift_detected = ws.get("drift_detected", False)
        watchdog_status = ws

    if txn_log is not None and txn_lock is not None:
        with txn_lock:
            txn_count = len(txn_log)
    else:
        txn_count = 0

    status = "healthy" if models_loaded and not drift_detected else "degraded"

    return jsonify({
        "status": status,
        "drift_detected": drift_detected,
        "models_loaded": models_loaded,
        "uptime_seconds": round(time.time() - _server_start, 1),
        "transaction_count": txn_count,
        "drift_watchdog": watchdog_status,
    }), 200


@health_bp.route("/api/v1/events/drift_alerts", methods=["GET"])
def drift_alerts():
    """Return recent drift alert events.

    Returns:
        JSON list of up to 10 drift alert records.
    """
    watchdog = current_app.config.get("DRIFT_WATCHDOG")
    alerts = watchdog.get_alerts()[:10] if watchdog else []
    return jsonify({"alerts": alerts, "count": len(alerts)}), 200
