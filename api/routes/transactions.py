"""
api/routes/transactions.py — GET /api/v1/transactions endpoint.

Returns the in-memory transaction log, optionally filtered to flagged only.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request, current_app

from config import LOG_DATE_FORMAT, LOG_FORMAT

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

transactions_bp = Blueprint("transactions", __name__)


@transactions_bp.route("/api/v1/transactions", methods=["GET"])
def get_transactions():
    """Return recent UPI transactions from the in-memory log.

    Query params:
        limit (int): Maximum number of transactions to return (default 100, max 1000).
        flagged_only (bool): If true, only return flagged/fraud transactions.

    Returns:
        JSON list of transaction records.
    """
    try:
        limit = min(int(request.args.get("limit", 100)), 1000)
        flagged_only = request.args.get("flagged_only", "false").lower() == "true"
    except ValueError:
        return jsonify({"error": "invalid_params", "message": "limit must be an integer"}), 400

    txn_log = current_app.config.get("TXN_LOG")
    txn_lock = current_app.config.get("TXN_LOCK")

    if txn_log is None:
        return jsonify({"transactions": [], "total": 0}), 200

    with txn_lock:
        txns = list(txn_log)

    if flagged_only:
        txns = [t for t in txns if t.get("consensus") or t.get("risk_level") in ("HIGH", "CRITICAL")]

    txns = txns[:limit]

    return jsonify({
        "transactions": txns,
        "total": len(txns),
        "flagged_only": flagged_only,
    }), 200
