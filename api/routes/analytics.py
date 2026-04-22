"""
api/routes/analytics.py — Analytics endpoints.

Aggregates in-memory transaction data into summary statistics,
time-series, and model metrics.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

from flask import Blueprint, jsonify, request, current_app

from config import LOG_DATE_FORMAT, LOG_FORMAT, MODEL_IF_PATH, MODEL_AE_PATH

logging.basicConfig(format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger(__name__)

analytics_bp = Blueprint("analytics", __name__)


def _get_txns(app) -> List[Dict[str, Any]]:
    """Thread-safe read of transaction log."""
    txn_log = app.config.get("TXN_LOG")
    txn_lock = app.config.get("TXN_LOCK")
    if txn_log is None:
        return []
    with txn_lock:
        return list(txn_log)


@analytics_bp.route("/api/v1/analytics/summary", methods=["GET"])
def analytics_summary():
    """Return aggregated UPI transaction analytics.

    Returns:
        JSON with totals, flag rates, breakdowns by category/hour/risk.
    """
    txns = _get_txns(current_app)

    total = len(txns)
    flagged = sum(1 for t in txns if t.get("consensus") or t.get("risk_level") in ("HIGH", "CRITICAL", "MEDIUM"))
    critical = sum(1 for t in txns if t.get("risk_level") == "CRITICAL")
    flag_rate = round(flagged / total * 100, 2) if total else 0.0
    latencies = [t.get("latency_ms", 0) for t in txns if t.get("latency_ms") is not None]
    avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # By merchant category
    by_cat: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "flagged": 0})
    for t in txns:
        cat = t.get("merchant_category", "unknown")
        by_cat[cat]["total"] += 1
        if t.get("risk_level") in ("MEDIUM", "HIGH", "CRITICAL"):
            by_cat[cat]["flagged"] += 1
    for cat, d in by_cat.items():
        d["flag_rate"] = round(d["flagged"] / d["total"] * 100, 2) if d["total"] else 0.0

    # By hour
    by_hour: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "flagged": 0})
    for t in txns:
        h = str(t.get("hour_of_day", 0))
        by_hour[h]["total"] += 1
        if t.get("risk_level") in ("MEDIUM", "HIGH", "CRITICAL"):
            by_hour[h]["flagged"] += 1

    # By risk level
    by_risk: Dict[str, int] = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for t in txns:
        rl = t.get("risk_level", "LOW")
        if rl in by_risk:
            by_risk[rl] += 1

    return jsonify({
        "total_scored": total,
        "flagged_count": flagged,
        "flag_rate_pct": flag_rate,
        "critical_count": critical,
        "avg_latency_ms": avg_latency,
        "by_merchant_category": dict(by_cat),
        "by_hour": dict(by_hour),
        "by_risk_level": by_risk,
    }), 200


@analytics_bp.route("/api/v1/analytics/timeseries", methods=["GET"])
def analytics_timeseries():
    """Return flagged transaction counts bucketed by minute.

    Query params:
        window (int): Number of minutes to look back (default 60).

    Returns:
        JSON with timestamps, flagged counts, and total counts per minute.
    """
    try:
        window = int(request.args.get("window", 60))
    except ValueError:
        window = 60

    txns = _get_txns(current_app)
    now = datetime.now(timezone.utc)

    buckets: Dict[str, Dict[str, int]] = {}
    for i in range(window):
        minute_ts = (now - timedelta(minutes=window - 1 - i)).replace(second=0, microsecond=0)
        key = minute_ts.isoformat()
        buckets[key] = {"total": 0, "flagged": 0}

    for t in txns:
        ts_str = t.get("timestamp", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            age_min = (now - ts).total_seconds() / 60
            if age_min > window:
                continue
            bucket_ts = ts.replace(second=0, microsecond=0).isoformat()
            if bucket_ts in buckets:
                buckets[bucket_ts]["total"] += 1
                if t.get("risk_level") in ("MEDIUM", "HIGH", "CRITICAL"):
                    buckets[bucket_ts]["flagged"] += 1
        except Exception:
            pass

    sorted_keys = sorted(buckets.keys())
    return jsonify({
        "timestamps": sorted_keys,
        "flagged": [buckets[k]["flagged"] for k in sorted_keys],
        "total": [buckets[k]["total"] for k in sorted_keys],
    }), 200


@analytics_bp.route("/api/v1/models/metrics", methods=["GET"])
def models_metrics():
    """Return saved metrics for both models side by side.

    Returns:
        JSON with isolation_forest and autoencoder metrics.
    """
    result: Dict[str, Any] = {}

    for name, path in [("isolation_forest", MODEL_IF_PATH), ("autoencoder", MODEL_AE_PATH)]:
        metrics_path = Path(path) / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                result[name] = json.load(f)
        else:
            result[name] = {"status": "not_trained"}

    return jsonify(result), 200
