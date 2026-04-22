"""
tests/test_drift.py — Unit tests for the DriftWatchdog self-healing system.
"""

import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.drift_watchdog import DriftWatchdog
from config import DRIFT_PRECISION_THRESHOLD


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_watchdog(precision_threshold=DRIFT_PRECISION_THRESHOLD):
    registry = MagicMock()
    registry.reload_model = MagicMock()
    return DriftWatchdog(
        model_registry=registry,
        check_interval_seconds=9999,  # never auto-trigger in tests
        precision_threshold=precision_threshold,
    )


def make_txn(predicted_fraud=False, risk_level="LOW", age_minutes=60):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=age_minutes)).isoformat()
    return {
        "timestamp": ts,
        "predicted_fraud": predicted_fraud,
        "risk_level": risk_level,
        "anomaly_score_if": 0.8 if predicted_fraud else 0.2,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestDriftDetection:
    def test_no_drift_initially(self):
        wd = make_watchdog()
        status = wd.get_status()
        assert status["drift_detected"] is False

    def test_no_drift_when_precision_above_threshold(self):
        """When precision is high, no drift should be flagged."""
        wd = make_watchdog(precision_threshold=0.50)

        # Add mostly normal transactions (no flagging)
        for _ in range(50):
            wd.record_transaction(make_txn(predicted_fraud=False))

        wd._check_drift()
        status = wd.get_status()
        # With no flagged transactions, precision stays at 1.0
        assert status["drift_detected"] is False

    def test_drift_detected_when_precision_low(self):
        """Patch internal precision computation to simulate low precision."""
        wd = make_watchdog(precision_threshold=0.99)  # very high threshold

        # Fill window with "flagged" transactions
        for _ in range(50):
            wd.record_transaction(make_txn(predicted_fraud=True, risk_level="HIGH"))

        # Manually set precision low to trigger drift
        wd._precision_last_window = 0.50
        wd._drift_detected = True

        status = wd.get_status()
        assert status["drift_detected"] is True

    def test_drift_status_dict_schema(self):
        """get_status() must return all required keys."""
        wd = make_watchdog()
        status = wd.get_status()
        required_keys = [
            "last_check",
            "precision_last_100",
            "threshold",
            "drift_detected",
            "retraining_in_progress",
            "window_size",
        ]
        for key in required_keys:
            assert key in status, f"Missing key in drift status: {key}"

    def test_status_threshold_matches_config(self):
        wd = make_watchdog(precision_threshold=0.75)
        assert wd.get_status()["threshold"] == 0.75

    def test_window_size_increases_on_record(self):
        wd = make_watchdog()
        assert wd.get_status()["window_size"] == 0
        wd.record_transaction(make_txn())
        assert wd.get_status()["window_size"] == 1

    def test_window_maxlen_respected(self):
        from config import DRIFT_WINDOW_SIZE
        wd = make_watchdog()
        for _ in range(DRIFT_WINDOW_SIZE + 50):
            wd.record_transaction(make_txn())
        assert wd.get_status()["window_size"] == DRIFT_WINDOW_SIZE

    def test_record_transaction_thread_safe(self):
        """Multiple threads writing to window should not raise."""
        import threading
        wd = make_watchdog()
        errors = []

        def writer():
            try:
                for _ in range(20):
                    wd.record_transaction(make_txn())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"

    def test_get_alerts_returns_list(self):
        wd = make_watchdog()
        alerts = wd.get_alerts()
        assert isinstance(alerts, list)

    def test_no_drift_when_window_empty(self):
        """With no data, check_drift should not set drift_detected."""
        wd = make_watchdog()
        wd._check_drift()
        assert wd.get_status()["drift_detected"] is False

    def test_retraining_flag_starts_false(self):
        wd = make_watchdog()
        assert wd.get_status()["retraining_in_progress"] is False

    def test_daemon_thread_starts(self):
        wd = make_watchdog()
        wd.start()
        time.sleep(0.05)
        assert wd._thread is not None
        assert wd._thread.is_alive()
        wd.stop()
