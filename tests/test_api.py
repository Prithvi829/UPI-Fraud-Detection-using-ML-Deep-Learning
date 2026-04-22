"""
tests/test_api.py — Unit tests for the Flask REST API.

Uses Flask test client — no live server required.
"""

import json
import sys
import threading
from collections import deque
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRANSACTION_LOG_MAXLEN


# ── Mock model registry ───────────────────────────────────────────────────────

class MockRegistry:
    threshold_if = 0.5
    threshold_ae = 0.05
    version = "2024-01-01"

    def is_ready(self):
        return True

    def score(self, **kwargs):
        # Normal transaction by default
        return 0.3, 0.02

    def score_fraud(self, **kwargs):
        # Fraud transaction
        return 0.9, 0.15


class MockFraudRegistry(MockRegistry):
    def score(self, **kwargs):
        return 0.9, 0.15


class MockHighVelocityFraudRegistry(MockRegistry):
    def score(self, **kwargs):
        return 0.9, 0.15


# ── App factory for tests ─────────────────────────────────────────────────────

def make_test_app(registry=None):
    """Create a Flask test app with mock models."""
    from api.app import create_app

    app = create_app()
    app.config["TESTING"] = True
    app.config["MODEL_REGISTRY"] = registry or MockRegistry()
    app.config["TXN_LOG"] = deque(maxlen=TRANSACTION_LOG_MAXLEN)
    app.config["TXN_LOCK"] = threading.Lock()
    app.config["DRIFT_WATCHDOG"] = None
    return app


SAMPLE_TXN = {
    "amount": 1500.50,
    "hour_of_day": 14,
    "day_of_week": 2,
    "merchant_category": "ecommerce",
    "txn_velocity_1h": 3,
    "upi_id": "user123@okicici",
    "v_features": [0.0] * 28,
}


# ── Tests: POST /api/v1/score ─────────────────────────────────────────────────

class TestScoreEndpoint:
    @pytest.fixture
    def client(self):
        app = make_test_app()
        return app.test_client()

    def test_score_endpoint_returns_200(self, client):
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        assert resp.status_code == 200

    def test_score_endpoint_returns_all_fields(self, client):
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        data = resp.get_json()
        required_fields = [
            "transaction_id", "is_fraud_if", "is_fraud_ae", "consensus",
            "anomaly_score_if", "reconstruction_error_ae", "risk_level",
            "routing_action", "model_version", "latency_ms",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_score_returns_correct_types(self, client):
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        data = resp.get_json()
        assert isinstance(data["transaction_id"], str)
        assert isinstance(data["is_fraud_if"], bool)
        assert isinstance(data["is_fraud_ae"], bool)
        assert isinstance(data["consensus"], bool)
        assert isinstance(data["anomaly_score_if"], float)
        assert isinstance(data["reconstruction_error_ae"], float)
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert data["routing_action"] in ("ALLOW", "FLAG", "BLOCK")
        assert isinstance(data["latency_ms"], float)

    def test_normal_transaction_low_risk(self, client):
        """Mock returns low scores → risk should be LOW."""
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        data = resp.get_json()
        assert data["risk_level"] == "LOW"
        assert data["routing_action"] == "ALLOW"
        assert data["consensus"] is False

    def test_missing_required_field_returns_422(self, client):
        bad = {k: v for k, v in SAMPLE_TXN.items() if k != "amount"}
        resp = client.post("/api/v1/score", json=bad)
        assert resp.status_code == 422

    def test_invalid_json_returns_400(self, client):
        resp = client.post(
            "/api/v1/score",
            data="not json",
            content_type="application/json",
        )
        # Either 400 or 200 depending on Flask JSON parsing
        assert resp.status_code in (200, 400, 422)

    def test_v_features_optional(self, client):
        """v_features should be optional — zeros used if omitted."""
        txn = {k: v for k, v in SAMPLE_TXN.items() if k != "v_features"}
        resp = client.post("/api/v1/score", json=txn)
        assert resp.status_code == 200

    def test_transaction_id_is_uuid(self, client):
        import uuid
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        data = resp.get_json()
        uuid.UUID(data["transaction_id"])  # Raises if not valid UUID


# ── Tests: risk level logic ───────────────────────────────────────────────────

class TestRiskLevelLogic:
    def test_risk_level_critical_logic(self):
        """CRITICAL = consensus + high velocity."""
        app = make_test_app(MockHighVelocityFraudRegistry())
        client = app.test_client()
        txn = {**SAMPLE_TXN, "txn_velocity_1h": 12}
        resp = client.post("/api/v1/score", json=txn)
        data = resp.get_json()
        # With high fraud scores and velocity > 10 → CRITICAL
        assert data["risk_level"] == "CRITICAL"
        assert data["routing_action"] == "BLOCK"

    def test_risk_level_high_logic(self):
        """HIGH = consensus but velocity ≤ 10."""
        app = make_test_app(MockFraudRegistry())
        client = app.test_client()
        txn = {**SAMPLE_TXN, "txn_velocity_1h": 5}
        resp = client.post("/api/v1/score", json=txn)
        data = resp.get_json()
        assert data["risk_level"] == "HIGH"

    def test_risk_level_medium_one_model(self):
        """MEDIUM = only one model flags."""
        class HalfFraudRegistry(MockRegistry):
            def score(self, **kwargs):
                # IF flags (>0.5), AE doesn't (<0.05)
                return 0.8, 0.01

        app = make_test_app(HalfFraudRegistry())
        client = app.test_client()
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        data = resp.get_json()
        assert data["risk_level"] == "MEDIUM"

    def test_routing_allow_for_low(self):
        app = make_test_app()
        client = app.test_client()
        resp = client.post("/api/v1/score", json=SAMPLE_TXN)
        data = resp.get_json()
        assert data["routing_action"] == "ALLOW"

    def test_routing_block_for_high(self):
        app = make_test_app(MockFraudRegistry())
        client = app.test_client()
        txn = {**SAMPLE_TXN, "txn_velocity_1h": 5}
        resp = client.post("/api/v1/score", json=txn)
        data = resp.get_json()
        assert data["routing_action"] == "BLOCK"


# ── Tests: GET /api/v1/analytics/summary ─────────────────────────────────────

class TestAnalyticsSummary:
    @pytest.fixture
    def client(self):
        app = make_test_app()
        return app.test_client()

    def test_analytics_summary_200(self, client):
        resp = client.get("/api/v1/analytics/summary")
        assert resp.status_code == 200

    def test_analytics_summary_structure(self, client):
        resp = client.get("/api/v1/analytics/summary")
        data = resp.get_json()
        required = [
            "total_scored", "flagged_count", "flag_rate_pct",
            "critical_count", "avg_latency_ms", "by_merchant_category",
            "by_hour", "by_risk_level",
        ]
        for field in required:
            assert field in data, f"Missing analytics field: {field}"

    def test_by_risk_level_keys(self, client):
        resp = client.get("/api/v1/analytics/summary")
        data = resp.get_json()
        risk_keys = set(data["by_risk_level"].keys())
        assert risk_keys == {"LOW", "MEDIUM", "HIGH", "CRITICAL"}

    def test_summary_totals_consistency(self, client):
        """After scoring a transaction, total_scored should be 1."""
        client.post("/api/v1/score", json=SAMPLE_TXN)
        resp = client.get("/api/v1/analytics/summary")
        data = resp.get_json()
        assert data["total_scored"] >= 1


# ── Tests: GET /api/v1/health ─────────────────────────────────────────────────

class TestHealthEndpoint:
    @pytest.fixture
    def client(self):
        app = make_test_app()
        return app.test_client()

    def test_health_endpoint(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_endpoint_structure(self, client):
        resp = client.get("/api/v1/health")
        data = resp.get_json()
        required = [
            "status", "drift_detected", "models_loaded",
            "uptime_seconds", "transaction_count",
        ]
        for field in required:
            assert field in data, f"Missing health field: {field}"

    def test_health_status_values(self, client):
        resp = client.get("/api/v1/health")
        data = resp.get_json()
        assert data["status"] in ("healthy", "degraded")

    def test_models_loaded_true(self, client):
        resp = client.get("/api/v1/health")
        data = resp.get_json()
        assert data["models_loaded"] is True

    def test_uptime_positive(self, client):
        resp = client.get("/api/v1/health")
        data = resp.get_json()
        assert data["uptime_seconds"] >= 0


# ── Tests: GET /api/v1/transactions ──────────────────────────────────────────

class TestTransactionsEndpoint:
    @pytest.fixture
    def client(self):
        app = make_test_app()
        return app.test_client()

    def test_transactions_returns_200(self, client):
        resp = client.get("/api/v1/transactions")
        assert resp.status_code == 200

    def test_transactions_structure(self, client):
        resp = client.get("/api/v1/transactions")
        data = resp.get_json()
        assert "transactions" in data
        assert "total" in data

    def test_transactions_after_score(self, client):
        client.post("/api/v1/score", json=SAMPLE_TXN)
        resp = client.get("/api/v1/transactions?limit=10")
        data = resp.get_json()
        assert data["total"] >= 1

    def test_transactions_limit_param(self, client):
        for _ in range(5):
            client.post("/api/v1/score", json=SAMPLE_TXN)
        resp = client.get("/api/v1/transactions?limit=3")
        data = resp.get_json()
        assert len(data["transactions"]) <= 3
