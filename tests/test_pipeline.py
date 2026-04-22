"""
tests/test_pipeline.py — Unit tests for the data ingestion and feature pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.ingest import add_upi_columns, load_raw_data, validate_schema
from pipeline.features import engineer_features, split_data


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_df():
    """Minimal synthetic DataFrame matching creditcard.csv schema."""
    rng = np.random.default_rng(42)
    n = 1000
    data = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    data["Time"] = rng.uniform(0, 172_792, n)
    data["Amount"] = rng.lognormal(3, 1.5, n)
    data["is_fraud"] = (rng.random(n) < 0.05).astype(int)
    return pd.DataFrame(data)


@pytest.fixture
def enriched_df(synthetic_df):
    """Synthetic DataFrame with UPI columns added."""
    return add_upi_columns(synthetic_df)


@pytest.fixture
def engineered_df(enriched_df):
    """Synthetic DataFrame with engineered features."""
    return engineer_features(enriched_df)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSyntheticDataSchema:
    def test_synthetic_data_schema(self, synthetic_df):
        """All required base columns must be present."""
        required = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount", "is_fraud"]
        for col in required:
            assert col in synthetic_df.columns, f"Missing column: {col}"

    def test_upi_columns_added(self, enriched_df):
        """UPI metadata columns must be present after enrichment."""
        upi_cols = ["upi_id", "merchant_category", "hour_of_day", "day_of_week", "txn_velocity_1h"]
        for col in upi_cols:
            assert col in enriched_df.columns, f"Missing UPI column: {col}"

    def test_validate_schema_passes(self, enriched_df):
        """validate_schema must not raise on a valid DataFrame."""
        validate_schema(enriched_df)

    def test_validate_schema_raises_on_missing(self, synthetic_df):
        """validate_schema must raise ValueError when UPI columns are absent."""
        with pytest.raises(ValueError, match="missing columns"):
            validate_schema(synthetic_df)

    def test_upi_id_format(self, enriched_df):
        """UPI IDs must contain '@' separator."""
        assert enriched_df["upi_id"].str.contains("@").all()

    def test_merchant_categories_valid(self, enriched_df):
        from config import MERCHANT_CATEGORIES
        assert enriched_df["merchant_category"].isin(MERCHANT_CATEGORIES).all()

    def test_hour_of_day_range(self, enriched_df):
        assert enriched_df["hour_of_day"].between(0, 23).all()

    def test_day_of_week_range(self, enriched_df):
        assert enriched_df["day_of_week"].between(0, 6).all()

    def test_fraud_rate(self, synthetic_df):
        """Fraud rate should be between 0.01% and 20%."""
        rate = synthetic_df["is_fraud"].mean()
        assert 0.0001 < rate < 0.20


class TestSmoteClassBalance:
    def test_smote_class_balance(self, engineered_df):
        """After SMOTE, training fraud rate should be 40–60%."""
        pytest.importorskip("imblearn")
        train_df, val_df, test_df = split_data(engineered_df)
        fraud_rate = train_df["is_fraud"].mean()
        assert 0.40 <= fraud_rate <= 0.60, (
            f"Expected 40–60% fraud rate after SMOTE, got {fraud_rate:.2%}"
        )

    def test_val_test_splits_preserved(self, engineered_df):
        """Val and test splits must preserve original class ratio (not SMOTE'd)."""
        train_df, val_df, test_df = split_data(engineered_df)
        # Val and test should NOT be near 50% (they reflect original ~5% rate)
        assert val_df["is_fraud"].mean() < 0.30
        assert test_df["is_fraud"].mean() < 0.30

    def test_no_data_leakage(self, engineered_df):
        """Train + val + test rows must sum to original (before SMOTE inflates train)."""
        original_len = len(engineered_df)
        train_df, val_df, test_df = split_data(engineered_df)
        # Val + test should be ~30% of original
        assert len(val_df) + len(test_df) <= original_len


class TestFeatureEngineering:
    def test_amount_log_created(self, engineered_df):
        """Amount_log column must exist and be non-negative."""
        assert "Amount_log" in engineered_df.columns
        assert (engineered_df["Amount_log"] >= 0).all()

    def test_amount_zscore_created(self, engineered_df):
        """amount_zscore column must exist."""
        assert "amount_zscore" in engineered_df.columns

    def test_is_night_created(self, engineered_df):
        """is_night column must be binary."""
        assert "is_night" in engineered_df.columns
        assert set(engineered_df["is_night"].unique()).issubset({0, 1})

    def test_high_velocity_created(self, engineered_df):
        """high_velocity column must be binary."""
        assert "high_velocity" in engineered_df.columns
        assert set(engineered_df["high_velocity"].unique()).issubset({0, 1})

    def test_is_night_logic(self, enriched_df):
        """is_night must be 1 for hours 0-5 and 22-23."""
        df = engineer_features(enriched_df)
        night_hours = set(range(6)) | {22, 23}
        for _, row in df.sample(min(50, len(df))).iterrows():
            expected = 1 if row["hour_of_day"] in night_hours else 0
            assert row["is_night"] == expected

    def test_high_velocity_logic(self, enriched_df):
        """high_velocity must be 1 when txn_velocity_1h > 10."""
        df = engineer_features(enriched_df)
        for _, row in df.sample(min(50, len(df))).iterrows():
            expected = 1 if row["txn_velocity_1h"] > 10 else 0
            assert row["high_velocity"] == expected

    def test_amount_log_values(self, enriched_df):
        """Amount_log must equal log1p(Amount)."""
        import numpy as np
        df = engineer_features(enriched_df)
        expected = np.log1p(df["Amount"])
        np.testing.assert_allclose(df["Amount_log"], expected, rtol=1e-5)
