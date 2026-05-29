# =============================================================================
# tests/conftest.py — shared pytest fixtures for OP'26 test suite
# =============================================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, settings
from unittest.mock import MagicMock

from src.config import ForecastState, PricingDecision

# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis profile
# ─────────────────────────────────────────────────────────────────────────────
settings.register_profile(
    "ci",
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")


# ─────────────────────────────────────────────────────────────────────────────
# Sample DataFrames
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_acn_df() -> pd.DataFrame:
    """Minimal ACN DataFrame with required columns."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "connectionTime": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "kWhDelivered": rng.uniform(1.0, 50.0, n),
        "stationID": [f"S{i % 10}" for i in range(n)],
    })


@pytest.fixture
def sample_acn_df_with_zeros() -> pd.DataFrame:
    """ACN DataFrame containing some zero/null kWh rows."""
    rng = np.random.default_rng(42)
    n = 100
    kwh = rng.uniform(1.0, 50.0, n)
    # Inject zeros and nulls
    kwh[::10] = 0.0
    kwh[5::10] = np.nan
    return pd.DataFrame({
        "connectionTime": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "kWhDelivered": kwh,
        "stationID": [f"S{i % 5}" for i in range(n)],
    })


@pytest.fixture
def sample_urban_wide() -> dict[str, pd.DataFrame]:
    """Wide-format UrbanEV DataFrames (volume, occupancy, duration)."""
    rng = np.random.default_rng(42)
    n_time = 48
    n_stations = 5
    station_cols = [f"node_{i}" for i in range(n_stations)]
    time_col = [f"t{i}" for i in range(n_time)]

    def _make(low, high):
        data = {col: rng.uniform(low, high, n_time) for col in station_cols}
        df = pd.DataFrame(data, index=time_col)
        df.index.name = "time_step"
        df.reset_index(inplace=True)
        return df

    return {
        "volume": _make(0, 100),
        "occupancy": _make(0, 1),
        "duration": _make(10, 120),
    }


@pytest.fixture
def sample_unified_base() -> pd.DataFrame:
    """Minimal unified analytical base DataFrame for agent tests."""
    rng = np.random.default_rng(42)
    n = 300
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "hourly_timestamp": timestamps,
        "acn_sessions_count": rng.integers(1, 20, n),
        "acn_total_kwh": rng.uniform(5.0, 200.0, n),
        "acn_base_revenue": rng.uniform(75.0, 3000.0, n),
        "urban_mean_utilization": rng.uniform(0.0, 1.0, n),
        "urban_peak_queue": rng.uniform(0.0, 20.0, n),
        "urban_total_volume": rng.uniform(10.0, 500.0, n),
        "hour_of_day": [t.hour for t in timestamps],
        "day_of_week": [t.dayofweek for t in timestamps],
        "is_weekend": [(1 if t.dayofweek >= 5 else 0) for t in timestamps],
    })


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schema fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_forecast_state() -> ForecastState:
    """A typical ForecastState in the neutral regime."""
    return ForecastState(
        timestamp="2023-01-01 12:00:00",
        u_pred=0.55,
        q_pred=3.2,
        u_actual=0.58,
        q_actual=3.5,
        kwh_delivered=45.0,
        hour_of_day=12,
        is_weekend=0,
    )


@pytest.fixture
def sample_pricing_decision() -> PricingDecision:
    """A typical neutral PricingDecision."""
    return PricingDecision(
        p_new=15.0,
        regime="neutral",
        surge_scalar=0.0,
        discount_scalar=0.0,
        elasticity_used=1.2,
        rationale="Neutral regime — utilisation within normal range.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock Gemini client
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_gemini_always_raises():
    """
    A mock Gemini GenerativeModel whose generate_content() always raises
    an Exception, forcing agents to use their deterministic fallbacks.
    """
    mock = MagicMock()
    mock.generate_content.side_effect = Exception("Gemini unavailable (mock)")
    return mock
