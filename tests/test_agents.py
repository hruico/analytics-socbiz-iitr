# =============================================================================
# tests/test_agents.py — DemandPredictionAgent and TariffPricingAgent tests
# =============================================================================

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.config import (
    ForecastState,
    PricingDecision,
    P_BASE,
    P_SURGE_CAP,
    P_DISCOUNT_FLOOR,
    SURGE_THRESHOLD,
    DISCOUNT_THRESHOLD,
    engineer_features,
    FEATURE_COLS,
    TRAIN_RATIO,
)
from src.agents.pricing import TariffPricingAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_unified_csv(path: str, n: int = 300, seed: int = 42) -> None:
    """Write a minimal unified_analytical_base.csv to *path*."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({
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
    df.to_csv(path, index=False)


def _make_pricing_agent() -> TariffPricingAgent:
    """Create a TariffPricingAgent with a mocked ChatGroq."""
    with patch("src.agents.pricing.ChatGroq") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()
        agent = TariffPricingAgent.__new__(TariffPricingAgent)
        agent._theta = np.array([0.3, 4.0, 4.0], dtype=float)
        agent._max_retries = 3
        agent._llm = MagicMock()
        agent._graph = None
        return agent


# ─────────────────────────────────────────────────────────────────────────────
# Property 7: ForecastState output satisfies schema bounds
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    u_raw=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False),
    kwh_raw=st.floats(min_value=-100.0, max_value=500.0, allow_nan=False, allow_infinity=False),
)
def test_property7_forecast_state_bounds(u_raw, kwh_raw):
    """For any raw values, ForecastState must have u_pred in [0,1] and kwh_delivered >= 0.01."""
    u_clipped = float(np.clip(u_raw, 0.0, 1.0))
    kwh_floored = max(0.01, float(kwh_raw)) if kwh_raw > 0 else 0.01

    state = ForecastState(
        timestamp="2023-01-01 00:00:00",
        u_pred=u_raw,
        q_pred=abs(u_raw),
        u_actual=u_clipped,
        q_actual=0.0,
        kwh_delivered=kwh_floored,
        hour_of_day=12,
        is_weekend=0,
    )
    assert 0.0 <= state.u_pred <= 1.0
    assert state.kwh_delivered >= 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Property 8: Pricing regime and price bounds are consistent
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(u_pred=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_property8_pricing_regime_bounds(u_pred):
    """For any u_pred in [0,1], the deterministic fallback must return consistent regime/price."""
    agent = _make_pricing_agent()

    state = ForecastState(
        timestamp="t", u_pred=u_pred, q_pred=1.0,
        u_actual=u_pred, q_actual=1.0, kwh_delivered=10.0,
        hour_of_day=12, is_weekend=0,
    )
    decision = agent._deterministic_fallback(state)

    assert P_DISCOUNT_FLOOR <= decision.p_new <= P_SURGE_CAP

    if u_pred > SURGE_THRESHOLD:
        assert decision.regime == "surge"
        assert decision.p_new > P_BASE
    elif u_pred < DISCOUNT_THRESHOLD:
        assert decision.regime == "discount"
        assert decision.p_new < P_BASE
    else:
        assert decision.regime == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Property 9: Theta parameters remain within bounds after any update
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    d_eps=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    d_alpha=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    d_beta=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
)
def test_property9_theta_bounds_after_update(d_eps, d_alpha, d_beta):
    """For any delta vector, theta must stay within defined bounds after apply_update()."""
    agent = _make_pricing_agent()
    delta = np.array([d_eps, d_alpha, d_beta])
    agent.apply_update(delta)

    assert 0.1 <= agent.epsilon <= 5.0
    assert 1.0 <= agent.alpha <= 10.0
    assert 1.0 <= agent.beta <= 10.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: LLM failure falls back to deterministic decision
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_fallback_returns_valid_decision(monkeypatch):
    """When LLM always raises, compute_tariff() must return a valid PricingDecision."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    with patch("src.agents.pricing.ChatGroq") as mock_llm_cls, \
         patch("src.agents.pricing.time.sleep"):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")
        mock_llm_cls.return_value = mock_llm

        agent = TariffPricingAgent(epsilon_init=0.3, alpha_init=4.0, beta_init=4.0, max_retries=2)

        for u_pred in [0.1, 0.5, 0.9]:
            state = ForecastState(
                timestamp="t", u_pred=u_pred, q_pred=1.0,
                u_actual=u_pred, q_actual=1.0, kwh_delivered=10.0,
                hour_of_day=12, is_weekend=0,
            )
            decision = agent.compute_tariff(state)
            assert isinstance(decision, PricingDecision)
            assert P_DISCOUNT_FLOOR <= decision.p_new <= P_SURGE_CAP
            assert decision.elasticity_used > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: elasticity_used is always positive
# ─────────────────────────────────────────────────────────────────────────────

def test_elasticity_used_always_positive(monkeypatch):
    """elasticity_used must always be > 0 regardless of LLM output."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")

    with patch("src.agents.pricing.ChatGroq") as mock_llm_cls, \
         patch("src.agents.pricing.time.sleep"):
        mock_llm = MagicMock()
        # Simulate LLM returning elasticity_used=0.0 (the old bug)
        mock_llm.invoke.return_value.content = (
            '{"p_new": 20.0, "regime": "surge", "surge_scalar": 0.8, '
            '"discount_scalar": 0.0, "elasticity_used": 0.0, "rationale": "test"}'
        )
        mock_llm_cls.return_value = mock_llm

        agent = TariffPricingAgent(epsilon_init=0.3, alpha_init=4.0, beta_init=4.0, max_retries=1)

        state = ForecastState(
            timestamp="t", u_pred=0.9, q_pred=5.0,
            u_actual=0.9, q_actual=5.0, kwh_delivered=20.0,
            hour_of_day=14, is_weekend=0,
        )
        decision = agent.compute_tariff(state)
        # The agent must override elasticity_used=0.0 with current epsilon
        assert decision.elasticity_used > 0.0, \
            f"elasticity_used={decision.elasticity_used} should be > 0"


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: DemandPredictionAgent evaluation_metrics shape
# ─────────────────────────────────────────────────────────────────────────────

def test_demand_agent_evaluation_metrics_shape(monkeypatch):
    """evaluation_metrics() must return dict with both targets and RMSE/MAE/R2 keys."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr("src.agents.demand.XGB_PARAMS", {
        "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3,
        "n_jobs": 1, "verbosity": 0, "tree_method": "hist",
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "unified.csv")
        _make_unified_csv(csv_path, n=300)

        from src.agents.demand import DemandPredictionAgent
        agent = DemandPredictionAgent(csv_path=csv_path, use_lightgbm=False)
        metrics = agent.evaluation_metrics()

        assert "urban_mean_utilization" in metrics
        assert "urban_peak_queue" in metrics
        for target in ["urban_mean_utilization", "urban_peak_queue"]:
            assert "RMSE" in metrics[target]
            assert "MAE" in metrics[target]
            assert "R2" in metrics[target]
            assert isinstance(metrics[target]["RMSE"], float)


def test_demand_agent_compare_backends_shape(monkeypatch):
    """compare_backends() must return DataFrame with expected columns when LightGBM enabled."""
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setattr("src.agents.demand.XGB_PARAMS", {
        "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3,
        "n_jobs": 1, "verbosity": 0, "tree_method": "hist",
    })
    monkeypatch.setattr("src.agents.demand.LGB_PARAMS", {
        "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3,
        "n_jobs": 1, "verbose": -1,
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "unified.csv")
        _make_unified_csv(csv_path, n=300)

        from src.agents.demand import DemandPredictionAgent
        agent = DemandPredictionAgent(csv_path=csv_path, use_lightgbm=True)
        df = agent.compare_backends()

        expected_cols = {"target", "xgb_rmse", "xgb_mae", "xgb_r2",
                         "lgb_rmse", "lgb_mae", "lgb_r2"}
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: sensitivity analysis schema
# ─────────────────────────────────────────────────────────────────────────────

def test_sensitivity_analysis_schema():
    """run_sensitivity_analysis() must return DataFrame with correct columns."""
    agent = _make_pricing_agent()
    rng = np.random.default_rng(42)
    test_df = pd.DataFrame({
        "urban_mean_utilization": rng.uniform(0.0, 1.0, 50),
    })
    result = agent.run_sensitivity_analysis(test_df, epsilon_values=[0.5, 1.0, 1.5, 2.0])

    expected_cols = {
        "epsilon", "mean_revenue_gain_pct", "std_revenue_gain_pct",
        "min_revenue_gain_pct", "max_revenue_gain_pct",
    }
    assert expected_cols.issubset(set(result.columns))
    assert len(result) == 4
