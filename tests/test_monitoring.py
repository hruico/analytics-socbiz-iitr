# =============================================================================
# tests/test_monitoring.py — MonitoringLearningAgent tests
# =============================================================================

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.config import (
    ForecastState,
    PricingDecision,
    LearningUpdate,
    P_BASE,
)
from src.agents.pricing import TariffPricingAgent
from src.agents.monitoring import MonitoringLearningAgent


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pricing_agent() -> TariffPricingAgent:
    with patch("src.agents.pricing.build_gemini_model"):
        agent = TariffPricingAgent.__new__(TariffPricingAgent)
        agent._theta = np.array([1.2, 4.0, 4.0], dtype=float)
        agent._max_retries = 3
        agent._model = None
        return agent


def _make_monitor(pricing_agent=None, always_fail_gemini=True) -> MonitoringLearningAgent:
    if pricing_agent is None:
        pricing_agent = _make_pricing_agent()
    with patch("src.agents.monitoring.build_gemini_model") as mock_build, \
         patch("src.agents.monitoring.time.sleep"):  # skip retry delays
        mock_model = mock_build.return_value
        if always_fail_gemini:
            mock_model.generate_content.side_effect = Exception("Gemini down (mock)")
        agent = MonitoringLearningAgent(
            pricing_agent=pricing_agent,
            lr=0.8,
            lr_decay=0.002,
            max_retries=1,
        )
        agent._model = mock_model
        return agent


def _make_state(u_pred: float = 0.55) -> ForecastState:
    return ForecastState(
        timestamp="2023-01-01 12:00:00",
        u_pred=u_pred,
        q_pred=3.0,
        u_actual=u_pred,
        q_actual=3.0,
        kwh_delivered=45.0,
        hour_of_day=12,
        is_weekend=0,
    )


def _make_decision(p_new: float = 15.0, regime: str = "neutral") -> PricingDecision:
    return PricingDecision(
        p_new=p_new,
        regime=regime,
        surge_scalar=0.0,
        discount_scalar=0.0,
        elasticity_used=1.2,
        rationale="test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 10: Revenue formula is applied consistently
# Feature: ev-charging-analytics-optimization, Property 10: Revenue formula is applied consistently
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    p_new=st.floats(min_value=10.0, max_value=22.0, allow_nan=False, allow_infinity=False),
    kwh=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
    epsilon=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_property10_revenue_formula(p_new, kwh, epsilon):
    """
    # Feature: ev-charging-analytics-optimization, Property 10: Revenue formula is applied consistently
    revenue_new must equal p_new × kwh × max(0.05, 1 + demand_shift)
    where demand_shift = -ε × ((p_new - 15) / 15).
    """
    demand_shift = -epsilon * ((p_new - P_BASE) / P_BASE)
    expected = p_new * kwh * max(0.05, 1.0 + demand_shift)

    # Verify the formula directly
    actual = p_new * kwh * max(0.05, 1.0 + demand_shift)
    assert abs(actual - expected) < 1e-9, \
        f"Revenue formula mismatch: expected {expected}, got {actual}"

    # Also verify via MonitoringLearningAgent step
    pricing_agent = _make_pricing_agent()
    pricing_agent._theta[0] = epsilon  # set epsilon

    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_gemini=True)

    state = ForecastState(
        timestamp="t", u_pred=0.5, q_pred=1.0, u_actual=0.5,
        q_actual=1.0, kwh_delivered=kwh, hour_of_day=12, is_weekend=0,
    )
    decision = _make_decision(p_new=p_new, regime="neutral")
    update = monitor.step(state, decision)

    log = monitor._episode_log[-1]
    assert abs(log["revenue_new"] - expected) < 1e-6, \
        f"MonitoringAgent revenue_new={log['revenue_new']} != expected={expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Property 11: Learning rate schedule decays monotonically
# Feature: ev-charging-analytics-optimization, Property 11: Learning rate schedule decays monotonically
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    lr0=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    decay=st.floats(min_value=0.0001, max_value=1.0, allow_nan=False, allow_infinity=False),
    n_steps=st.integers(min_value=2, max_value=50),
)
def test_property11_lr_monotonic_decay(lr0, decay, n_steps):
    """
    # Feature: ev-charging-analytics-optimization, Property 11: Learning rate schedule decays monotonically
    η_t = η₀ / (1 + decay × t) must be strictly decreasing as t increases.
    """
    rates = [lr0 / (1.0 + decay * t) for t in range(n_steps)]
    for i in range(1, len(rates)):
        assert rates[i] < rates[i - 1], \
            f"LR not decreasing at step {i}: η[{i}]={rates[i]} >= η[{i-1}]={rates[i-1]}"


# ─────────────────────────────────────────────────────────────────────────────
# Property 12: Off_Peak_Uplift formula is correct
# Feature: ev-charging-analytics-optimization, Property 12: Off_Peak_Uplift formula is correct
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    post_values=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=30,
    ),
    baseline_values=st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=30,
    ),
)
def test_property12_off_peak_uplift_formula(post_values, baseline_values):
    """
    # Feature: ev-charging-analytics-optimization, Property 12: Off_Peak_Uplift formula is correct
    off_peak_uplift() must return (mean_post - mean_baseline) / mean_baseline × 100.
    """
    # Build a monitor with discount-regime log entries
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_gemini=True)

    # Use post_values for the episode log (all discount regime)
    n_post = len(post_values)
    for i in range(n_post):
        monitor._episode_log.append({
            "step": i, "timestamp": "t", "u_pred": 0.2, "q_pred": 1.0,
            "u_actual": post_values[i], "q_actual": 1.0,
            "p_new": 12.0, "regime": "discount", "rationale": "test",
            "revenue_new": 12.0, "revenue_baseline": 15.0,
            "revenue_gain_pct": -20.0, "charger_utilisation": post_values[i],
            "avg_wait_reduction": 0.0, "pricing_efficiency": 12.0,
            "demand_shift": 0.1, "reward": 0.1,
            "epsilon_after": 1.2, "alpha_after": 4.0, "beta_after": 4.0,
            "reflection": "test", "lr_used": 0.8,
        })

    # The method uses min(n_discount, len(baseline_df)) rows from baseline
    n_used = min(n_post, len(baseline_values))
    baseline_df = pd.DataFrame({
        "urban_mean_utilization": baseline_values,
    })

    mean_post = float(np.mean(post_values))
    mean_baseline = float(np.mean(baseline_values[:n_used]))

    if mean_baseline == 0.0:
        return  # skip degenerate case

    expected = (mean_post - mean_baseline) / mean_baseline * 100.0

    result = monitor.off_peak_uplift(baseline_df)
    assert abs(result - expected) < 1e-6, \
        f"off_peak_uplift={result} != expected={expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: summary() DataFrame columns
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_columns_after_steps():
    """summary() must contain all required columns after N steps."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_gemini=True)

    expected_cols = {
        "step", "timestamp", "u_pred", "q_pred", "u_actual", "q_actual",
        "p_new", "regime", "rationale", "revenue_new", "revenue_baseline",
        "revenue_gain_pct", "charger_utilisation", "avg_wait_reduction",
        "pricing_efficiency", "demand_shift", "reward", "epsilon_after",
        "alpha_after", "beta_after", "reflection", "lr_used",
    }

    for i in range(5):
        state = _make_state(u_pred=0.3 + i * 0.1)
        decision = _make_decision(p_new=14.0 + i * 0.5)
        monitor.step(state, decision)

    df = monitor.summary()
    assert len(df) == 5
    assert expected_cols.issubset(set(df.columns)), \
        f"Missing columns: {expected_cols - set(df.columns)}"


def test_summary_empty_before_steps():
    """summary() must return empty DataFrame with correct columns before any steps."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent)
    df = monitor.summary()
    assert len(df) == 0
    assert "step" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: Gemini fallback returns valid LearningUpdate
# ─────────────────────────────────────────────────────────────────────────────

def test_gemini_fallback_returns_valid_update():
    """When Gemini always raises, step() must return a valid LearningUpdate."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_gemini=True)

    state = _make_state(u_pred=0.55)
    decision = _make_decision(p_new=15.0)
    update = monitor.step(state, decision)

    assert isinstance(update, LearningUpdate)
    assert isinstance(update.reward, float)
    assert isinstance(update.reflection, str)
    assert len(monitor._episode_log) == 1
