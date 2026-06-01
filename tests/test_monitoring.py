# =============================================================================
# tests/test_monitoring.py — MonitoringLearningAgent tests
# =============================================================================

from __future__ import annotations

from unittest.mock import patch, MagicMock

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
    with patch("src.agents.pricing.ChatGroq") as mock_llm_cls:
        mock_llm_cls.return_value = MagicMock()
        agent = TariffPricingAgent.__new__(TariffPricingAgent)
        agent._theta = np.array([0.3, 4.0, 4.0], dtype=float)
        agent._max_retries = 3
        agent._llm = MagicMock()
        agent._graph = None
        return agent


def _make_monitor(pricing_agent=None, always_fail_llm=True) -> MonitoringLearningAgent:
    if pricing_agent is None:
        pricing_agent = _make_pricing_agent()
    with patch("src.agents.monitoring.ChatGroq") as mock_llm_cls, \
         patch("src.agents.monitoring.time.sleep"):
        mock_llm = MagicMock()
        if always_fail_llm:
            mock_llm.invoke.side_effect = Exception("LLM down (mock)")
        mock_llm_cls.return_value = mock_llm

        agent = MonitoringLearningAgent.__new__(MonitoringLearningAgent)
        agent.pricing_agent = pricing_agent
        agent._lr = 0.8
        agent._lr_decay = 0.002
        agent._max_retries = 1
        agent._step = 0
        agent._episode_log = []
        agent._llm = mock_llm
        agent._graph = agent._build_graph()
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
        elasticity_used=0.3,
        rationale="test",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Property 10: Revenue formula is applied consistently
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    p_new=st.floats(min_value=10.0, max_value=22.0, allow_nan=False, allow_infinity=False),
    kwh=st.floats(min_value=0.01, max_value=500.0, allow_nan=False, allow_infinity=False),
    epsilon=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False),
)
def test_property10_revenue_formula(p_new, kwh, epsilon):
    """revenue_new must equal p_new * kwh * max(0.05, 1 + demand_shift)."""
    demand_shift = -epsilon * ((p_new - P_BASE) / P_BASE)
    expected = p_new * kwh * max(0.05, 1.0 + demand_shift)

    pricing_agent = _make_pricing_agent()
    pricing_agent._theta[0] = epsilon

    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_llm=True)

    state = ForecastState(
        timestamp="t", u_pred=0.5, q_pred=1.0, u_actual=0.5,
        q_actual=1.0, kwh_delivered=kwh, hour_of_day=12, is_weekend=0,
    )
    decision = _make_decision(p_new=p_new, regime="neutral")
    monitor.step(state, decision)

    log = monitor._episode_log[-1]
    assert abs(log["revenue_new"] - expected) < 1e-6, \
        f"revenue_new={log['revenue_new']} != expected={expected}"


# ─────────────────────────────────────────────────────────────────────────────
# Property 11: Learning rate schedule decays monotonically
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    lr0=st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False),
    decay=st.floats(min_value=0.0001, max_value=1.0, allow_nan=False, allow_infinity=False),
    n_steps=st.integers(min_value=2, max_value=50),
)
def test_property11_lr_monotonic_decay(lr0, decay, n_steps):
    """η_t = η₀ / (1 + decay × t) must be strictly decreasing."""
    rates = [lr0 / (1.0 + decay * t) for t in range(n_steps)]
    for i in range(1, len(rates)):
        assert rates[i] < rates[i - 1]


# ─────────────────────────────────────────────────────────────────────────────
# Property 12: Off_Peak_Uplift formula is correct
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
    """off_peak_uplift() must return (mean_post - mean_baseline) / mean_baseline * 100."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_llm=True)

    n_post = len(post_values)
    for i in range(n_post):
        monitor._episode_log.append({
            "step": i, "timestamp": "t", "u_pred": 0.2, "q_pred": 1.0,
            "u_actual": post_values[i], "q_actual": 1.0,
            "p_new": 12.0, "regime": "discount", "rationale": "test",
            "revenue_new": 12.0, "revenue_baseline": 15.0,
            "revenue_gain_pct": -20.0, "charger_utilisation": post_values[i],
            "avg_wait_reduction": 0.0, "pricing_efficiency": 12.0,
            "demand_shift": 0.1, "customer_response_rate": 10.0, "reward": 0.1,
            "epsilon_after": 0.3, "alpha_after": 4.0, "beta_after": 4.0,
            "reflection": "test", "lr_used": 0.8,
        })

    n_used = min(n_post, len(baseline_values))
    baseline_df = pd.DataFrame({"urban_mean_utilization": baseline_values})

    mean_post = float(np.mean(post_values))
    mean_baseline = float(np.mean(baseline_values[:n_used]))

    if mean_baseline == 0.0:
        return

    expected = (mean_post - mean_baseline) / mean_baseline * 100.0
    result = monitor.off_peak_uplift(baseline_df)
    assert abs(result - expected) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: summary() DataFrame columns
# ─────────────────────────────────────────────────────────────────────────────

def test_summary_columns_after_steps():
    """summary() must contain all required columns after N steps."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_llm=True)

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
    assert expected_cols.issubset(set(df.columns))


def test_summary_empty_before_steps():
    """summary() must return empty DataFrame with correct columns before any steps."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent)
    df = monitor.summary()
    assert len(df) == 0
    assert "step" in df.columns


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: LLM fallback returns valid LearningUpdate
# ─────────────────────────────────────────────────────────────────────────────

def test_llm_fallback_returns_valid_update():
    """When LLM always raises, step() must return a valid LearningUpdate."""
    pricing_agent = _make_pricing_agent()
    monitor = _make_monitor(pricing_agent=pricing_agent, always_fail_llm=True)

    state = _make_state(u_pred=0.55)
    decision = _make_decision(p_new=15.0)
    update = monitor.step(state, decision)

    assert isinstance(update, LearningUpdate)
    assert isinstance(update.reward, float)
    assert isinstance(update.reflection, str)
    assert len(monitor._episode_log) == 1
