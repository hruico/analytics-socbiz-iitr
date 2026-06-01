# =============================================================================
# tests/test_schemas.py — Pydantic schema and config tests
# =============================================================================

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from pydantic import ValidationError

from src.config import (
    ForecastState,
    PricingDecision,
    LearningUpdate,
    P_BASE,
    P_SURGE_CAP,
    P_DISCOUNT_FLOOR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Property 15: Pydantic schemas enforce field bounds
# Feature: ev-charging-analytics-optimization, Property 15: Pydantic schemas enforce field bounds
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(u_pred=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
def test_property15_forecast_state_u_pred_clipped(u_pred):
    """
    # Feature: ev-charging-analytics-optimization, Property 15: Pydantic schemas enforce field bounds
    ForecastState.u_pred must be clipped to [0, 1] by the validator.
    """
    state = ForecastState(
        timestamp="2023-01-01 00:00:00",
        u_pred=u_pred,
        q_pred=1.0,
        u_actual=0.5,
        q_actual=1.0,
        kwh_delivered=10.0,
        hour_of_day=12,
        is_weekend=0,
    )
    assert 0.0 <= state.u_pred <= 1.0, \
        f"u_pred={state.u_pred} is outside [0, 1] for input {u_pred}"


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(p_new=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_property15_pricing_decision_p_new_clipped(p_new):
    """
    # Feature: ev-charging-analytics-optimization, Property 15: Pydantic schemas enforce field bounds
    PricingDecision.p_new must be clipped to [P_DISCOUNT_FLOOR, P_SURGE_CAP].
    """
    # Determine a valid regime for the clipped price
    clipped = float(np.clip(p_new, P_DISCOUNT_FLOOR, P_SURGE_CAP))
    if clipped > P_BASE:
        regime = "surge"
    elif clipped < P_BASE:
        regime = "discount"
    else:
        regime = "neutral"

    decision = PricingDecision(
        p_new=p_new,
        regime=regime,
        surge_scalar=0.0,
        discount_scalar=0.0,
        elasticity_used=1.0,
        rationale="test",
    )
    assert P_DISCOUNT_FLOOR <= decision.p_new <= P_SURGE_CAP, \
        f"p_new={decision.p_new} is outside [{P_DISCOUNT_FLOOR}, {P_SURGE_CAP}]"


def test_property15_forecast_state_kwh_must_be_positive():
    """
    ForecastState.kwh_delivered must be > 0; zero or negative must raise ValidationError.
    """
    with pytest.raises(ValidationError):
        ForecastState(
            timestamp="2023-01-01 00:00:00",
            u_pred=0.5,
            q_pred=1.0,
            u_actual=0.5,
            q_actual=1.0,
            kwh_delivered=0.0,   # invalid
            hour_of_day=12,
            is_weekend=0,
        )

    with pytest.raises(ValidationError):
        ForecastState(
            timestamp="2023-01-01 00:00:00",
            u_pred=0.5,
            q_pred=1.0,
            u_actual=0.5,
            q_actual=1.0,
            kwh_delivered=-5.0,  # invalid
            hour_of_day=12,
            is_weekend=0,
        )


def test_property15_forecast_state_hour_bounds():
    """hour_of_day must be in [0, 23]."""
    with pytest.raises(ValidationError):
        ForecastState(
            timestamp="t", u_pred=0.5, q_pred=0.0, u_actual=0.5,
            q_actual=0.0, kwh_delivered=1.0, hour_of_day=24, is_weekend=0,
        )
    with pytest.raises(ValidationError):
        ForecastState(
            timestamp="t", u_pred=0.5, q_pred=0.0, u_actual=0.5,
            q_actual=0.0, kwh_delivered=1.0, hour_of_day=-1, is_weekend=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: build_gemini_model() raises EnvironmentError without API key
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Unit test: GROQ_API_KEY check in agents
# ─────────────────────────────────────────────────────────────────────────────

def test_pricing_agent_no_api_key_raises(monkeypatch):
    """
    TariffPricingAgent must raise EnvironmentError when GROQ_API_KEY is unset.
    """
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    from src.agents.pricing import TariffPricingAgent
    with pytest.raises(EnvironmentError, match="GROQ_API_KEY"):
        TariffPricingAgent()

# ─────────────────────────────────────────────────────────────────────────────
# Unit test: LearningUpdate schema
# ─────────────────────────────────────────────────────────────────────────────

def test_learning_update_valid():
    """LearningUpdate must accept valid field values."""
    update = LearningUpdate(
        delta_epsilon=0.01,
        delta_alpha=0.02,
        delta_beta=-0.01,
        reward=0.5,
        revenue_gain_pct=5.0,
        charger_utilisation=0.7,
        avg_wait_reduction=1.2,
        pricing_efficiency=16.0,
        demand_shift=-0.05,
        reflection="Test reflection.",
    )
    assert update.reward == 0.5
    assert update.reflection == "Test reflection."
