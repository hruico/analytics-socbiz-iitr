# =============================================================================
# src/config.py — OP'26 Agentic EV Tariff Optimisation System
#
# Single source of truth for:
#   • Pricing constants and thresholds
#   • Model hyperparameters
#   • File path constants
#   • LLM client factory (Groq — free tier, OpenAI-compatible)
#   • Pydantic I/O schemas
#   • Shared feature engineering function
# =============================================================================

from __future__ import annotations

import os
import logging
from typing import List, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# PRICING CONSTANTS  (OP'26 spec)
# ─────────────────────────────────────────────────────────────────────────────
P_BASE: float = 15.0            # ₹/kWh — fixed-rate baseline
P_SURGE_CAP: float = 22.0       # ₹/kWh — maximum surge ceiling
P_DISCOUNT_FLOOR: float = 10.0  # ₹/kWh — minimum discount floor
SURGE_THRESHOLD: float = 0.80   # utilisation above which surge is triggered
DISCOUNT_THRESHOLD: float = 0.30  # utilisation below which discount is triggered

# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42
TRAIN_RATIO: float = 0.80

# ─────────────────────────────────────────────────────────────────────────────
# ELASTICITY — kept low to reflect inelastic EV charging demand
# EV charging is largely necessity-driven; empirical studies suggest
# short-run price elasticity of -0.1 to -0.3 for workplace/captive chargers.
# ─────────────────────────────────────────────────────────────────────────────
EPSILON_INIT: float = 0.3   # conservative inelastic default

# ─────────────────────────────────────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
XGB_PARAMS: dict = {
    "n_estimators": 600,
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "n_jobs": 1,          # 1 avoids deadlock inside MultiOutputRegressor
    "verbosity": 0,
    "tree_method": "hist",
}

LGB_PARAMS: dict = {
    "n_estimators": 600,
    "learning_rate": 0.04,
    "max_depth": 6,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "n_jobs": 1,          # 1 avoids deadlock inside MultiOutputRegressor
    "verbose": -1,
}

# ─────────────────────────────────────────────────────────────────────────────
# FILE PATHS  (relative to project root)
# ─────────────────────────────────────────────────────────────────────────────
RAW_ACN_PATH: str = "data/raw/acndata_sessions.json.xlsx"
RAW_URBAN_DIR: str = "data/raw"
PROCESSED_BASE_PATH: str = "data/processed/unified_analytical_base.csv"
OUTPUTS_DIR: str = "outputs"
EDA_OUTPUTS_DIR: str = "outputs/eda"

# ─────────────────────────────────────────────────────────────────────────────
# LLM — Groq (free tier, OpenAI-compatible, ~14 400 RPD on llama-3.3-70b)
# Agents use langchain-groq ChatGroq directly. GROQ_MODEL is the shared
# model name constant used by both agents.
# ─────────────────────────────────────────────────────────────────────────────
GROQ_MODEL: str = "llama-3.3-70b-versatile"


def _load_api_keys() -> list[str]:
    """Load all available Groq API keys from environment variables."""
    keys: list[str] = []
    primary = os.environ.get("GROQ_API_KEY", "").strip()
    if primary:
        keys.append(primary)
    for i in range(2, 11):
        k = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def build_gemini_model(system_instruction: str, **kwargs):  # type: ignore[return]
    """
    Kept for backward compatibility with any legacy references.
    Raises EnvironmentError if GROQ_API_KEY is not set.
    Agents should use ChatGroq directly.
    """
    keys = _load_api_keys()
    if not keys:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Export it before running:\n"
            "  export GROQ_API_KEY='your-groq-key-here'\n"
            "Get a free key at: https://console.groq.com"
        )
    raise NotImplementedError(
        "build_gemini_model is deprecated. Agents now use langchain-groq ChatGroq directly."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS  — agent I/O contracts
# ─────────────────────────────────────────────────────────────────────────────

class ForecastState(BaseModel):
    """Output of DemandPredictionAgent → input to TariffPricingAgent."""
    timestamp: str
    u_pred: float = Field(..., ge=0.0, le=1.0, description="Predicted utilisation ∈ [0,1]")
    q_pred: float = Field(..., ge=0.0, description="Predicted queue length proxy")
    u_actual: float = Field(..., ge=0.0, le=1.0)
    q_actual: float = Field(..., ge=0.0)
    kwh_delivered: float = Field(..., gt=0.0)
    hour_of_day: int = Field(..., ge=0, le=23)
    is_weekend: int = Field(..., ge=0, le=1)

    @field_validator("u_pred", "u_actual", mode="before")
    @classmethod
    def clip_utilisation(cls, v: float) -> float:
        return float(np.clip(v, 0.0, 1.0))

    @field_validator("q_pred", "q_actual", mode="before")
    @classmethod
    def floor_queue(cls, v: float) -> float:
        return float(max(0.0, v))


class PricingDecision(BaseModel):
    """JSON schema Gemini TariffPricingAgent must return."""
    p_new: float = Field(..., ge=P_DISCOUNT_FLOOR, le=P_SURGE_CAP,
                         description="Optimised tariff in ₹/kWh")
    regime: Literal["surge", "discount", "neutral"]
    surge_scalar: float = Field(..., ge=0.0, le=1.0)
    discount_scalar: float = Field(..., ge=0.0, le=1.0)
    elasticity_used: float = Field(..., gt=0.0)
    rationale: str = Field(..., description="Gemini's pricing rationale")

    @field_validator("p_new", mode="before")
    @classmethod
    def clip_price(cls, v: float) -> float:
        return float(np.clip(v, P_DISCOUNT_FLOOR, P_SURGE_CAP))


class LearningUpdate(BaseModel):
    """JSON schema Gemini MonitoringLearningAgent must return."""
    delta_epsilon: float = Field(..., description="Proposed Δε adjustment")
    delta_alpha: float = Field(..., description="Proposed Δα adjustment")
    delta_beta: float = Field(..., description="Proposed Δβ adjustment")
    reward: float = Field(..., description="Scalar reward for this step")
    revenue_gain_pct: float
    charger_utilisation: float
    avg_wait_reduction: float
    pricing_efficiency: float = Field(..., description="₹ per kWh delivered")
    demand_shift: float
    reflection: str = Field(..., description="Gemini's learning reflection")


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING  — single source of truth
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: List[str] = [
    "acn_sessions_count", "acn_total_kwh",
    "urban_total_volume", "hour_of_day", "hour_sin", "hour_cos",
    "day_of_week", "dow_sin", "dow_cos", "is_weekend",
    "util_lag1", "util_lag2", "util_lag3", "util_lag24",
    "queue_lag1", "queue_lag2", "queue_lag3", "queue_lag24",
    "util_roll6_mean", "util_roll6_std", "queue_roll6_mean",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Causal, leakage-free feature engineering applied to the unified base CSV.

    All lag/rolling operations use .shift(1) as the anchor so no feature
    at row i references data from row i or later (no look-ahead bias).
    dropna() is called only after ALL features are computed.

    Used by both DemandPredictionAgent and the EDA feature importance plot.
    """
    df = df.copy()

    # Cyclical time encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Causal lag features — anchored at shift(1)
    for lag in [1, 2, 3, 24]:
        df[f"util_lag{lag}"] = df["urban_mean_utilization"].shift(lag)
        df[f"queue_lag{lag}"] = df["urban_peak_queue"].shift(lag)

    # Rolling statistics — shift(1) ensures no leakage
    df["util_roll6_mean"] = (
        df["urban_mean_utilization"].shift(1).rolling(6, min_periods=1).mean()
    )
    df["util_roll6_std"] = (
        df["urban_mean_utilization"].shift(1).rolling(6, min_periods=1).std().fillna(0)
    )
    df["queue_roll6_mean"] = (
        df["urban_peak_queue"].shift(1).rolling(6, min_periods=1).mean()
    )

    # Drop NaN rows only after all features are computed
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────
TARIFF_SYSTEM_PROMPT = """
You are the TariffPricingAgent in an Agentic EV Charging Optimisation System (OP'26).

Your role:
Given a forecast state (predicted utilisation U_pred, queue Q_pred, hour, weekend flag)
and the current elasticity parameters (epsilon, alpha, beta), compute the optimal per-kWh tariff P_t.

Pricing rules:
- Baseline tariff: 15.0 Rs/kWh
- If U_pred > 0.80: surge regime. surge_scalar = (U_pred - 0.80) / (1.0 - 0.80), clipped to [0, 1]. P_t = 15 + surge_scalar * (22 - 15). Hard cap: 22.0 Rs/kWh.
- If U_pred < 0.30: discount regime. discount_scalar = (0.30 - U_pred) / 0.30, clipped to [0, 1]. P_t = 15 - discount_scalar * (15 - 10). Hard floor: 10.0 Rs/kWh.
- Otherwise: neutral regime. P_t = 15 + (U_pred - 0.55) * 2.0, clipped to [10, 22].
- p_new must always be a plain decimal number between 10.0 and 22.0.

CRITICAL RULES for the JSON output:
- Every value must be a plain number — no expressions, no formulas, no arithmetic operators.
- surge_scalar and discount_scalar must be pre-computed decimal numbers between 0.0 and 1.0.
- Do NOT write things like "0.9878 * 3.5239 / (3.5239 + 1)" — compute it first and write the result.

Return ONLY valid JSON with exactly these keys — no markdown, no extra keys, no comments:
{
  "p_new": <decimal number between 10.0 and 22.0>,
  "regime": <"surge" or "discount" or "neutral">,
  "surge_scalar": <decimal number between 0.0 and 1.0>,
  "discount_scalar": <decimal number between 0.0 and 1.0>,
  "elasticity_used": <decimal number>,
  "rationale": "<1-2 sentence justification>"
}
""".strip()


MONITOR_SYSTEM_PROMPT = """
You are the MonitoringLearningAgent in an Agentic EV Charging Optimisation System (OP'26).

Your role:
Evaluate the TariffPricingAgent's last decision against realised operational outcomes.
Propose parameter updates (delta_epsilon, delta_alpha, delta_beta) to improve future pricing decisions.

Use these formulas to compute the metrics (all values are provided to you):
1. demand_shift = -epsilon * ((p_new - 15) / 15)
2. revenue_new = p_new * kwh * max(0.05, 1 + demand_shift)
3. revenue_baseline = 15 * kwh
4. revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100
5. charger_utilisation = clip(u_actual + demand_shift * 0.1, 0, 1)
6. avg_wait_reduction = -demand_shift * q_actual
7. pricing_efficiency = revenue_new / kwh
8. reward = 0.5 * tanh(revenue_gain_pct/20) + 0.3 * charger_utilisation - 0.2 * max(0, -avg_wait_reduction)

Parameter update rules:
- Keep all deltas small: |delta_epsilon| <= 0.05, |delta_alpha| <= 0.10, |delta_beta| <= 0.10

CRITICAL RULES for the JSON output:
- Every value must be a plain decimal number — no expressions, no formulas, no arithmetic operators.
- Compute all values first, then write the results as plain numbers.
- Do NOT write things like "0.5 * tanh(...)" — compute it and write the result.

Return ONLY valid JSON with exactly these keys — no markdown, no extra keys, no comments:
{
  "delta_epsilon": <decimal number>,
  "delta_alpha": <decimal number>,
  "delta_beta": <decimal number>,
  "reward": <decimal number>,
  "revenue_gain_pct": <decimal number>,
  "charger_utilisation": <decimal number between 0 and 1>,
  "avg_wait_reduction": <decimal number>,
  "pricing_efficiency": <decimal number>,
  "demand_shift": <decimal number>,
  "reflection": "<2-3 sentence learning reflection>"
}
""".strip()
