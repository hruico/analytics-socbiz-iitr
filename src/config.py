# =============================================================================
# src/config.py — OP'26 Agentic EV Tariff Optimisation System
#
# Single source of truth for:
#   • Pricing constants and thresholds
#   • Model hyperparameters
#   • File path constants
#   • Gemini client factory
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
from google import genai
from google.genai import types as genai_types

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
    "n_jobs": -1,
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
    "n_jobs": -1,
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
# GEMINI
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.0-flash"


class _GeminiWrapper:
    """
    Thin wrapper around google.genai Client that exposes a
    generate_content(prompt) method compatible with the agent code.
    """

    def __init__(
        self,
        client: genai.Client,
        model: str,
        system_instruction: str,
        temperature: float,
        response_mime: str,
    ) -> None:
        self._client = client
        self._model = model
        self._system_instruction = system_instruction
        self._temperature = temperature
        self._response_mime = response_mime

    def generate_content(self, prompt: str):  # type: ignore[return]
        config = genai_types.GenerateContentConfig(
            system_instruction=self._system_instruction,
            temperature=self._temperature,
            response_mime_type=self._response_mime,
        )
        return self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )


def build_gemini_model(
    system_instruction: str,
    temperature: float = 0.2,
    response_mime: str = "application/json",
) -> _GeminiWrapper:
    """
    Configure and return a Gemini model wrapper.
    Reads GEMINI_API_KEY exclusively from the environment variable.
    Raises EnvironmentError with the exact export command if the key is missing.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Export it before running:\n"
            "  export GEMINI_API_KEY='your-key-here'"
        )
    client = genai.Client(api_key=api_key)
    return _GeminiWrapper(
        client=client,
        model=GEMINI_MODEL,
        system_instruction=system_instruction,
        temperature=temperature,
        response_mime=response_mime,
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
and the current elasticity parameters (ε, α, β), compute the optimal per-kWh tariff P_t.

Pricing rules:
- Baseline tariff: ₹15/kWh
- If U_pred > 0.80 → surge regime. surge_scalar ∈ (0,1] from distance above threshold.
  P_t = 15 + surge_scalar × (22 − 15). Hard cap: ₹22/kWh.
- If U_pred < 0.30 → discount regime. discount_scalar ∈ (0,1] from distance below threshold.
  P_t = 15 − discount_scalar × (15 − 10). Hard floor: ₹10/kWh.
- Otherwise → neutral regime. P_t near ₹15 with small linear adjustment.
- Never return a price outside [10, 22].

Use α to modulate surge intensity and β for discount depth.
Higher ε means demand is more elastic (price-sensitive).

Return ONLY valid JSON matching exactly this schema — no markdown, no extra keys:
{
  "p_new": <float, ₹/kWh>,
  "regime": <"surge"|"discount"|"neutral">,
  "surge_scalar": <float 0-1>,
  "discount_scalar": <float 0-1>,
  "elasticity_used": <float>,
  "rationale": "<1-2 sentence business justification>"
}
""".strip()


MONITOR_SYSTEM_PROMPT = """
You are the MonitoringLearningAgent in an Agentic EV Charging Optimisation System (OP'26).

Your role:
Evaluate the TariffPricingAgent's last decision against realised operational outcomes.
Propose parameter updates (Δε, Δα, Δβ) to improve future pricing decisions.

Compute:
1. demand_shift = −ε × ((p_new − 15) / 15)
2. revenue_new = p_new × kwh × max(0.05, 1 + demand_shift)
3. revenue_baseline = 15 × kwh
4. revenue_gain_pct = (revenue_new − revenue_baseline) / revenue_baseline × 100
5. charger_utilisation = clip(u_actual + demand_shift × 0.1, 0, 1)
6. avg_wait_reduction = −demand_shift × q_actual  (positive = improvement)
7. pricing_efficiency = revenue_new / kwh
8. reward = 0.5 × tanh(revenue_gain_pct/20) + 0.3 × charger_utilisation − 0.2 × max(0, −avg_wait_reduction)

Parameter update logic:
- Keep all deltas small: |Δε| ≤ 0.05, |Δα| ≤ 0.10, |Δβ| ≤ 0.10

Return ONLY valid JSON matching exactly this schema — no markdown, no extra keys:
{
  "delta_epsilon": <float>,
  "delta_alpha": <float>,
  "delta_beta": <float>,
  "reward": <float>,
  "revenue_gain_pct": <float>,
  "charger_utilisation": <float>,
  "avg_wait_reduction": <float>,
  "pricing_efficiency": <float>,
  "demand_shift": <float>,
  "reflection": "<2-3 sentence learning reflection>"
}
""".strip()
