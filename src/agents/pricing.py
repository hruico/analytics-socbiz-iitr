# =============================================================================
# src/agents/pricing.py — TariffPricingAgent
#
# Gemini-powered tariff agent that translates demand forecasts into optimal
# per-kWh tariffs.  Falls back to a deterministic sigmoid-based formula when
# the Gemini API is unavailable or returns invalid JSON.
#
# Parameter vector Θ = [ε, α, β]:
#   ε — price elasticity (demand sensitivity)
#   α — surge pricing sensitivity
#   β — discount pricing sensitivity
# =============================================================================

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.config import (
    build_gemini_model,
    TARIFF_SYSTEM_PROMPT,
    P_BASE,
    P_SURGE_CAP,
    P_DISCOUNT_FLOOR,
    SURGE_THRESHOLD,
    DISCOUNT_THRESHOLD,
    ForecastState,
    PricingDecision,
)

logger = logging.getLogger(__name__)


def _parse_retry_delay(exc: Exception, default: float = 5.0) -> float:
    """
    Extract the retry_delay seconds from a Gemini 429 ResourceExhausted error.
    Falls back to *default* if the hint is not present.
    """
    try:
        msg = str(exc)
        import re
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
        if m:
            return float(m.group(1)) + 2.0  # add 2s buffer
    except Exception:
        pass
    return default


class TariffPricingAgent:
    """
    Gemini-powered tariff agent.

    Translates a ``ForecastState`` (predicted utilisation, queue, kWh, etc.)
    into a ``PricingDecision`` (regime, p_new, scalars, rationale).

    Gemini is called up to ``max_retries`` times with exponential backoff.
    If all retries are exhausted, a deterministic sigmoid-based fallback is
    used and an ERROR is logged.

    The mutable parameter vector Θ = [ε, α, β] is updated via
    ``apply_update(delta)`` and each component is clipped to its defined
    bounds after every update.

    Parameters
    ----------
    epsilon_init : float
        Initial price-elasticity parameter ε.  Default 1.2.
    alpha_init : float
        Initial surge-sensitivity parameter α.  Default 4.0.
    beta_init : float
        Initial discount-sensitivity parameter β.  Default 4.0.
    max_retries : int
        Maximum number of Gemini API attempts before falling back.  Default 3.
    """

    # Bounds for each component of Θ
    _THETA_BOUNDS: dict[str, tuple[float, float]] = {
        "epsilon": (0.1, 5.0),
        "alpha": (1.0, 10.0),
        "beta": (1.0, 10.0),
    }

    def __init__(
        self,
        epsilon_init: float = 1.2,
        alpha_init: float = 4.0,
        beta_init: float = 4.0,
        max_retries: int = 3,
    ) -> None:
        # ── Θ = [ε, α, β] stored as a numpy array ────────────────────────────
        self._theta: np.ndarray = np.array(
            [epsilon_init, alpha_init, beta_init], dtype=float
        )
        self._max_retries: int = max_retries

        # ── Build Gemini model ────────────────────────────────────────────────
        self._model = build_gemini_model(TARIFF_SYSTEM_PROMPT)
        logger.info(
            "TariffPricingAgent initialised: ε=%.3f  α=%.3f  β=%.3f  max_retries=%d",
            self.epsilon,
            self.alpha,
            self.beta,
            self._max_retries,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """Price-elasticity parameter ε."""
        return float(self._theta[0])

    @property
    def alpha(self) -> float:
        """Surge-sensitivity parameter α."""
        return float(self._theta[1])

    @property
    def beta(self) -> float:
        """Discount-sensitivity parameter β."""
        return float(self._theta[2])

    # ── Core public interface ─────────────────────────────────────────────────

    def compute_tariff(self, state: ForecastState) -> PricingDecision:
        """
        Compute the optimal tariff for the given forecast state.

        Attempts to call Gemini up to ``max_retries`` times with exponential
        backoff (``1.5 × attempt`` seconds between retries).  If all attempts
        fail, falls back to the deterministic sigmoid formula and logs ERROR.

        After receiving a Gemini response, regime/price consistency is enforced:
        - If regime is "surge" but p_new ≤ P_BASE → override with fallback
        - If regime is "discount" but p_new ≥ P_BASE → override with fallback

        ``p_new`` is always clipped to [P_DISCOUNT_FLOOR, P_SURGE_CAP].

        Requirements: 5.1, 5.2, 5.3, 5.4, 5.6
        """
        prompt = self._build_prompt(state)

        for attempt in range(self._max_retries):
            try:
                response = self._model.generate_content(prompt)
                raw_text = response.text.strip()

                # Strip markdown code fences if present
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    raw_text = "\n".join(
                        line for line in lines
                        if not line.startswith("```")
                    ).strip()

                data: dict[str, Any] = json.loads(raw_text)
                decision = PricingDecision(**data)

                # ── Enforce regime/price consistency ─────────────────────────
                decision = self._enforce_consistency(state, decision)
                return decision

            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                logger.warning(
                    "TariffPricingAgent: Gemini attempt %d/%d failed — %s: %s",
                    attempt + 1,
                    self._max_retries,
                    type(exc).__name__,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    # Honour retry_delay hint from 429 responses
                    wait = _parse_retry_delay(exc, default=1.5 * (attempt + 1))
                    time.sleep(wait)

        # ── All retries exhausted — use deterministic fallback ────────────────
        logger.error(
            "TariffPricingAgent: all %d Gemini retries exhausted for state "
            "u_pred=%.4f; using deterministic fallback.",
            self._max_retries,
            state.u_pred,
        )
        return self._deterministic_fallback(state)

    def apply_update(self, delta: np.ndarray) -> None:
        """
        Update Θ by adding *delta* and clip each component to its bounds.

        Parameters
        ----------
        delta : np.ndarray
            Shape (3,) array ``[Δε, Δα, Δβ]``.

        Requirements: 5.5
        """
        self._theta = self._theta + delta
        # Clip each component to its defined bounds
        eps_lo, eps_hi = self._THETA_BOUNDS["epsilon"]
        alp_lo, alp_hi = self._THETA_BOUNDS["alpha"]
        bet_lo, bet_hi = self._THETA_BOUNDS["beta"]
        self._theta[0] = float(np.clip(self._theta[0], eps_lo, eps_hi))
        self._theta[1] = float(np.clip(self._theta[1], alp_lo, alp_hi))
        self._theta[2] = float(np.clip(self._theta[2], bet_lo, bet_hi))
        logger.debug(
            "TariffPricingAgent: θ updated → ε=%.4f  α=%.4f  β=%.4f",
            self.epsilon,
            self.alpha,
            self.beta,
        )

    def run_sensitivity_analysis(
        self,
        test_df: pd.DataFrame,
        epsilon_values: list[float],
    ) -> pd.DataFrame:
        """
        Sweep over *epsilon_values*, running the deterministic fallback on
        every row of *test_df* for each ε, and return a summary DataFrame.

        The original ε is restored after the sweep.

        Parameters
        ----------
        test_df : pd.DataFrame
            DataFrame whose rows are used as forecast states.  Must contain
            at least ``u_pred`` and ``kwh_delivered`` columns (or equivalent
            columns used by the fallback).
        epsilon_values : list[float]
            ε values to test, e.g. ``[0.5, 1.0, 1.5, 2.0]``.

        Returns
        -------
        pd.DataFrame
            Columns: ``epsilon``, ``mean_revenue_gain_pct``,
            ``std_revenue_gain_pct``, ``min_revenue_gain_pct``,
            ``max_revenue_gain_pct``.

        Requirements: 8.6
        """
        original_epsilon = float(self._theta[0])
        rows: list[dict[str, float]] = []

        for eps in epsilon_values:
            # Temporarily override ε
            self._theta[0] = float(eps)

            revenue_gains: list[float] = []
            for _, row in test_df.iterrows():
                u_pred = float(row.get("u_pred", row.get("urban_mean_utilization", 0.5)))
                kwh = float(row.get("kwh_delivered", row.get("acn_total_kwh", 10.0)))
                kwh = max(0.01, kwh)
                p_new = self._fallback_price(u_pred)
                demand_shift = -eps * ((p_new - P_BASE) / P_BASE)
                revenue_new = p_new * kwh * max(0.05, 1.0 + demand_shift)
                revenue_baseline = P_BASE * kwh
                revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100.0
                revenue_gains.append(revenue_gain_pct)

            gains_arr = np.array(revenue_gains)
            rows.append(
                {
                    "epsilon": float(eps),
                    "mean_revenue_gain_pct": float(np.mean(gains_arr)),
                    "std_revenue_gain_pct": float(np.std(gains_arr)),
                    "min_revenue_gain_pct": float(np.min(gains_arr)),
                    "max_revenue_gain_pct": float(np.max(gains_arr)),
                }
            )

        # Restore original ε
        self._theta[0] = original_epsilon

        return pd.DataFrame(
            rows,
            columns=[
                "epsilon",
                "mean_revenue_gain_pct",
                "std_revenue_gain_pct",
                "min_revenue_gain_pct",
                "max_revenue_gain_pct",
            ],
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_prompt(self, state: ForecastState) -> str:
        """Construct the Gemini prompt from the current state and Θ."""
        return (
            f"Forecast state:\n"
            f"  u_pred={state.u_pred:.4f}  q_pred={state.q_pred:.4f}\n"
            f"  hour_of_day={state.hour_of_day}  is_weekend={state.is_weekend}\n"
            f"  kwh_delivered={state.kwh_delivered:.4f}\n"
            f"\nCurrent parameters:\n"
            f"  epsilon={self.epsilon:.4f}  alpha={self.alpha:.4f}  beta={self.beta:.4f}\n"
            f"\nCompute the optimal tariff and return JSON."
        )

    def _fallback_price(self, u_pred: float) -> float:
        """
        Compute the deterministic fallback price for a given *u_pred*.

        Uses the current α and β from Θ.  Returns p_new clipped to
        [P_DISCOUNT_FLOOR, P_SURGE_CAP].
        """
        if u_pred > SURGE_THRESHOLD:
            # Surge regime
            surge_scalar = float(np.clip((u_pred - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 0.0, 1.0))
            p_new = P_BASE + surge_scalar * self.alpha * (P_SURGE_CAP - P_BASE) / 10.0
        elif u_pred < DISCOUNT_THRESHOLD:
            # Discount regime
            discount_scalar = float(np.clip((DISCOUNT_THRESHOLD - u_pred) / DISCOUNT_THRESHOLD, 0.0, 1.0))
            p_new = P_BASE - discount_scalar * self.beta * (P_BASE - P_DISCOUNT_FLOOR) / 10.0
        else:
            # Neutral regime
            p_new = P_BASE + (u_pred - 0.55) * 2.0

        return float(np.clip(p_new, P_DISCOUNT_FLOOR, P_SURGE_CAP))

    def _deterministic_fallback(self, state: ForecastState) -> PricingDecision:
        """
        Deterministic sigmoid-based fallback pricing decision.

        Computes regime, scalars, and p_new from the current Θ without
        calling Gemini.  p_new is always clipped to [P_DISCOUNT_FLOOR, P_SURGE_CAP].
        """
        u_pred = state.u_pred

        if u_pred > SURGE_THRESHOLD:
            regime = "surge"
            surge_scalar = float(np.clip(
                (u_pred - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 0.0, 1.0
            ))
            discount_scalar = 0.0
            p_new = P_BASE + surge_scalar * self.alpha * (P_SURGE_CAP - P_BASE) / 10.0

        elif u_pred < DISCOUNT_THRESHOLD:
            regime = "discount"
            surge_scalar = 0.0
            discount_scalar = float(np.clip(
                (DISCOUNT_THRESHOLD - u_pred) / DISCOUNT_THRESHOLD, 0.0, 1.0
            ))
            p_new = P_BASE - discount_scalar * self.beta * (P_BASE - P_DISCOUNT_FLOOR) / 10.0

        else:
            regime = "neutral"
            surge_scalar = 0.0
            discount_scalar = 0.0
            p_new = P_BASE + (u_pred - 0.55) * 2.0

        p_new = float(np.clip(p_new, P_DISCOUNT_FLOOR, P_SURGE_CAP))

        return PricingDecision(
            p_new=p_new,
            regime=regime,  # type: ignore[arg-type]
            surge_scalar=surge_scalar,
            discount_scalar=discount_scalar,
            elasticity_used=self.epsilon,
            rationale=(
                f"Deterministic fallback: u_pred={u_pred:.4f} → "
                f"regime={regime}, p_new=₹{p_new:.2f}/kWh "
                f"(ε={self.epsilon:.3f}, α={self.alpha:.3f}, β={self.beta:.3f})"
            ),
        )

    def _enforce_consistency(
        self,
        state: ForecastState,
        decision: PricingDecision,
    ) -> PricingDecision:
        """
        Enforce regime/price consistency on a Gemini-returned decision.

        Rules:
        - If regime is "surge" but p_new ≤ P_BASE → override with fallback
        - If regime is "discount" but p_new ≥ P_BASE → override with fallback
        - p_new is always clipped to [P_DISCOUNT_FLOOR, P_SURGE_CAP] (handled
          by the PricingDecision Pydantic validator, but we double-check here)

        Returns the (possibly overridden) decision.
        """
        u_pred = state.u_pred
        override = False

        if decision.regime == "surge" and decision.p_new <= P_BASE:
            logger.warning(
                "TariffPricingAgent: Gemini returned regime='surge' but "
                "p_new=%.2f ≤ P_BASE=%.2f — overriding with fallback.",
                decision.p_new,
                P_BASE,
            )
            override = True

        elif decision.regime == "discount" and decision.p_new >= P_BASE:
            logger.warning(
                "TariffPricingAgent: Gemini returned regime='discount' but "
                "p_new=%.2f ≥ P_BASE=%.2f — overriding with fallback.",
                decision.p_new,
                P_BASE,
            )
            override = True

        if override:
            return self._deterministic_fallback(state)

        return decision
