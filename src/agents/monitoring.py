# =============================================================================
# src/agents/monitoring.py — MonitoringLearningAgent
#
# Gemini-powered monitoring and learning agent that evaluates pricing decisions
# against realised operational outcomes and proposes parameter updates Δθ.
#
# Computes all OP'26 metrics: demand_shift, revenue_new, revenue_gain_pct,
# charger_utilisation, avg_wait_reduction, pricing_efficiency, reward.
#
# Scales Gemini-proposed Δθ by learning rate η = η₀ / (1 + decay × t) before
# calling pricing_agent.apply_update().
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
    MONITOR_SYSTEM_PROMPT,
    P_BASE,
    ForecastState,
    PricingDecision,
    LearningUpdate,
)
from src.agents.pricing import TariffPricingAgent

logger = logging.getLogger(__name__)


def _parse_retry_delay(exc: Exception, default: float = 5.0) -> float:
    """Extract retry_delay seconds from a Gemini 429 error, with fallback.
    Handles both new SDK JSON format ('retryDelay': '41s') and old proto format.
    """
    try:
        import re
        msg = str(exc)
        # New SDK format: 'retryDelay': '41s'
        m = re.search(r"['\"]retryDelay['\"]:\s*['\"](\d+(?:\.\d+)?)s['\"]", msg)
        if m:
            return float(m.group(1)) + 2.0
        # Old proto format: retry_delay { seconds: 41 }
        m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", msg)
        if m:
            return float(m.group(1)) + 2.0
    except Exception:
        pass
    return default


class MonitoringLearningAgent:
    """
    Gemini-powered monitoring and learning agent.

    Evaluates each pricing decision against realised outcomes, computes all
    OP'26 metrics, and proposes parameter updates Δθ = [Δε, Δα, Δβ].

    Gemini is called up to ``max_retries`` times with exponential backoff.
    If all retries are exhausted, a deterministic economic fallback is used
    and an ERROR is logged.

    The proposed Δθ is scaled by the current learning rate
    η = η₀ / (1 + decay × t) before being applied to the pricing agent.

    Parameters
    ----------
    pricing_agent : TariffPricingAgent
        Reference to the pricing agent whose parameters will be updated.
    lr : float
        Initial learning rate η₀.  Default 0.8.
    lr_decay : float
        Learning rate decay coefficient.  Default 0.002.
    max_retries : int
        Maximum number of Gemini API attempts before falling back.  Default 3.
    """

    def __init__(
        self,
        pricing_agent: TariffPricingAgent,
        lr: float = 0.8,
        lr_decay: float = 0.002,
        max_retries: int = 3,
    ) -> None:
        self.pricing_agent = pricing_agent
        self._lr: float = lr
        self._lr_decay: float = lr_decay
        self._max_retries: int = max_retries
        self._step: int = 0
        self._episode_log: list[dict[str, Any]] = []

        # Build Gemini model
        self._model = build_gemini_model(MONITOR_SYSTEM_PROMPT)
        logger.info(
            "MonitoringLearningAgent initialised: lr=%.3f  decay=%.5f  max_retries=%d",
            self._lr,
            self._lr_decay,
            self._max_retries,
        )

    def step(
        self,
        state: ForecastState,
        decision: PricingDecision,
    ) -> LearningUpdate:
        """
        Evaluate the pricing decision against realised outcomes and update θ.

        Computes all OP'26 metrics deterministically, then calls Gemini to
        propose Δθ. Scales Δθ by η = η₀ / (1 + decay × t) before applying.

        Appends all metrics plus θ state and lr_used to the episode log.

        NOTE — Assumptions & Limitations:
        • demand_shift is a model-based elasticity proxy
          (−ε × Δp/p_base), NOT an observed causal effect.
          It approximates how demand responds to price changes
          based on the assumed elasticity parameter ε.
        • charger_utilisation is a post-pricing estimate derived
          from u_actual and demand_shift, not a direct measurement.
        • avg_wait_reduction is inferred from demand_shift × q_actual;
          no ground-truth wait-time data is available.
        • All revenue figures assume the demand model is correct;
          actual revenue depends on real user behaviour.

        Requirements: 6.1, 6.2, 6.3
        """
        # ── 1. Compute metrics deterministically ─────────────────────────────
        epsilon = self.pricing_agent.epsilon
        demand_shift = -epsilon * ((decision.p_new - P_BASE) / P_BASE)
        revenue_new = decision.p_new * state.kwh_delivered * max(0.05, 1.0 + demand_shift)
        revenue_baseline = P_BASE * state.kwh_delivered
        revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100.0
        charger_utilisation = float(np.clip(state.u_actual + demand_shift * 0.1, 0.0, 1.0))
        avg_wait_reduction = -demand_shift * state.q_actual
        pricing_efficiency = revenue_new / state.kwh_delivered
        reward = (
            0.5 * float(np.tanh(revenue_gain_pct / 20.0))
            + 0.3 * charger_utilisation
            - 0.2 * max(0.0, -avg_wait_reduction)
        )

        # ── 2. Try Gemini for LearningUpdate ──────────────────────────────────
        update = self._get_learning_update(
            state, decision, demand_shift, revenue_gain_pct,
            charger_utilisation, avg_wait_reduction, pricing_efficiency, reward
        )

        # ── 3. Compute learning rate ──────────────────────────────────────────
        eta = self._lr / (1.0 + self._lr_decay * self._step)

        # ── 4. Scale deltas and apply update ──────────────────────────────────
        delta = np.array([
            update.delta_epsilon,
            update.delta_alpha,
            update.delta_beta,
        ]) * eta
        self.pricing_agent.apply_update(delta)

        # ── 5. Append to episode log ──────────────────────────────────────────
        self._episode_log.append({
            "step": self._step,
            "timestamp": state.timestamp,
            "u_pred": state.u_pred,
            "q_pred": state.q_pred,
            "u_actual": state.u_actual,
            "q_actual": state.q_actual,
            "p_new": decision.p_new,
            "regime": decision.regime,
            "rationale": decision.rationale,
            "revenue_new": revenue_new,
            "revenue_baseline": revenue_baseline,
            "revenue_gain_pct": revenue_gain_pct,
            "charger_utilisation": charger_utilisation,
            "avg_wait_reduction": avg_wait_reduction,
            "pricing_efficiency": pricing_efficiency,
            "demand_shift": demand_shift,
            # Customer Response Rate: % shift in session volume due to tariff change
            # Positive = demand increased (discount worked), Negative = demand fell (surge effect)
            "customer_response_rate": demand_shift * 100.0,
            "reward": reward,
            "epsilon_after": self.pricing_agent.epsilon,
            "alpha_after": self.pricing_agent.alpha,
            "beta_after": self.pricing_agent.beta,
            "reflection": update.reflection,
            "lr_used": eta,
        })

        # ── 6. Increment step counter ─────────────────────────────────────────
        self._step += 1

        return update

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per episode step containing all logged
        metrics.

        Returns
        -------
        pd.DataFrame
            Columns: step, timestamp, u_pred, q_pred, u_actual, q_actual,
            p_new, regime, rationale, revenue_new, revenue_baseline,
            revenue_gain_pct, charger_utilisation, avg_wait_reduction,
            pricing_efficiency, demand_shift, reward, epsilon_after,
            alpha_after, beta_after, reflection, lr_used.

        Requirements: 6.5
        """
        if not self._episode_log:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                "step", "timestamp", "u_pred", "q_pred", "u_actual", "q_actual",
                "p_new", "regime", "rationale", "revenue_new", "revenue_baseline",
                "revenue_gain_pct", "charger_utilisation", "avg_wait_reduction",
                "pricing_efficiency", "demand_shift", "customer_response_rate",
                "reward", "epsilon_after", "alpha_after", "beta_after",
                "reflection", "lr_used",
            ])
        return pd.DataFrame(self._episode_log)

    def off_peak_uplift(self, baseline_df: pd.DataFrame) -> float:
        """
        Compute Off_Peak_Uplift: percentage change in mean session count during
        discount-regime hours compared to pre-optimisation baseline.

        Formula: (mean_post − mean_baseline) / mean_baseline × 100

        Parameters
        ----------
        baseline_df : pd.DataFrame
            Baseline data with ``urban_mean_utilization`` column.

        Returns
        -------
        float
            Off_Peak_Uplift percentage. Returns 0.0 if no discount-regime hours
            are present in the episode log.

        Requirements: 6.6, 8.4
        """
        if not self._episode_log:
            return 0.0

        # Filter episode log to discount-regime hours
        discount_rows = [
            row for row in self._episode_log if row["regime"] == "discount"
        ]
        if not discount_rows:
            return 0.0

        # Mean utilisation in post-optimisation discount hours
        mean_post = float(np.mean([row["u_actual"] for row in discount_rows]))

        # Mean utilisation in baseline (use same number of rows from start)
        n_discount = len(discount_rows)
        if len(baseline_df) < n_discount:
            logger.warning(
                "off_peak_uplift: baseline_df has fewer rows (%d) than discount "
                "hours (%d) — using all available baseline rows.",
                len(baseline_df),
                n_discount,
            )
            n_discount = len(baseline_df)

        mean_baseline = float(baseline_df["urban_mean_utilization"].iloc[:n_discount].mean())

        if mean_baseline == 0.0:
            logger.warning("off_peak_uplift: mean_baseline is zero — returning 0.0")
            return 0.0

        uplift = (mean_post - mean_baseline) / mean_baseline * 100.0
        return float(uplift)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_learning_update(
        self,
        state: ForecastState,
        decision: PricingDecision,
        demand_shift: float,
        revenue_gain_pct: float,
        charger_utilisation: float,
        avg_wait_reduction: float,
        pricing_efficiency: float,
        reward: float,
    ) -> LearningUpdate:
        """
        Try Gemini up to max_retries times to get a LearningUpdate.
        Falls back to deterministic economic logic if all retries fail.

        Requirements: 6.4
        """
        prompt = self._build_prompt(
            state, decision, demand_shift, revenue_gain_pct,
            charger_utilisation, avg_wait_reduction, pricing_efficiency, reward
        )

        for attempt in range(self._max_retries):
            try:
                response = self._model.generate_content(prompt)
                raw_text = response.text.strip()

                # Strip markdown code fences if present
                if raw_text.startswith("```"):
                    lines = raw_text.splitlines()
                    raw_text = "\n".join(
                        line for line in lines if not line.startswith("```")
                    ).strip()

                data: dict[str, Any] = json.loads(raw_text)
                update = LearningUpdate(**data)
                return update

            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                logger.warning(
                    "MonitoringLearningAgent: Gemini attempt %d/%d failed — %s: %s",
                    attempt + 1,
                    self._max_retries,
                    type(exc).__name__,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    # For 400 errors (bad JSON), retry immediately
                    # For 429 rate limit errors, honour the retry-after hint
                    wait = _parse_retry_delay(exc, default=2.0)
                    logger.info("MonitoringLearningAgent: waiting %.1fs before retry …", wait)
                    time.sleep(wait)

        # ── All retries exhausted — use deterministic fallback ────────────────
        logger.error(
            "MonitoringLearningAgent: all %d Gemini retries exhausted; "
            "using deterministic fallback.",
            self._max_retries,
        )
        return self._deterministic_fallback(
            demand_shift, revenue_gain_pct, charger_utilisation,
            avg_wait_reduction, pricing_efficiency, reward
        )

    def _build_prompt(
        self,
        state: ForecastState,
        decision: PricingDecision,
        demand_shift: float,
        revenue_gain_pct: float,
        charger_utilisation: float,
        avg_wait_reduction: float,
        pricing_efficiency: float,
        reward: float,
    ) -> str:
        """Construct the Gemini prompt from state, decision, and metrics."""
        return (
            f"Forecast state:\n"
            f"  u_pred={state.u_pred:.4f}  q_pred={state.q_pred:.4f}\n"
            f"  u_actual={state.u_actual:.4f}  q_actual={state.q_actual:.4f}\n"
            f"  kwh_delivered={state.kwh_delivered:.4f}\n"
            f"\nPricing decision:\n"
            f"  p_new=₹{decision.p_new:.2f}/kWh  regime={decision.regime}\n"
            f"  rationale: {decision.rationale}\n"
            f"\nComputed metrics:\n"
            f"  demand_shift={demand_shift:.4f}\n"
            f"  revenue_gain_pct={revenue_gain_pct:.2f}%\n"
            f"  charger_utilisation={charger_utilisation:.4f}\n"
            f"  avg_wait_reduction={avg_wait_reduction:.4f}\n"
            f"  pricing_efficiency=₹{pricing_efficiency:.2f}/kWh\n"
            f"  reward={reward:.4f}\n"
            f"\nCurrent parameters:\n"
            f"  epsilon={self.pricing_agent.epsilon:.4f}\n"
            f"  alpha={self.pricing_agent.alpha:.4f}\n"
            f"  beta={self.pricing_agent.beta:.4f}\n"
            f"\nPropose parameter updates (Δε, Δα, Δβ) and return JSON."
        )

    def _deterministic_fallback(
        self,
        demand_shift: float,
        revenue_gain_pct: float,
        charger_utilisation: float,
        avg_wait_reduction: float,
        pricing_efficiency: float,
        reward: float,
    ) -> LearningUpdate:
        """
        Deterministic economic fallback for LearningUpdate when Gemini fails.

        Simple heuristic:
        - Increase ε if revenue gain is positive, else decrease
        - Increase α if utilisation is high (> 0.7), else decrease
        - Increase β if utilisation is low (< 0.3), else decrease

        Requirements: 6.4
        """
        delta_epsilon = 0.01 if revenue_gain_pct > 0.0 else -0.01
        delta_alpha = 0.02 if charger_utilisation > 0.7 else -0.02
        delta_beta = 0.02 if charger_utilisation < 0.3 else -0.02

        return LearningUpdate(
            delta_epsilon=delta_epsilon,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            reward=reward,
            revenue_gain_pct=revenue_gain_pct,
            charger_utilisation=charger_utilisation,
            avg_wait_reduction=avg_wait_reduction,
            pricing_efficiency=pricing_efficiency,
            demand_shift=demand_shift,
            reflection="Deterministic fallback: Gemini unavailable.",
        )
