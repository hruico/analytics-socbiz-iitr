"""Monitoring agent — evaluates pricing outcomes and proposes parameter updates via LLM."""
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LearningUpdate(BaseModel):
    delta_epsilon: float
    delta_alpha: float
    delta_beta: float
    reward: float
    revenue_gain_pct: float
    reflection: str


class StepMetrics(BaseModel):
    step: int
    regime: str
    revenue_gain_pct: float
    u_actual: float
    u_pred: float


class MonitoringAgent:
    """
    Evaluates each pricing step and proposes updates to theta = [epsilon, alpha, beta].

    On each call the LLM receives recent history and current outcome, then suggests
    parameter deltas with a written reflection. If the LLM is unavailable a lightweight
    rule-based fallback applies regime-specific adjustments.
    """

    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        self.llm_success_count = 0
        self.fallback_count = 0
        self.last_epsilon_reduction_step = -999

    def evaluate_and_propose(
        self,
        step: int,
        revenue_gain_pct: float,
        u_actual: float,
        u_pred: float,
        regime: str,
        recent_history: List[StepMetrics],
        current_theta: np.ndarray,
    ) -> LearningUpdate:
        """Try LLM parameter update; fall back to deterministic rules if unavailable."""
        reward = revenue_gain_pct

        if self.llm is not None:
            update = self._llm_parameter_update(
                step, revenue_gain_pct, u_actual, u_pred,
                regime, recent_history, current_theta, reward
            )
            if update is not None:
                self.llm_success_count += 1
                return update

        self.fallback_count += 1
        return self._deterministic_fallback(
            step, revenue_gain_pct, u_actual, regime,
            recent_history, reward, current_theta[0]
        )

    def _llm_parameter_update(
        self,
        step: int,
        revenue_gain_pct: float,
        u_actual: float,
        u_pred: float,
        regime: str,
        recent_history: List[StepMetrics],
        current_theta: np.ndarray,
        reward: float,
    ) -> Optional[LearningUpdate]:
        """Stateless LLM call — fresh context each step, no conversation history."""
        if len(recent_history) >= 3:
            rr = [m.revenue_gain_pct for m in recent_history[-3:]]
            ru = [m.u_actual for m in recent_history[-3:]]
            history_summary = (
                f"Last 3 steps: revenue=[{rr[0]:+.1f}%, {rr[1]:+.1f}%, {rr[2]:+.1f}%], "
                f"utilization=[{ru[0]:.1%}, {ru[1]:.1%}, {ru[2]:.1%}]"
            )
        else:
            history_summary = "Insufficient history (< 3 steps)"

        epsilon, alpha, beta = current_theta

        prompt = f"""You are optimizing EV charging tariff parameters. Evaluate this pricing step and propose adjustments.

OUTCOME:
- Revenue gain: {revenue_gain_pct:+.2f}%
- Actual utilization: {u_actual:.1%}
- Predicted utilization: {u_pred:.1%}
- Regime: {regime}
- Step: {step}

PARAMETERS:
- ε={epsilon:.3f}  [demand elasticity, range: 0.1–5.0]
- α={alpha:.3f}  [surge multiplier, range: 1.0–10.0]
- β={beta:.3f}  [discount multiplier, range: 1.0–10.0]

RECENT HISTORY:
{history_summary}

ADJUSTMENT GUIDELINES:
- ε: adjust based on neutral-regime revenue trends
- α: adjust based on surge-regime revenue extraction
- β: adjust based on discount-regime demand uplift

CONSTRAINTS: |Δε| ≤ 0.05, |Δα| ≤ 0.10, |Δβ| ≤ 0.10

Respond with JSON only:
{{"delta_epsilon": <float>, "delta_alpha": <float>, "delta_beta": <float>, "reflection": "<reasoning>"}}"""

        try:
            response = self.llm.invoke_with_retry(prompt, response_format="json")
            if response is None:
                return None

            def safe_float(val, default=0.0):
                if val is None:
                    return default
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, list) and val:
                    return float(val[0])
                if isinstance(val, str):
                    return float(val)
                return default

            delta_eps   = np.clip(safe_float(response.get("delta_epsilon")), -0.05, 0.05)
            delta_alpha = np.clip(safe_float(response.get("delta_alpha")),   -0.10, 0.10)
            delta_beta  = np.clip(safe_float(response.get("delta_beta")),    -0.10, 0.10)

            logger.info(f"LLM monitoring response: Δε={delta_eps}, Δα={delta_alpha}, Δβ={delta_beta}")

            # Sanity checks — correct directionally wrong proposals
            if len(recent_history) >= 3:
                recent_revenue = [m.revenue_gain_pct for m in recent_history[-3:]]
                recent_util    = [m.u_actual          for m in recent_history[-3:]]
                if all(r < 0 for r in recent_revenue) and delta_eps > 0:
                    delta_eps = 0.0
                if all(u > 0.80 for u in recent_util) and regime == "surge" and delta_alpha < 0:
                    delta_alpha = 0.0

            return LearningUpdate(
                delta_epsilon=delta_eps,
                delta_alpha=delta_alpha,
                delta_beta=delta_beta,
                reward=reward,
                revenue_gain_pct=revenue_gain_pct,
                reflection=response.get("reflection", "LLM parameter adjustment"),
            )

        except Exception as e:
            logger.warning(f"LLM monitoring failed: {e}")
            return None

    def _deterministic_fallback(
        self,
        step: int,
        revenue_gain_pct: float,
        u_actual: float,
        regime: str,
        recent_history: List[StepMetrics],
        reward: float,
        current_epsilon: float,
    ) -> LearningUpdate:
        """Regime-aware rule-based parameter updates when LLM is unavailable."""
        delta_eps = delta_alpha = delta_beta = 0.0

        if len(recent_history) >= 3:
            recent_revenue  = [m.revenue_gain_pct for m in recent_history[-3:]]
            recent_regimes  = [m.regime           for m in recent_history[-3:]]

            if regime == "neutral":
                cooldown_passed = (step - self.last_epsilon_reduction_step) >= 5
                if all(r < 0 for r in recent_revenue) and cooldown_passed:
                    delta_eps = -0.02
                    self.last_epsilon_reduction_step = step
                elif all(r > 2.0 for r in recent_revenue):
                    delta_eps = 0.01

            if sum(1 for r in recent_regimes if r == "surge") >= 1 and all(r < 5.0 for r in recent_revenue):
                delta_alpha = 0.05

            if sum(1 for r in recent_regimes if r == "discount") >= 1 and all(r < 0 for r in recent_revenue):
                delta_beta = 0.05

        return LearningUpdate(
            delta_epsilon=delta_eps,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            reward=reward,
            revenue_gain_pct=revenue_gain_pct,
            reflection="Deterministic fallback adjustment",
        )

    def get_stats(self) -> dict:
        total = self.llm_success_count + self.fallback_count
        llm_rate = (self.llm_success_count / total * 100) if total > 0 else 0
        return {
            "llm_success": self.llm_success_count,
            "fallback_used": self.fallback_count,
            "llm_success_rate": llm_rate,
        }
