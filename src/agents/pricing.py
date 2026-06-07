"""Pricing agent — LLM classifies regime, deterministic formula sets price."""
from typing import Literal, Optional
from pydantic import BaseModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PricingDecision(BaseModel):
    p_new: float
    regime: Literal["surge", "neutral", "discount"]
    surge_scalar: float = 0.0
    discount_scalar: float = 0.0
    elasticity_used: float
    rationale: str
    fallback_used: bool = False


class PricingAgent:
    """
    Determines dynamic tariffs via LLM-guided regime classification.

    The LLM decides which pricing regime applies (surge / neutral / discount).
    Prices are then computed deterministically from learned parameters (theta),
    keeping the math auditable and the LLM focused on strategy.

    Thresholds are calibrated to the UrbanEV data distribution:
      surge_threshold  = P75 of utilization (~0.48)
      discount_threshold = P25 of utilization (~0.36)
    """

    def __init__(
        self,
        baseline: float,
        bounds: tuple,
        theta: np.ndarray,
        llm_provider=None,
        surge_threshold: float = 0.48,
        discount_threshold: float = 0.36,
    ):
        self.baseline = baseline
        self.lower, self.upper = bounds
        self.theta = theta          # [epsilon, alpha, beta]
        self.llm = llm_provider
        self.surge_threshold = surge_threshold
        self.discount_threshold = discount_threshold
        self.neutral_mid = (surge_threshold + discount_threshold) / 2.0
        self.llm_success_count = 0
        self.fallback_count = 0

    def compute_tariff(
        self,
        u_pred: float,
        q_pred: float,
        hour: int,
        is_weekend: bool,
        congestion_prob: float,
    ) -> PricingDecision:
        """Try LLM first, fall back to deterministic rule if LLM unavailable."""
        epsilon, alpha, beta = self.theta

        if self.llm is not None:
            decision = self._llm_pricing_decision(
                u_pred, q_pred, hour, is_weekend, congestion_prob, epsilon, alpha, beta
            )
            if decision is not None:
                self.llm_success_count += 1
                return decision

        self.fallback_count += 1
        return self._deterministic_fallback(u_pred, q_pred, epsilon, alpha, beta)

    def _compute_price_from_regime(
        self, regime: str, u_pred: float, epsilon: float, alpha: float, beta: float
    ) -> float:
        """
        Pricing formulas per regime:
          surge:   p = baseline * (1 + alpha * excess_above_threshold)
          discount: p = baseline - deficit * beta
          neutral:  p = baseline * (1 + 0.10 * position_in_band)
        """
        if regime == "surge":
            p_new = self.baseline * (1 + alpha * (u_pred - self.surge_threshold))
        elif regime == "discount":
            p_new = self.baseline - (self.discount_threshold - u_pred) * beta
        else:
            band = self.surge_threshold - self.discount_threshold
            position = (u_pred - self.discount_threshold) / band
            p_new = self.baseline * (1 + 0.10 * position)
        return np.clip(p_new, self.lower, self.upper)

    def _llm_pricing_decision(
        self,
        u_pred: float,
        q_pred: float,
        hour: int,
        is_weekend: bool,
        congestion_prob: float,
        epsilon: float,
        alpha: float,
        beta: float,
    ) -> Optional[PricingDecision]:
        """
        Fresh stateless LLM call per step — no conversation history.
        LLM outputs regime label; price is computed here from that label + theta.
        Near-threshold decisions are overridden by the rule to avoid boundary noise.
        """
        time_context = "weekend" if is_weekend else "weekday"

        prompt = f"""You are an EV charging pricing strategist. Classify the optimal pricing regime based on demand forecast.

DEMAND FORECAST:
- Predicted Utilization: {u_pred:.1%}
- Queue Length: {q_pred:.1f} vehicles
- Congestion Probability: {congestion_prob:.1%}
- Time: {hour}:00 ({time_context})

REGIME CLASSIFICATION RULES:
- IF utilization > {self.surge_threshold:.0%}: regime = "surge"
- IF utilization < {self.discount_threshold:.0%}: regime = "discount"
- IF {self.discount_threshold:.0%} <= utilization <= {self.surge_threshold:.0%}: regime = "neutral"

RESPOND WITH JSON ONLY:
{{"regime": "<surge|neutral|discount>", "rationale": "<your_strategic_reasoning>"}}"""

        try:
            response = self.llm.invoke_with_retry(prompt, response_format="json")
            if response is None:
                return None

            regime = response.get("regime", "neutral")
            rationale = response.get("rationale", "LLM regime classification")

            logger.info(f"LLM regime suggestion: {regime} at u_pred={u_pred:.1%}")

            # Determine rule-based regime for confidence check
            if u_pred > self.surge_threshold:
                rule_regime = "surge"
                threshold = self.surge_threshold
            elif u_pred < self.discount_threshold:
                rule_regime = "discount"
                threshold = self.discount_threshold
            else:
                rule_regime = "neutral"
                threshold = self.neutral_mid

            # Override LLM when predicted utilization is close to a threshold boundary
            confidence = abs(u_pred - threshold) / max(threshold, 1 - threshold)
            if confidence > 0.15:
                final_regime = regime
            else:
                final_regime = rule_regime
                if regime != rule_regime:
                    logger.info(f"Low confidence ({confidence:.2f}) - overriding LLM {regime} → {rule_regime}")

            logger.info(f"Final regime: {final_regime} (u_pred={u_pred:.1%})")
            p_new = self._compute_price_from_regime(final_regime, u_pred, epsilon, alpha, beta)

            if final_regime == "surge":
                surge_scalar = (u_pred - self.surge_threshold) / (1.0 - self.surge_threshold)
                discount_scalar = 0.0
            elif final_regime == "discount":
                surge_scalar = 0.0
                discount_scalar = (self.discount_threshold - u_pred) / self.discount_threshold
            else:
                surge_scalar = discount_scalar = 0.0

            logger.info(f"Computed price: ₹{p_new:.2f} for regime={final_regime}")

            return PricingDecision(
                p_new=p_new,
                regime=final_regime,
                surge_scalar=surge_scalar,
                discount_scalar=discount_scalar,
                elasticity_used=epsilon,
                rationale=rationale,
                fallback_used=False,
            )

        except Exception as e:
            logger.warning(f"LLM pricing failed: {e}")
            return None

    def _deterministic_fallback(
        self, u_pred: float, q_pred: float, epsilon: float, alpha: float, beta: float
    ) -> PricingDecision:
        """Rule-based fallback using the same pricing formulas as the LLM path."""
        if u_pred > self.surge_threshold:
            regime = "surge"
            surge_scalar = (u_pred - self.surge_threshold) / (1.0 - self.surge_threshold)
            discount_scalar = 0.0
            p_new = self.baseline * (1 + alpha * (u_pred - self.surge_threshold))
            rationale = f"Surge: u={u_pred:.2%} > threshold={self.surge_threshold:.0%}"
        elif u_pred < self.discount_threshold:
            regime = "discount"
            surge_scalar = 0.0
            discount_scalar = (self.discount_threshold - u_pred) / self.discount_threshold
            p_new = self.baseline - (self.discount_threshold - u_pred) * beta
            rationale = f"Discount: u={u_pred:.2%} < threshold={self.discount_threshold:.0%}"
        else:
            regime = "neutral"
            surge_scalar = discount_scalar = 0.0
            band = self.surge_threshold - self.discount_threshold
            position = (u_pred - self.discount_threshold) / band
            p_new = self.baseline * (1 + 0.10 * position)
            rationale = f"Neutral: u={u_pred:.2%} in [{self.discount_threshold:.0%}, {self.surge_threshold:.0%}]"

        p_new = np.clip(p_new, self.lower, self.upper)
        return PricingDecision(
            p_new=p_new,
            regime=regime,
            surge_scalar=surge_scalar,
            discount_scalar=discount_scalar,
            elasticity_used=epsilon,
            rationale=rationale,
            fallback_used=True,
        )

    def apply_update(self, delta: np.ndarray) -> None:
        """Apply parameter update from monitoring agent, clipped to valid ranges."""
        self.theta += delta
        self.theta[0] = np.clip(self.theta[0], 0.1, 5.0)
        self.theta[1] = np.clip(self.theta[1], 1.0, 10.0)
        self.theta[2] = np.clip(self.theta[2], 1.0, 10.0)
        logger.debug(f"Theta updated: {self.theta}, delta={delta}")

    def get_stats(self) -> dict:
        total = self.llm_success_count + self.fallback_count
        llm_rate = (self.llm_success_count / total * 100) if total > 0 else 0
        return {
            "llm_success": self.llm_success_count,
            "fallback_used": self.fallback_count,
            "llm_success_rate": llm_rate,
        }
