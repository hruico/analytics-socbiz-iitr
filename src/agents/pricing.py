"""Pricing agent with LLM reasoning and robust fallback."""
from typing import Literal, Optional
from pydantic import BaseModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PricingDecision(BaseModel):
    """Pricing decision output."""
    p_new: float
    regime: Literal["surge", "neutral", "discount"]
    surge_scalar: float = 0.0
    discount_scalar: float = 0.0
    elasticity_used: float
    rationale: str
    fallback_used: bool = False


class PricingAgent:
    """Determines optimal tariffs using LLM reasoning with fallback."""
    
    def __init__(self, baseline: float, bounds: tuple, theta: np.ndarray, llm_provider=None):
        self.baseline = baseline
        self.lower, self.upper = bounds
        self.theta = theta  # [epsilon, alpha, beta]
        self.llm = llm_provider
        self.llm_success_count = 0
        self.fallback_count = 0
    
    def compute_tariff(self, u_pred: float, q_pred: float, hour: int, 
                      is_weekend: bool, congestion_prob: float) -> PricingDecision:
        """
        Compute optimal tariff using LLM reasoning with fallback.
        
        First tries LLM, falls back to deterministic if LLM fails.
        """
        epsilon, alpha, beta = self.theta
        
        # Try LLM reasoning first
        if self.llm is not None:
            decision = self._llm_pricing_decision(u_pred, q_pred, hour, is_weekend, 
                                                 congestion_prob, epsilon, alpha, beta)
            if decision is not None:
                self.llm_success_count += 1
                return decision
        
        # Fallback to deterministic
        self.fallback_count += 1
        return self._deterministic_fallback(u_pred, q_pred, epsilon, alpha, beta)
    
    def _compute_price_from_regime(self, regime: str, u_pred: float, 
                                   epsilon: float, alpha: float, beta: float) -> float:
        """
        Deterministically compute price from regime classification and theta parameters.
        LLM decides strategy, code enforces the math.
        """
        if regime == "surge":
            # FIX 1: Multiplicative surge pricing for meaningful premium
            # baseline * (1 + alpha * excess) ensures surge prices in ₹18-22 range
            # Example: 15 * (1 + 2.5 * 0.045) = ₹16.69 minimum, scaling with alpha learning
            p_new = self.baseline * (1 + alpha * (u_pred - 0.80))
        elif regime == "discount":
            # Discount pricing: baseline - utilization deficit × beta × epsilon
            p_new = self.baseline - (0.30 - u_pred) * beta * epsilon
        else:  # neutral
            # FIX 3: Re-centred neutral pricing for revenue neutrality
            # Centres at baseline when u=55%, discounts below, premiums above
            # This prevents systematic revenue loss from demand destruction
            p_new = self.baseline + (u_pred - 0.55) * 8.0
        
        return np.clip(p_new, self.lower, self.upper)
    
    def _llm_pricing_decision(self, u_pred: float, q_pred: float, hour: int,
                             is_weekend: bool, congestion_prob: float,
                             epsilon: float, alpha: float, beta: float) -> Optional[PricingDecision]:
        """
        Use LLM to classify regime only. Price computed deterministically.
        FIX 1: Stateless - no history, fresh prompt each call.
        FIX 2: LLM outputs regime, code computes price.
        """
        time_context = "weekend" if is_weekend else "weekday"
        
        # FIX 1: Build fresh prompt each time - NO HISTORY
        prompt = f"""You are an EV charging pricing strategist. Classify the optimal pricing regime based on demand forecast.

DEMAND FORECAST:
- Predicted Utilization: {u_pred:.1%}
- Queue Length: {q_pred:.1f} vehicles
- Congestion Probability: {congestion_prob:.1%}
- Time: {hour}:00 ({time_context})

REGIME CLASSIFICATION RULES:
- IF utilization > 80%: regime = "surge" (high demand, maximize revenue)
- IF utilization < 30%: regime = "discount" (low demand, attract customers)
- IF 30% ≤ utilization ≤ 80%: regime = "neutral" (balanced pricing)

YOUR TASK:
Classify the regime and explain your reasoning. DO NOT calculate prices - that's handled separately.

RESPOND WITH JSON ONLY:
{{"regime": "<surge|neutral|discount>", "rationale": "<your_strategic_reasoning>"}}"""
        
        try:
            # FIX 1: Invoke LLM with NO conversation history
            response = self.llm.invoke_with_retry(prompt, response_format="json")
            
            if response is None:
                return None
            
            # Extract regime classification
            regime = response.get("regime", "neutral")
            rationale = response.get("rationale", "LLM regime classification")
            
            logger.info(f"LLM regime suggestion: {regime} at u_pred={u_pred:.1%}")
            
            # HARD BOUNDARY ENFORCEMENT - thresholds are system rules, not LLM judgment calls
            # LLM decision is advisory; mathematical thresholds are law
            if u_pred > 0.80:
                if regime != "surge":
                    logger.warning(f"Overriding LLM {regime} → surge (u_pred={u_pred:.1%} > 80%)")
                regime = "surge"
            elif u_pred < 0.30:
                if regime != "discount":
                    logger.warning(f"Overriding LLM {regime} → discount (u_pred={u_pred:.1%} < 30%)")
                regime = "discount"
            else:
                if regime not in ["neutral"]:
                    logger.warning(f"Overriding LLM {regime} → neutral (30% ≤ u_pred={u_pred:.1%} ≤ 80%)")
                regime = "neutral"
            
            logger.info(f"Final regime: {regime} (u_pred={u_pred:.1%})")
            
            # FIX 2: Compute price deterministically from regime
            p_new = self._compute_price_from_regime(regime, u_pred, epsilon, alpha, beta)
            
            # Compute scalars for tracking
            if regime == "surge":
                surge_scalar = (u_pred - 0.80) / 0.20
                discount_scalar = 0.0
            elif regime == "discount":
                surge_scalar = 0.0
                discount_scalar = (0.30 - u_pred) / 0.30
            else:
                surge_scalar = 0.0
                discount_scalar = 0.0
            
            logger.info(f"Computed price: ₹{p_new:.2f} for regime={regime} (u_pred={u_pred:.1%})")
            
            return PricingDecision(
                p_new=p_new,
                regime=regime,
                surge_scalar=surge_scalar,
                discount_scalar=discount_scalar,
                elasticity_used=epsilon,
                rationale=rationale,
                fallback_used=False
            )
            
        except Exception as e:
            logger.warning(f"LLM pricing failed: {e}")
            return None
    
    def _deterministic_fallback(self, u_pred: float, q_pred: float,
                               epsilon: float, alpha: float, beta: float) -> PricingDecision:
        """Deterministic fallback formula."""
    def _deterministic_fallback(self, u_pred: float, q_pred: float,
                               epsilon: float, alpha: float, beta: float) -> PricingDecision:
        """Deterministic fallback formula with improved granularity."""
        
        if u_pred > 0.80:
            regime = "surge"
            surge_scalar = (u_pred - 0.80) / 0.20
            discount_scalar = 0.0
            # Improved granularity: /100.0 instead of /10.0
            p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 100.0
            rationale = f"High utilization ({u_pred:.2%}) warrants surge pricing [FALLBACK]"
            
        elif u_pred < 0.30:
            regime = "discount"
            surge_scalar = 0.0
            discount_scalar = (0.30 - u_pred) / 0.30
            # Improved granularity: /100.0 instead of /10.0
            p_new = self.baseline - discount_scalar * beta * (self.baseline - self.lower) / 100.0
            rationale = f"Low utilization ({u_pred:.2%}) allows discount pricing [FALLBACK]"
            
        else:
            regime = "neutral"
            surge_scalar = 0.0
            discount_scalar = 0.0
            # Scale proportionally to bounds range
            p_new = self.baseline + (u_pred - 0.55) * (self.upper - self.lower) / 2.0
            rationale = f"Moderate utilization ({u_pred:.2%}) suggests neutral pricing [FALLBACK]"
        
        p_new = np.clip(p_new, self.lower, self.upper)
        
        return PricingDecision(
            p_new=p_new,
            regime=regime,
            surge_scalar=surge_scalar,
            discount_scalar=discount_scalar,
            elasticity_used=epsilon,
            rationale=rationale,
            fallback_used=True
        )
    
    def apply_update(self, delta: np.ndarray) -> None:
        """Update theta parameters with logging."""
        theta_before = self.theta.copy()
        self.theta += delta
        self.theta[0] = np.clip(self.theta[0], 0.1, 5.0)
        self.theta[1] = np.clip(self.theta[1], 1.0, 10.0)
        self.theta[2] = np.clip(self.theta[2], 1.0, 10.0)
        
        # Log the update for debugging
        logger.debug(f"Theta update: {theta_before} -> {self.theta}, delta={delta}")
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        total = self.llm_success_count + self.fallback_count
        llm_rate = (self.llm_success_count / total * 100) if total > 0 else 0
        return {
            "llm_success": self.llm_success_count,
            "fallback_used": self.fallback_count,
            "llm_success_rate": llm_rate
        }
