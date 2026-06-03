"""Pricing agent with deterministic fallback (LLM integration TBD)."""
from typing import Literal
from pydantic import BaseModel
import numpy as np


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
    """Determines optimal tariffs (prototype: deterministic only)."""
    
    def __init__(self, baseline: float, bounds: tuple, theta: np.ndarray):
        self.baseline = baseline
        self.lower, self.upper = bounds
        self.theta = theta  # [epsilon, alpha, beta]
    
    def compute_tariff(self, u_pred: float, q_pred: float, hour: int, 
                      is_weekend: bool, congestion_prob: float) -> PricingDecision:
        """
        Compute optimal tariff using deterministic formula.
        
        In full implementation, this would use LLM with LangGraph.
        """
        epsilon, alpha, beta = self.theta
        
        # Determine regime based on utilization
        if u_pred > 0.80:
            regime = "surge"
            surge_scalar = (u_pred - 0.80) / 0.20
            discount_scalar = 0.0
            p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 10.0
            rationale = f"High utilization ({u_pred:.2%}) warrants surge pricing"
            
        elif u_pred < 0.30:
            regime = "discount"
            surge_scalar = 0.0
            discount_scalar = (0.30 - u_pred) / 0.30
            p_new = self.baseline - discount_scalar * beta * (self.baseline - self.lower) / 10.0
            rationale = f"Low utilization ({u_pred:.2%}) allows discount pricing"
            
        else:
            regime = "neutral"
            surge_scalar = 0.0
            discount_scalar = 0.0
            p_new = self.baseline + (u_pred - 0.55) * 2.0
            rationale = f"Moderate utilization ({u_pred:.2%}) suggests neutral pricing"
        
        # Clip to bounds
        p_new = np.clip(p_new, self.lower, self.upper)
        
        return PricingDecision(
            p_new=p_new,
            regime=regime,
            surge_scalar=surge_scalar,
            discount_scalar=discount_scalar,
            elasticity_used=epsilon,
            rationale=rationale,
            fallback_used=True  # Prototype uses deterministic fallback
        )
    
    def apply_update(self, delta: np.ndarray) -> None:
        """Update theta parameters."""
        self.theta += delta
        # Clip to valid ranges
        self.theta[0] = np.clip(self.theta[0], 0.1, 5.0)  # epsilon
        self.theta[1] = np.clip(self.theta[1], 1.0, 10.0)  # alpha
        self.theta[2] = np.clip(self.theta[2], 1.0, 10.0)  # beta
