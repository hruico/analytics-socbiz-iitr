"""Monitoring agent with deterministic parameter updates (LLM integration TBD)."""
from typing import List, Optional
from pydantic import BaseModel
import numpy as np


class LearningUpdate(BaseModel):
    """Parameter update proposal."""
    delta_epsilon: float
    delta_alpha: float
    delta_beta: float
    reward: float
    revenue_gain_pct: float
    reflection: str


class StepMetrics(BaseModel):
    """Metrics for a single optimization step."""
    step: int
    regime: str
    revenue_gain_pct: float
    u_actual: float
    u_pred: float


class MonitoringAgent:
    """Evaluates outcomes and proposes parameter updates (prototype: deterministic)."""
    
    def __init__(self):
        pass
    
    def evaluate_and_propose(
        self,
        step: int,
        revenue_gain_pct: float,
        u_actual: float,
        u_pred: float,
        regime: str,
        recent_history: List[StepMetrics]
    ) -> LearningUpdate:
        """
        Evaluate pricing decision and propose parameter adjustment.
        
        In full implementation, this would use LLM reasoning.
        """
        # Compute reward (simplified)
        reward = revenue_gain_pct
        
        # Deterministic parameter adjustment
        delta_eps, delta_alpha, delta_beta = 0.0, 0.0, 0.0
        
        # Check for persistent patterns in recent history
        if len(recent_history) >= 3:
            recent_revenue = [m.revenue_gain_pct for m in recent_history[-3:]]
            recent_util = [m.u_actual for m in recent_history[-3:]]
            
            # If revenue declining for 3 steps
            if all(r < 0 for r in recent_revenue):
                delta_eps = -0.02
            
            # If high utilization persists during surge
            if all(u > 0.80 for u in recent_util) and regime == "surge":
                delta_alpha = 0.10
            
            # If low utilization persists during discount
            if all(u < 0.30 for u in recent_util) and regime == "discount":
                delta_beta = 0.10
        
        return LearningUpdate(
            delta_epsilon=delta_eps,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            reward=reward,
            revenue_gain_pct=revenue_gain_pct,
            reflection="Deterministic adjustment based on recent patterns"
        )
