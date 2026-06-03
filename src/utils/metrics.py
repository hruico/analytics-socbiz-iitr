"""Metrics computation engine."""
import numpy as np
from typing import Tuple


class MetricsEngine:
    """Computes all optimization metrics deterministically."""
    
    def __init__(self, reward_weights: Tuple[float, float, float] = (1.0, 0.5, 0.3)):
        self.w1, self.w2, self.w3 = reward_weights
    
    def compute_demand_shift(self, epsilon: float, p_new: float, baseline: float) -> float:
        """Compute demand shift from price elasticity."""
        return -epsilon * (p_new - baseline) / baseline
    
    def compute_revenue(self, p_new: float, kwh: float, demand_shift: float, baseline: float) -> Tuple[float, float, float]:
        """
        Compute revenue metrics.
        
        Returns:
            (revenue_baseline, revenue_new, revenue_gain_pct)
        """
        revenue_baseline = baseline * kwh
        adjusted_demand = max(0.05, 1.0 + demand_shift)
        revenue_new = p_new * kwh * adjusted_demand
        revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100
        
        return revenue_baseline, revenue_new, revenue_gain_pct
    
    def compute_utilization_adjustment(self, u_actual: float, demand_shift: float) -> float:
        """Compute utilization after elasticity adjustment."""
        return np.clip(u_actual + demand_shift * 0.1, 0.0, 1.0)
    
    def compute_reward(
        self,
        revenue_gain_pct: float,
        u_baseline: float,
        u_new: float,
        q_actual: float,
        q_baseline_mean: float
    ) -> dict:
        """
        PROBLEM 5 FIX: Compute multi-objective reward with normalization and decomposition.
        
        All components are normalized to percentage scale (0-100) before summing.
        Weights sum to 1.0 (default: w1=0.33, w2=0.33, w3=0.33).
        
        Returns:
            dict with reward and decomposition
        """
        # Component 1: Revenue gain (already in %)
        revenue_component = revenue_gain_pct  # Range: -100 to +100
        
        # Component 2: Utilization improvement (convert to %)
        utilization_improvement = (u_new - u_baseline) * 100  # Range: -100 to +100
        
        # Component 3: Congestion penalty (normalize to %)
        # queue delta normalized by baseline, capped at 100%
        queue_delta = (q_actual - q_baseline_mean) / max(q_baseline_mean, 0.1)
        congestion_penalty = min(queue_delta * 100, 100)  # Cap at 100%
        
        # Combine with normalized weights
        reward = self.w1 * revenue_component + self.w2 * utilization_improvement - self.w3 * congestion_penalty
        
        return {
            'reward': reward,
            'revenue_component': revenue_component,
            'utilization_component': utilization_improvement,
            'congestion_component': congestion_penalty
        }
    
    def compute_step_metrics(
        self,
        p_new: float,
        kwh: float,
        u_actual: float,
        q_actual: float,
        epsilon: float,
        baseline: float,
        q_baseline_mean: float,
        regime: str = "neutral"  # Added regime parameter
    ) -> dict:
        """Compute all metrics for a single step with regime-based safety bounds."""
        demand_shift = self.compute_demand_shift(epsilon, p_new, baseline)
        
        # SAFETY: Apply regime-specific demand_shift bounds to prevent runaway elasticity
        # This prevents extreme revenue losses from high surge prices
        if regime == "surge":
            # Cap negative demand shift in surge regime to -30%
            # Rationale: Even with high prices, some inelastic demand remains
            demand_shift = max(demand_shift, -0.30)
        elif regime == "discount":
            # Cap positive demand shift in discount regime to +40%
            # Rationale: Capacity constraints limit how much demand can increase
            demand_shift = min(demand_shift, 0.40)
        else:  # neutral
            # Cap both directions in neutral regime
            demand_shift = np.clip(demand_shift, -0.25, 0.25)
        
        rev_base, rev_new, rev_gain = self.compute_revenue(p_new, kwh, demand_shift, baseline)
        u_new = self.compute_utilization_adjustment(u_actual, demand_shift)
        
        # PROBLEM 5 FIX: Get reward with decomposition
        reward_result = self.compute_reward(rev_gain, u_actual, u_new, q_actual, q_baseline_mean)
        
        return {
            'demand_shift': demand_shift,
            'revenue_baseline': rev_base,
            'revenue_new': rev_new,
            'revenue_gain_pct': rev_gain,
            'utilization_new': u_new,
            'reward': reward_result['reward'],
            'reward_revenue_component': reward_result['revenue_component'],
            'reward_utilization_component': reward_result['utilization_component'],
            'reward_congestion_component': reward_result['congestion_component'],
            'pricing_efficiency': rev_new / kwh if kwh > 0 else 0
        }
