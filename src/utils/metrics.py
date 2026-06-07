"""Revenue and reward computation engine."""
import numpy as np
from typing import Tuple

# Mean kWh delivered per ACN session (used as constant across all revenue calculations)
ACN_AVG_KWH = 9.0
ACN_BASELINE_REVENUE = ACN_AVG_KWH * 15.0  # ₹135 per session at flat ₹15/kWh


class MetricsEngine:
    """Computes step-level revenue, utilization, and reward metrics."""

    def __init__(self, reward_weights: Tuple[float, float, float] = (0.33, 0.33, 0.34)):
        self.w1, self.w2, self.w3 = reward_weights

    def compute_demand_shift(self, epsilon: float, p_new: float, baseline: float) -> float:
        """Fractional demand change from price elasticity: ds = -ε × (Δp / p0)."""
        return -epsilon * (p_new - baseline) / baseline

    def compute_revenue(
        self,
        p_new: float,
        total_volume: float,
        demand_shift: float,
        baseline: float,
    ) -> Tuple[float, float, float]:
        """
        Session-count revenue model: revenue = price × ACN_AVG_KWH × sessions.

        total_volume is the UrbanEV session count for this time-slot.
        Returns (revenue_baseline, revenue_new, revenue_gain_pct).
        """
        revenue_baseline = baseline * ACN_AVG_KWH * total_volume
        adjusted_sessions = max(0.01 * total_volume, total_volume * (1.0 + demand_shift))
        revenue_new = p_new * ACN_AVG_KWH * adjusted_sessions
        revenue_gain_pct = (
            (revenue_new - revenue_baseline) / revenue_baseline * 100
            if revenue_baseline > 0 else 0.0
        )
        return revenue_baseline, revenue_new, revenue_gain_pct

    def compute_utilization_adjustment(self, u_actual: float, demand_shift: float) -> float:
        """Utilization after demand-side elasticity adjustment."""
        return np.clip(u_actual + demand_shift * 0.1, 0.0, 1.0)

    def compute_reward(
        self,
        revenue_gain_pct: float,
        u_baseline: float,
        u_new: float,
        q_actual: float,
        q_baseline_mean: float,
    ) -> dict:
        """
        Multi-objective reward: revenue gain + utilization change - congestion penalty.
        All components normalised to percentage scale before weighting.
        """
        revenue_component      = revenue_gain_pct
        utilization_improvement = (u_new - u_baseline) * 100
        queue_delta            = (q_actual - q_baseline_mean) / max(q_baseline_mean, 0.1)
        congestion_penalty     = min(queue_delta * 100, 100)

        reward = (
            self.w1 * revenue_component
            + self.w2 * utilization_improvement
            - self.w3 * congestion_penalty
        )
        return {
            "reward": reward,
            "revenue_component": revenue_component,
            "utilization_component": utilization_improvement,
            "congestion_component": congestion_penalty,
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
        regime: str = "neutral",
        total_volume: float = None,
    ) -> dict:
        """
        Compute all metrics for one optimization step.
        total_volume is the UrbanEV session count; falls back to kwh / ACN_AVG_KWH if absent.
        """
        if total_volume is None or total_volume <= 0:
            total_volume = max(1.0, kwh / ACN_AVG_KWH)

        demand_shift = self.compute_demand_shift(epsilon, p_new, baseline)

        # Cap elasticity response per regime to prevent runaway demand swings
        if regime == "surge":
            demand_shift = max(demand_shift, -0.30)
        elif regime == "discount":
            demand_shift = min(demand_shift, 0.40)
        else:
            demand_shift = np.clip(demand_shift, -0.25, 0.25)

        rev_base, rev_new, rev_gain = self.compute_revenue(p_new, total_volume, demand_shift, baseline)
        u_new = self.compute_utilization_adjustment(u_actual, demand_shift)
        reward_result = self.compute_reward(rev_gain, u_actual, u_new, q_actual, q_baseline_mean)

        return {
            "demand_shift": demand_shift,
            "revenue_baseline": rev_base,
            "revenue_new": rev_new,
            "revenue_gain_pct": rev_gain,
            "utilization_new": u_new,
            "reward": reward_result["reward"],
            "reward_revenue_component": reward_result["revenue_component"],
            "reward_utilization_component": reward_result["utilization_component"],
            "reward_congestion_component": reward_result["congestion_component"],
            "pricing_efficiency": rev_new / (total_volume * ACN_AVG_KWH) if total_volume > 0 else 0,
        }
