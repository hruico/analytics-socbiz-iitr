"""Monitoring agent with LLM reasoning and robust fallback."""
from typing import List, Optional
from pydantic import BaseModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
    """Evaluates outcomes and proposes parameter updates using LLM with fallback."""
    
    def __init__(self, llm_provider=None):
        self.llm = llm_provider
        self.llm_success_count = 0
        self.fallback_count = 0
        self.last_epsilon_reduction_step = -999  # Track last epsilon reduction for cooldown
    
    def evaluate_and_propose(
        self,
        step: int,
        revenue_gain_pct: float,
        u_actual: float,
        u_pred: float,
        regime: str,
        recent_history: List[StepMetrics],
        current_theta: np.ndarray
    ) -> LearningUpdate:
        """
        Evaluate pricing decision and propose parameter adjustment using LLM with fallback.
        """
        reward = revenue_gain_pct
        
        # Try LLM reasoning first
        if self.llm is not None:
            update = self._llm_parameter_update(step, revenue_gain_pct, u_actual, u_pred,
                                               regime, recent_history, current_theta, reward)
            if update is not None:
                self.llm_success_count += 1
                return update
        
        # Fallback to deterministic
        self.fallback_count += 1
        return self._deterministic_fallback(step, revenue_gain_pct, u_actual, regime, 
                                           recent_history, reward, current_theta[0])  # Pass epsilon
    
    def _llm_parameter_update(self, step: int, revenue_gain_pct: float, u_actual: float,
                             u_pred: float, regime: str, recent_history: List[StepMetrics],
                             current_theta: np.ndarray, reward: float) -> Optional[LearningUpdate]:
        """
        Use LLM to propose parameter updates.
        FIX 1: Stateless - no history, fresh prompt each call.
        """
        
        # Prepare history summary
        if len(recent_history) >= 3:
            recent_revenue = [m.revenue_gain_pct for m in recent_history[-3:]]
            recent_util = [m.u_actual for m in recent_history[-3:]]
            history_summary = f"Last 3 steps: revenue=[{recent_revenue[0]:+.1f}%, {recent_revenue[1]:+.1f}%, {recent_revenue[2]:+.1f}%], utilization=[{recent_util[0]:.1%}, {recent_util[1]:.1%}, {recent_util[2]:.1%}]"
        else:
            history_summary = "Insufficient history (< 3 steps)"
        
        epsilon, alpha, beta = current_theta
        
        # FIX 1: Build fresh prompt each time - NO HISTORY
        prompt = f"""You are optimizing EV charging parameters. Evaluate the pricing outcome and propose adjustments.

OUTCOME:
- Revenue gain: {revenue_gain_pct:+.2f}%
- Actual util: {u_actual:.1%}
- Predicted util: {u_pred:.1%}
- Regime: {regime}
- Step: {step}

PARAMETERS:
- ε={epsilon:.3f} [range: 0.1-5.0]
- α={alpha:.3f} [range: 1.0-10.0]
- β={beta:.3f} [range: 1.0-10.0]

HISTORY:
{history_summary}

GUIDELINES:
- If revenue declining 3+ steps IN NEUTRAL REGIME: reduce ε (negative delta)
- If high util persists (>80%) in surge: increase α (positive delta)
- If low util persists (<30%) in discount: increase β (positive delta)
- Otherwise: keep stable (zeros)

IMPORTANT: Epsilon adjustments only make sense in neutral regime. Surge/discount have different dynamics.

CONSTRAINTS: |Δε|≤0.05, |Δα|≤0.10, |Δβ|≤0.10

Respond with JSON only:
{{"delta_epsilon": <-0.05 to 0.05>, "delta_alpha": <-0.10 to 0.10>, "delta_beta": <-0.10 to 0.10>, "reflection": "<your_reasoning>"}}"""
        
        try:
            # FIX 1: Invoke LLM with NO conversation history
            response = self.llm.invoke_with_retry(prompt, response_format="json")
            
            if response is None:
                return None
            
            # Extract and validate deltas with robust type handling
            def safe_float_extract(value, default=0.0):
                """Extract float from various response formats (float, list, string)."""
                if value is None:
                    return default
                if isinstance(value, (int, float)):
                    return float(value)
                if isinstance(value, list) and len(value) > 0:
                    return float(value[0])
                if isinstance(value, str):
                    return float(value)
                return default
            
            delta_eps = safe_float_extract(response.get("delta_epsilon"), 0.0)
            delta_alpha = safe_float_extract(response.get("delta_alpha"), 0.0)
            delta_beta = safe_float_extract(response.get("delta_beta"), 0.0)
            
            logger.info(f"LLM monitoring response: Δε={delta_eps}, Δα={delta_alpha}, Δβ={delta_beta}")
            
            # Enforce magnitude constraints
            delta_eps = np.clip(delta_eps, -0.05, 0.05)
            delta_alpha = np.clip(delta_alpha, -0.10, 0.10)
            delta_beta = np.clip(delta_beta, -0.10, 0.10)
            
            # Validate business logic
            if len(recent_history) >= 3:
                recent_revenue = [m.revenue_gain_pct for m in recent_history[-3:]]
                recent_util = [m.u_actual for m in recent_history[-3:]]
                
                # Check directional correctness
                if all(r < 0 for r in recent_revenue) and delta_eps > 0:
                    logger.warning("LLM proposed increase ε during revenue decline, correcting")
                    delta_eps = min(delta_eps, 0.0)
                
                if all(u > 0.80 for u in recent_util) and regime == "surge" and delta_alpha < 0:
                    logger.warning("LLM proposed decrease α during high util, correcting")
                    delta_alpha = max(delta_alpha, 0.0)
            
            return LearningUpdate(
                delta_epsilon=delta_eps,
                delta_alpha=delta_alpha,
                delta_beta=delta_beta,
                reward=reward,
                revenue_gain_pct=revenue_gain_pct,
                reflection=response.get("reflection", "LLM parameter adjustment")
            )
            
        except Exception as e:
            logger.warning(f"LLM monitoring failed: {e}")
            return None
    
    def _deterministic_fallback(self, revenue_gain_pct: float, u_actual: float,
                               regime: str, recent_history: List[StepMetrics],
                               reward: float) -> LearningUpdate:
        """Deterministic fallback for parameter updates."""
    def _deterministic_fallback(self, step: int, revenue_gain_pct: float, u_actual: float,
                               regime: str, recent_history: List[StepMetrics],
                               reward: float, current_epsilon: float) -> LearningUpdate:
        """
        Deterministic fallback for parameter updates with regime-aware logic.
        FIX 3: Regime-aware epsilon cooldown - only reduce in neutral, with 5-step cooldown.
        """
        
        delta_eps, delta_alpha, delta_beta = 0.0, 0.0, 0.0
        
        if len(recent_history) >= 3:
            recent_revenue = [m.revenue_gain_pct for m in recent_history[-3:]]
            recent_util = [m.u_actual for m in recent_history[-3:]]
            recent_regimes = [m.regime for m in recent_history[-3:]]
            
            # FIX 2: Epsilon reduction - ONLY in neutral regime, with cooldown AND floor
            if regime == "neutral":
                # Check if cooldown period has passed (5 steps minimum)
                cooldown_passed = (step - self.last_epsilon_reduction_step) >= 5
                
                # FIX 2: Add epsilon floor check - never reduce below 1.0
                epsilon_above_floor = current_epsilon > 1.0
                
                if all(r < 0 for r in recent_revenue) and cooldown_passed and epsilon_above_floor:
                    delta_eps = -0.02
                    self.last_epsilon_reduction_step = step
                    logger.info(f"Epsilon reduction triggered at step {step} (ε={current_epsilon:.3f} > 1.0)")
                elif all(r < 0 for r in recent_revenue) and not cooldown_passed:
                    logger.info(f"Epsilon reduction skipped: cooldown active (last at step {self.last_epsilon_reduction_step})")
                elif all(r < 0 for r in recent_revenue) and not epsilon_above_floor:
                    logger.info(f"Epsilon reduction skipped: floor reached (ε={current_epsilon:.3f} ≤ 1.0)")
            else:
                # FIX 3: NO epsilon reduction in surge/discount regimes
                logger.debug(f"Epsilon reduction skipped: {regime} regime (only applies to neutral)")
            
            # Alpha: RELAXED - even a single surge step with high util triggers update
            high_util_surge_count = sum(1 for u, r in zip(recent_util, recent_regimes) if u > 0.80 and r == "surge")
            if high_util_surge_count >= 1:
                delta_alpha = 0.05
                logger.info(f"Alpha update triggered: {high_util_surge_count} surge step(s) with u>80%")
            
            # Beta: RELAXED - even a single discount step with low util triggers update
            low_util_discount_count = sum(1 for u, r in zip(recent_util, recent_regimes) if u < 0.30 and r == "discount")
            if low_util_discount_count >= 1:
                delta_beta = 0.05
                logger.info(f"Beta update triggered: {low_util_discount_count} discount step(s) with u<30%")
        
        return LearningUpdate(
            delta_epsilon=delta_eps,
            delta_alpha=delta_alpha,
            delta_beta=delta_beta,
            reward=reward,
            revenue_gain_pct=revenue_gain_pct,
            reflection="Deterministic adjustment based on recent patterns [FALLBACK]"
        )
    
    def get_stats(self) -> dict:
        """Get agent statistics."""
        total = self.llm_success_count + self.fallback_count
        llm_rate = (self.llm_success_count / total * 100) if total > 0 else 0
        return {
            "llm_success": self.llm_success_count,
            "fallback_used": self.fallback_count,
            "llm_success_rate": llm_rate
        }
