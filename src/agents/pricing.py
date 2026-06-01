# =============================================================================
# src/agents/pricing.py — TariffPricingAgent (LangGraph-based)
#
# A genuine LangGraph agent that reasons contextually about demand state
# before proposing a tariff. The LLM is asked to think, not to execute formulas.
#
# Graph nodes:
#   analyse_state   → LLM reasons about the demand context
#   compute_price   → LLM proposes a price with justification
#   validate        → Python enforces hard constraints
#
# Falls back to deterministic formula only if all LLM attempts fail.
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
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from pydantic import ValidationError

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from src.config import (
    P_BASE,
    P_SURGE_CAP,
    P_DISCOUNT_FLOOR,
    SURGE_THRESHOLD,
    DISCOUNT_THRESHOLD,
    ForecastState,
    PricingDecision,
    GROQ_MODEL,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state schema
# ─────────────────────────────────────────────────────────────────────────────

class PricingState(TypedDict):
    state: ForecastState
    epsilon: float
    alpha: float
    beta: float
    analysis: str           # LLM's contextual reasoning
    decision: PricingDecision | None
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# System prompt — reasoning-first, not formula-execution
# ─────────────────────────────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """You are a dynamic pricing strategist for an EV charging network in India.

Your job is to analyse the current demand situation and decide whether the context
justifies surge pricing, discount pricing, or staying neutral — and by how much.

Think about:
- Is the utilisation genuinely high enough to warrant surge, or is it borderline?
- Is the time of day / weekend context relevant? (e.g. late night surge may drive users away)
- Is the queue long enough to suggest real congestion?
- What does the current elasticity parameter tell you about how sensitive demand is?

Respond with 2-3 sentences of reasoning. Be specific about the numbers.
Do NOT return JSON yet — just your analysis."""

_PRICE_SYSTEM = """You are a dynamic pricing engine for an EV charging network.

Based on the demand analysis provided, compute the optimal tariff.

Hard rules (you MUST follow these):
- Baseline: 15.0 Rs/kWh
- Surge regime (utilisation > 0.80): price between 15.01 and 22.0 Rs/kWh
- Discount regime (utilisation < 0.30): price between 10.0 and 14.99 Rs/kWh
- Neutral regime (0.30 to 0.80): price between 10.0 and 22.0 Rs/kWh
- elasticity_used MUST equal the current epsilon value provided to you exactly

You may deviate from the mechanical formula if your analysis justifies it.
For example: borderline surge at 11pm on a weekend might warrant a softer price
than the formula would give, to avoid driving away users.

Return ONLY valid JSON with exactly these keys:
{
  "p_new": <number between 10.0 and 22.0>,
  "regime": <"surge" or "discount" or "neutral">,
  "surge_scalar": <number between 0.0 and 1.0>,
  "discount_scalar": <number between 0.0 and 1.0>,
  "elasticity_used": <must equal the epsilon value you were given>,
  "rationale": "<1-2 sentences explaining your pricing decision>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# TariffPricingAgent
# ─────────────────────────────────────────────────────────────────────────────

class TariffPricingAgent:
    """
    LangGraph-based tariff pricing agent.

    Uses a two-step reasoning graph:
      1. analyse_state — LLM reasons about demand context
      2. compute_price — LLM proposes price with justification
      3. validate      — Python enforces hard constraints

    Falls back to deterministic formula if LLM fails.
    """

    _THETA_BOUNDS: dict[str, tuple[float, float]] = {
        "epsilon": (0.1, 5.0),
        "alpha": (1.0, 10.0),
        "beta": (1.0, 10.0),
    }

    def __init__(
        self,
        epsilon_init: float = 0.3,
        alpha_init: float = 4.0,
        beta_init: float = 4.0,
        max_retries: int = 3,
    ) -> None:
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")

        self._theta = np.array([epsilon_init, alpha_init, beta_init], dtype=float)
        self._max_retries = max_retries

        self._llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=0.2,
        )

        self._graph = self._build_graph()

        logger.info(
            "TariffPricingAgent (LangGraph) initialised: ε=%.3f α=%.3f β=%.3f",
            self.epsilon, self.alpha, self.beta,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        return float(self._theta[0])

    @property
    def alpha(self) -> float:
        return float(self._theta[1])

    @property
    def beta(self) -> float:
        return float(self._theta[2])

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(PricingState)
        graph.add_node("analyse_state", self._node_analyse)
        graph.add_node("compute_price", self._node_compute_price)
        graph.add_node("validate", self._node_validate)
        graph.set_entry_point("analyse_state")
        graph.add_edge("analyse_state", "compute_price")
        graph.add_edge("compute_price", "validate")
        graph.add_edge("validate", END)
        return graph.compile()

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def _node_analyse(self, state: PricingState) -> PricingState:
        """LLM reasons about the demand context."""
        s = state["state"]
        prompt = (
            f"Current demand state:\n"
            f"  Predicted utilisation: {s.u_pred:.4f} (surge threshold: {SURGE_THRESHOLD}, discount threshold: {DISCOUNT_THRESHOLD})\n"
            f"  Predicted queue length: {s.q_pred:.2f}\n"
            f"  Hour of day: {s.hour_of_day} ({'weekend' if s.is_weekend else 'weekday'})\n"
            f"  kWh to be delivered: {s.kwh_delivered:.2f}\n"
            f"  Current elasticity (epsilon): {state['epsilon']:.4f}\n"
            f"  Current surge sensitivity (alpha): {state['alpha']:.4f}\n"
            f"  Current discount sensitivity (beta): {state['beta']:.4f}\n\n"
            f"Analyse this demand situation. Should we surge, discount, or stay neutral? "
            f"Consider the time of day, queue length, and how elastic demand is."
        )
        try:
            response = self._llm.invoke([
                SystemMessage(content=_ANALYSIS_SYSTEM),
                HumanMessage(content=prompt),
            ])
            analysis = response.content.strip()
            logger.debug("TariffPricingAgent analysis: %s", analysis[:120])
        except Exception as exc:
            logger.warning("TariffPricingAgent: analysis node failed — %s", exc)
            analysis = f"Fallback: u_pred={s.u_pred:.4f}, hour={s.hour_of_day}"

        return {**state, "analysis": analysis}

    def _node_compute_price(self, state: PricingState) -> PricingState:
        """LLM proposes a price based on the analysis."""
        s = state["state"]
        prompt = (
            f"Demand analysis:\n{state['analysis']}\n\n"
            f"State summary:\n"
            f"  utilisation={s.u_pred:.4f}, queue={s.q_pred:.2f}, "
            f"hour={s.hour_of_day}, weekend={bool(s.is_weekend)}\n"
            f"  epsilon={state['epsilon']:.4f} (use this exact value for elasticity_used)\n\n"
            f"Compute the optimal tariff and return JSON."
        )
        for attempt in range(self._max_retries):
            try:
                response = self._llm.invoke([
                    SystemMessage(content=_PRICE_SYSTEM),
                    HumanMessage(content=prompt),
                ])
                raw = response.content.strip()
                # Strip markdown fences if present
                if raw.startswith("```"):
                    raw = "\n".join(
                        l for l in raw.splitlines() if not l.startswith("```")
                    ).strip()
                data = json.loads(raw)
                # Ensure elasticity_used is always set to current epsilon
                data["elasticity_used"] = float(state["epsilon"])
                decision = PricingDecision(**data)
                return {**state, "decision": decision, "error": None}
            except Exception as exc:
                logger.warning(
                    "TariffPricingAgent: compute_price attempt %d/%d failed — %s",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(2.0)

        # All retries failed — use deterministic fallback
        logger.error("TariffPricingAgent: LLM failed, using deterministic fallback")
        return {**state, "decision": None, "error": "llm_failed"}

    def _node_validate(self, state: PricingState) -> PricingState:
        """Python enforces hard constraints on the LLM decision."""
        if state["decision"] is None:
            # LLM failed — compute deterministic fallback
            decision = self._deterministic_fallback(state["state"])
            return {**state, "decision": decision}

        decision = state["decision"]
        s = state["state"]

        # Enforce regime/price consistency
        if decision.regime == "surge" and decision.p_new <= P_BASE:
            logger.warning("TariffPricingAgent: regime/price mismatch — overriding")
            decision = self._deterministic_fallback(s)
        elif decision.regime == "discount" and decision.p_new >= P_BASE:
            logger.warning("TariffPricingAgent: regime/price mismatch — overriding")
            decision = self._deterministic_fallback(s)

        return {**state, "decision": decision}

    # ── Public interface ──────────────────────────────────────────────────────

    def compute_tariff(self, state: ForecastState) -> PricingDecision:
        """Run the LangGraph pricing pipeline for the given state."""
        initial: PricingState = {
            "state": state,
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "beta": self.beta,
            "analysis": "",
            "decision": None,
            "error": None,
        }
        result = self._graph.invoke(initial)
        return result["decision"]

    def apply_update(self, delta: np.ndarray) -> None:
        """Update Θ by adding delta and clip each component to its bounds."""
        self._theta = self._theta + delta
        eps_lo, eps_hi = self._THETA_BOUNDS["epsilon"]
        alp_lo, alp_hi = self._THETA_BOUNDS["alpha"]
        bet_lo, bet_hi = self._THETA_BOUNDS["beta"]
        self._theta[0] = float(np.clip(self._theta[0], eps_lo, eps_hi))
        self._theta[1] = float(np.clip(self._theta[1], alp_lo, alp_hi))
        self._theta[2] = float(np.clip(self._theta[2], bet_lo, bet_hi))
        logger.debug(
            "TariffPricingAgent: θ updated → ε=%.4f α=%.4f β=%.4f",
            self.epsilon, self.alpha, self.beta,
        )

    def run_sensitivity_analysis(
        self,
        test_df: pd.DataFrame,
        epsilon_values: list[float],
    ) -> pd.DataFrame:
        """Sweep over epsilon_values using the deterministic fallback."""
        original_epsilon = float(self._theta[0])
        rows: list[dict[str, float]] = []

        for eps in epsilon_values:
            self._theta[0] = float(eps)
            revenue_gains: list[float] = []
            for _, row in test_df.iterrows():
                u_pred = float(row.get("u_pred", row.get("urban_mean_utilization", 0.5)))
                kwh = max(0.01, float(row.get("kwh_delivered", row.get("acn_total_kwh", 10.0))))
                p_new = self._fallback_price(u_pred)
                demand_shift = -eps * ((p_new - P_BASE) / P_BASE)
                revenue_new = p_new * kwh * max(0.05, 1.0 + demand_shift)
                revenue_baseline = P_BASE * kwh
                revenue_gains.append((revenue_new - revenue_baseline) / revenue_baseline * 100.0)

            gains_arr = np.array(revenue_gains)
            rows.append({
                "epsilon": float(eps),
                "mean_revenue_gain_pct": float(np.mean(gains_arr)),
                "std_revenue_gain_pct": float(np.std(gains_arr)),
                "min_revenue_gain_pct": float(np.min(gains_arr)),
                "max_revenue_gain_pct": float(np.max(gains_arr)),
            })

        self._theta[0] = original_epsilon
        return pd.DataFrame(rows, columns=[
            "epsilon", "mean_revenue_gain_pct", "std_revenue_gain_pct",
            "min_revenue_gain_pct", "max_revenue_gain_pct",
        ])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fallback_price(self, u_pred: float) -> float:
        if u_pred > SURGE_THRESHOLD:
            surge_scalar = float(np.clip(
                (u_pred - SURGE_THRESHOLD) / (1.0 - SURGE_THRESHOLD), 0.0, 1.0
            ))
            return float(np.clip(
                P_BASE + surge_scalar * self.alpha * (P_SURGE_CAP - P_BASE) / 10.0,
                P_DISCOUNT_FLOOR, P_SURGE_CAP,
            ))
        elif u_pred < DISCOUNT_THRESHOLD:
            discount_scalar = float(np.clip(
                (DISCOUNT_THRESHOLD - u_pred) / DISCOUNT_THRESHOLD, 0.0, 1.0
            ))
            return float(np.clip(
                P_BASE - discount_scalar * self.beta * (P_BASE - P_DISCOUNT_FLOOR) / 10.0,
                P_DISCOUNT_FLOOR, P_SURGE_CAP,
            ))
        else:
            return float(np.clip(P_BASE + (u_pred - 0.55) * 2.0, P_DISCOUNT_FLOOR, P_SURGE_CAP))

    def _deterministic_fallback(self, state: ForecastState) -> PricingDecision:
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
                f"regime={regime}, p_new=Rs{p_new:.2f}/kWh "
                f"(ε={self.epsilon:.3f}, α={self.alpha:.3f}, β={self.beta:.3f})"
            ),
        )
