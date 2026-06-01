# =============================================================================
# src/agents/monitoring.py — MonitoringLearningAgent (LangGraph-based)
#
# A genuine LangGraph agent that evaluates pricing outcomes and proposes
# parameter updates through contextual reasoning, not formula execution.
#
# Graph nodes:
#   compute_metrics  → Python computes all OP'26 metrics deterministically
#   evaluate_outcome → LLM reflects on what the pricing decision achieved
#   propose_update   → LLM proposes Δθ with economic reasoning
#   apply_update     → Python applies and clips the update
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
    ForecastState,
    PricingDecision,
    LearningUpdate,
    GROQ_MODEL,
)
from src.agents.pricing import TariffPricingAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state schema
# ─────────────────────────────────────────────────────────────────────────────

class MonitorState(TypedDict):
    state: ForecastState
    decision: PricingDecision
    epsilon: float
    alpha: float
    beta: float
    step: int
    # Computed metrics
    demand_shift: float
    revenue_new: float
    revenue_baseline: float
    revenue_gain_pct: float
    charger_utilisation: float
    avg_wait_reduction: float
    pricing_efficiency: float
    reward: float
    # LLM outputs
    reflection: str
    update: LearningUpdate | None
    error: str | None


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_EVALUATE_SYSTEM = """You are a pricing performance analyst for an EV charging network.

Your job is to evaluate whether the last pricing decision was good or bad,
and explain WHY in economic terms.

Think about:
- Did the revenue gain justify the price change?
- Did the charger utilisation improve or worsen?
- Was the demand response (customer_response_rate) what we expected given the elasticity?
- Is the current elasticity parameter (epsilon) calibrated correctly?
  If demand barely shifted when we surged, epsilon might be too high (demand is more inelastic).
  If demand collapsed when we surged, epsilon might be too low.

Write 2-3 sentences of honest evaluation. Be specific about the numbers."""

_UPDATE_SYSTEM = """You are a parameter tuning agent for an EV charging pricing system.

Based on the performance evaluation, propose small adjustments to the three parameters:
- epsilon: price elasticity (how sensitive demand is to price changes)
- alpha: surge pricing sensitivity (how aggressively we surge)
- beta: discount pricing sensitivity (how aggressively we discount)

Rules:
- Keep deltas small: |delta_epsilon| <= 0.05, |delta_alpha| <= 0.10, |delta_beta| <= 0.10
- If revenue gain was positive and demand held up, epsilon might be slightly too high — decrease it
- If demand collapsed during surge, epsilon is too low — increase it
- If we're in surge regime and revenue is good, alpha is working — small positive delta
- If we're in discount regime and utilisation improved, beta is working — small positive delta

Return ONLY valid JSON with exactly these keys:
{
  "delta_epsilon": <number, magnitude <= 0.05>,
  "delta_alpha": <number, magnitude <= 0.10>,
  "delta_beta": <number, magnitude <= 0.10>,
  "reward": <the reward value you were given>,
  "revenue_gain_pct": <the revenue_gain_pct you were given>,
  "charger_utilisation": <the charger_utilisation you were given>,
  "avg_wait_reduction": <the avg_wait_reduction you were given>,
  "pricing_efficiency": <the pricing_efficiency you were given>,
  "demand_shift": <the demand_shift you were given>,
  "reflection": "<your 2-3 sentence evaluation of this step>"
}"""


# ─────────────────────────────────────────────────────────────────────────────
# MonitoringLearningAgent
# ─────────────────────────────────────────────────────────────────────────────

class MonitoringLearningAgent:
    """
    LangGraph-based monitoring and learning agent.

    Evaluates each pricing decision against realised outcomes, computes all
    OP'26 metrics deterministically, then uses LLM reasoning to propose Δθ.
    """

    def __init__(
        self,
        pricing_agent: TariffPricingAgent,
        lr: float = 0.8,
        lr_decay: float = 0.002,
        max_retries: int = 3,
    ) -> None:
        import os
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY not set")

        self.pricing_agent = pricing_agent
        self._lr = lr
        self._lr_decay = lr_decay
        self._max_retries = max_retries
        self._step = 0
        self._episode_log: list[dict[str, Any]] = []

        self._llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=0.2,
        )

        self._graph = self._build_graph()

        logger.info(
            "MonitoringLearningAgent (LangGraph) initialised: lr=%.3f decay=%.5f",
            self._lr, self._lr_decay,
        )

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        graph = StateGraph(MonitorState)
        graph.add_node("compute_metrics", self._node_compute_metrics)
        graph.add_node("evaluate_outcome", self._node_evaluate_outcome)
        graph.add_node("propose_update", self._node_propose_update)
        graph.add_node("apply_update", self._node_apply_update)
        graph.set_entry_point("compute_metrics")
        graph.add_edge("compute_metrics", "evaluate_outcome")
        graph.add_edge("evaluate_outcome", "propose_update")
        graph.add_edge("propose_update", "apply_update")
        graph.add_edge("apply_update", END)
        return graph.compile()

    # ── Graph nodes ───────────────────────────────────────────────────────────

    def _node_compute_metrics(self, state: MonitorState) -> MonitorState:
        """Deterministically compute all OP'26 metrics."""
        s = state["state"]
        d = state["decision"]
        epsilon = state["epsilon"]

        demand_shift = -epsilon * ((d.p_new - P_BASE) / P_BASE)
        revenue_new = d.p_new * s.kwh_delivered * max(0.05, 1.0 + demand_shift)
        revenue_baseline = P_BASE * s.kwh_delivered
        revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100.0
        charger_utilisation = float(np.clip(s.u_actual + demand_shift * 0.1, 0.0, 1.0))
        avg_wait_reduction = -demand_shift * s.q_actual
        pricing_efficiency = revenue_new / s.kwh_delivered
        reward = (
            0.5 * float(np.tanh(revenue_gain_pct / 20.0))
            + 0.3 * charger_utilisation
            - 0.2 * max(0.0, -avg_wait_reduction)
        )

        return {
            **state,
            "demand_shift": demand_shift,
            "revenue_new": revenue_new,
            "revenue_baseline": revenue_baseline,
            "revenue_gain_pct": revenue_gain_pct,
            "charger_utilisation": charger_utilisation,
            "avg_wait_reduction": avg_wait_reduction,
            "pricing_efficiency": pricing_efficiency,
            "reward": reward,
        }

    def _node_evaluate_outcome(self, state: MonitorState) -> MonitorState:
        """LLM reflects on what the pricing decision achieved."""
        s = state["state"]
        d = state["decision"]
        prompt = (
            f"Pricing decision just made:\n"
            f"  Regime: {d.regime}, Price: Rs{d.p_new:.2f}/kWh (baseline: Rs{P_BASE}/kWh)\n"
            f"  Rationale given: {d.rationale}\n\n"
            f"Outcomes:\n"
            f"  Revenue gain: {state['revenue_gain_pct']:+.2f}%\n"
            f"  Demand shift (customer response): {state['demand_shift']:+.4f} "
            f"({state['demand_shift']*100:+.1f}% change in demand)\n"
            f"  Charger utilisation after: {state['charger_utilisation']:.4f} "
            f"(actual before pricing: {s.u_actual:.4f})\n"
            f"  Avg wait reduction: {state['avg_wait_reduction']:+.4f}\n"
            f"  Pricing efficiency: Rs{state['pricing_efficiency']:.2f}/kWh\n"
            f"  Reward: {state['reward']:+.4f}\n\n"
            f"Current parameters: epsilon={state['epsilon']:.4f}, "
            f"alpha={state['alpha']:.4f}, beta={state['beta']:.4f}\n\n"
            f"Evaluate this pricing decision. Was it good? What should change?"
        )
        try:
            response = self._llm.invoke([
                SystemMessage(content=_EVALUATE_SYSTEM),
                HumanMessage(content=prompt),
            ])
            reflection = response.content.strip()
            logger.debug("MonitoringAgent reflection: %s", reflection[:120])
        except Exception as exc:
            logger.warning("MonitoringAgent: evaluate_outcome failed — %s", exc)
            reflection = f"Fallback evaluation: reward={state['reward']:.4f}"

        return {**state, "reflection": reflection}

    def _node_propose_update(self, state: MonitorState) -> MonitorState:
        """LLM proposes Δθ based on the evaluation."""
        prompt = (
            f"Performance evaluation:\n{state['reflection']}\n\n"
            f"Metrics to echo back exactly:\n"
            f"  reward={state['reward']:.6f}\n"
            f"  revenue_gain_pct={state['revenue_gain_pct']:.6f}\n"
            f"  charger_utilisation={state['charger_utilisation']:.6f}\n"
            f"  avg_wait_reduction={state['avg_wait_reduction']:.6f}\n"
            f"  pricing_efficiency={state['pricing_efficiency']:.6f}\n"
            f"  demand_shift={state['demand_shift']:.6f}\n\n"
            f"Current parameters: epsilon={state['epsilon']:.4f}, "
            f"alpha={state['alpha']:.4f}, beta={state['beta']:.4f}\n\n"
            f"Propose parameter updates and return JSON."
        )
        for attempt in range(self._max_retries):
            try:
                response = self._llm.invoke([
                    SystemMessage(content=_UPDATE_SYSTEM),
                    HumanMessage(content=prompt),
                ])
                raw = response.content.strip()
                if raw.startswith("```"):
                    raw = "\n".join(
                        l for l in raw.splitlines() if not l.startswith("```")
                    ).strip()
                data = json.loads(raw)
                # Always use our computed metrics, not LLM's echoed values
                data["reward"] = state["reward"]
                data["revenue_gain_pct"] = state["revenue_gain_pct"]
                data["charger_utilisation"] = state["charger_utilisation"]
                data["avg_wait_reduction"] = state["avg_wait_reduction"]
                data["pricing_efficiency"] = state["pricing_efficiency"]
                data["demand_shift"] = state["demand_shift"]
                data["reflection"] = state["reflection"]
                update = LearningUpdate(**data)
                return {**state, "update": update, "error": None}
            except Exception as exc:
                logger.warning(
                    "MonitoringAgent: propose_update attempt %d/%d failed — %s",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(2.0)

        logger.error("MonitoringAgent: LLM failed, using deterministic fallback")
        return {**state, "update": None, "error": "llm_failed"}

    def _node_apply_update(self, state: MonitorState) -> MonitorState:
        """Apply the proposed update (or deterministic fallback) to θ."""
        if state["update"] is None:
            update = self._deterministic_fallback(
                state["demand_shift"], state["revenue_gain_pct"],
                state["charger_utilisation"], state["avg_wait_reduction"],
                state["pricing_efficiency"], state["reward"],
                state["reflection"],
            )
        else:
            update = state["update"]

        eta = self._lr / (1.0 + self._lr_decay * state["step"])
        delta = np.array([
            update.delta_epsilon,
            update.delta_alpha,
            update.delta_beta,
        ]) * eta
        self.pricing_agent.apply_update(delta)

        return {**state, "update": update}

    # ── Public interface ──────────────────────────────────────────────────────

    def step(self, state: ForecastState, decision: PricingDecision) -> LearningUpdate:
        """Run the LangGraph monitoring pipeline for one step."""
        initial: MonitorState = {
            "state": state,
            "decision": decision,
            "epsilon": self.pricing_agent.epsilon,
            "alpha": self.pricing_agent.alpha,
            "beta": self.pricing_agent.beta,
            "step": self._step,
            "demand_shift": 0.0,
            "revenue_new": 0.0,
            "revenue_baseline": 0.0,
            "revenue_gain_pct": 0.0,
            "charger_utilisation": 0.0,
            "avg_wait_reduction": 0.0,
            "pricing_efficiency": 0.0,
            "reward": 0.0,
            "reflection": "",
            "update": None,
            "error": None,
        }
        result = self._graph.invoke(initial)
        update = result["update"]

        eta = self._lr / (1.0 + self._lr_decay * self._step)
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
            "revenue_new": result["revenue_new"],
            "revenue_baseline": result["revenue_baseline"],
            "revenue_gain_pct": result["revenue_gain_pct"],
            "charger_utilisation": result["charger_utilisation"],
            "avg_wait_reduction": result["avg_wait_reduction"],
            "pricing_efficiency": result["pricing_efficiency"],
            "demand_shift": result["demand_shift"],
            "customer_response_rate": result["demand_shift"] * 100.0,
            "reward": result["reward"],
            "epsilon_after": self.pricing_agent.epsilon,
            "alpha_after": self.pricing_agent.alpha,
            "beta_after": self.pricing_agent.beta,
            "reflection": update.reflection,
            "lr_used": eta,
        })

        self._step += 1
        return update

    def summary(self) -> pd.DataFrame:
        if not self._episode_log:
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
        Compute Off_Peak_Uplift: percentage change in mean charger utilisation
        during discount-regime hours compared to pre-optimisation baseline.

        Per PS: increase in sessions during low-demand periods (utilisation < 30%)
        after discount pricing. We use u_actual as the utilisation proxy since
        session-level counts are not available in the episode log.

        Formula: (mean_post - mean_baseline) / mean_baseline * 100
        """
        if not self._episode_log:
            return 0.0
        discount_rows = [r for r in self._episode_log if r["regime"] == "discount"]
        if not discount_rows:
            return 0.0
        mean_post = float(np.mean([r["u_actual"] for r in discount_rows]))
        n = min(len(discount_rows), len(baseline_df))
        # Use urban_mean_utilization as the baseline utilisation proxy
        util_col = "urban_mean_utilization" if "urban_mean_utilization" in baseline_df.columns else baseline_df.columns[0]
        mean_baseline = float(baseline_df[util_col].iloc[:n].mean())
        if mean_baseline == 0.0:
            return 0.0
        return float((mean_post - mean_baseline) / mean_baseline * 100.0)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _deterministic_fallback(
        self,
        demand_shift: float,
        revenue_gain_pct: float,
        charger_utilisation: float,
        avg_wait_reduction: float,
        pricing_efficiency: float,
        reward: float,
        reflection: str = "",
    ) -> LearningUpdate:
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
            reflection=reflection or "Deterministic fallback: LLM unavailable.",
        )
