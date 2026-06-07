#!/usr/bin/env python3
"""
Compute all six evaluation metrics from agentic outcomes.

Metric sources:
  Revenue Gain %            — ACN (session-count model)
  Charger Utilization Rate  — UrbanEV (elasticity simulation)
  Off-Peak Uplift           — UrbanEV (all 168 rows)
  Avg Waiting Time Reduction — UrbanEV (all 168 rows, peak hours)
  Customer Response Rate    — ACN + elasticity
  Pricing Efficiency Score  — ACN (kWh model)
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ACN_AVG_KWH = 9.0
SURGE_THRESHOLD    = 0.48
DISCOUNT_THRESHOLD = 0.36


def simulate_price(u: float, baseline: float, alpha: float, beta: float,
                   lower: float = 10.0, upper: float = 22.0) -> float:
    """Replicate pricing-agent formula for full-dataset simulation."""
    band = SURGE_THRESHOLD - DISCOUNT_THRESHOLD
    if u > SURGE_THRESHOLD:
        p = baseline * (1 + alpha * (u - SURGE_THRESHOLD))
    elif u < DISCOUNT_THRESHOLD:
        p = baseline - (DISCOUNT_THRESHOLD - u) * beta
    else:
        position = (u - DISCOUNT_THRESHOLD) / band
        p = baseline * (1 + 0.10 * position)
    return float(np.clip(p, lower, upper))


def evaluate_demand_agent(outcomes_path: str = "outputs/agentic_outcomes.csv") -> dict:
    df = pd.read_csv(outcomes_path)
    u_pred, u_actual = df["u_pred"].values, df["u_actual"].values
    rmse = np.sqrt(np.mean((u_pred - u_actual) ** 2))
    mae  = np.mean(np.abs(u_pred - u_actual))
    ss_res = np.sum((u_actual - u_pred) ** 2)
    ss_tot = np.sum((u_actual - u_actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    logger.info(f"\n[0/6] Demand Agent  |  n={len(df)}  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n_samples": len(df)}


def evaluate_all_metrics(
    outcomes_path:  str   = "outputs/agentic_outcomes.csv",
    unified_path:   str   = "data/processed/unified_analytical_base.csv",
    baseline_price: float = 15.0,
    elasticity:     float = 0.25,
    learned_epsilon: float = None,
    learned_alpha:   float = None,
    learned_beta:    float = None,
) -> dict:
    outcomes = pd.read_csv(outcomes_path)
    unified  = pd.read_csv(unified_path)

    if any(p is None for p in [learned_epsilon, learned_alpha, learned_beta]):
        last = outcomes.iloc[-1]
        learned_epsilon = float(last.get("epsilon", 0.291))
        learned_alpha   = float(last.get("alpha",   4.0))
        learned_beta    = float(last.get("beta",    3.928))

    logger.info(f"\nLearned θ:  ε={learned_epsilon:.3f}  α={learned_alpha:.3f}  β={learned_beta:.3f}")

    merged = outcomes.merge(unified, left_on="step", right_on="time_step", how="inner", suffixes=("", "_base"))
    logger.info(f"Outcomes: {len(outcomes)}  |  Unified: {len(unified)}  |  Merged: {len(merged)}")

    results = {}

    # ── 1. Revenue Gain ───────────────────────────────────────────────────
    logger.info("\n[1/6] Revenue Gain % (session-count model)...")
    session_count  = merged["total_volume"] if "total_volume" in merged.columns else merged["acn_total_kwh"] / ACN_AVG_KWH
    dynamic_prices = merged["p_new"]

    old_rev = (baseline_price * ACN_AVG_KWH * session_count).sum()
    new_rev = (dynamic_prices * ACN_AVG_KWH * session_count).sum()
    rev_gain = (new_rev - old_rev) / old_rev * 100 if old_rev > 0 else 0

    results.update(revenue_gain_pct=rev_gain, new_revenue=new_rev, old_revenue=old_rev)
    logger.info(f"  Old: ₹{old_rev:,.0f}  New: ₹{new_rev:,.0f}  Gain: {rev_gain:+.2f}%")

    # ── 2. Charger Utilization ────────────────────────────────────────────
    logger.info("\n[2/6] Charger Utilization Rate (UrbanEV)...")
    util_col = "urban_mean_utilization_base" if "urban_mean_utilization_base" in merged.columns else "urban_mean_utilization"
    util_before = merged[util_col].mean()
    util_after  = (merged[util_col] * (1 - elasticity * (dynamic_prices - baseline_price) / baseline_price)).clip(0, 1).mean()
    util_change_pct = (util_after - util_before) / util_before * 100

    results.update(utilization_before=util_before, utilization_after=util_after, utilization_change_pct=util_change_pct)
    logger.info(f"  Before: {util_before:.1%}  After: {util_after:.1%}  Δ: {util_change_pct:+.2f}%")

    # ── 3. Off-Peak Uplift ────────────────────────────────────────────────
    logger.info("\n[3/6] Off-Peak Uplift (UrbanEV, all 168 rows)...")
    u_full  = unified["urban_mean_utilization"]
    p_sim   = u_full.apply(lambda u: simulate_price(u, baseline_price, learned_alpha, learned_beta))
    u_after = (u_full * (1 - learned_epsilon * (p_sim - baseline_price) / baseline_price)).clip(0, 1)

    off_before = (u_full  < DISCOUNT_THRESHOLD).sum()
    off_after  = (u_after < DISCOUNT_THRESHOLD).sum()
    uplift_pct = (off_after - off_before) / off_before * 100 if off_before > 0 else 0

    results.update(off_peak_before=off_before, off_peak_after=off_after,
                   off_peak_uplift=int(off_after - off_before), off_peak_uplift_pct=uplift_pct)
    logger.info(f"  Before: {off_before}  After: {off_after}  Uplift: {uplift_pct:+.1f}%")

    # ── 4. Wait Time Reduction ────────────────────────────────────────────
    logger.info("\n[4/6] Avg Wait Time Reduction (UrbanEV peak hours)...")
    q_before = u_full.apply(lambda u: max(0, (u - 0.5) * 10))
    q_after  = u_after.apply(lambda u: max(0, (u - 0.5) * 10))

    peak_mask = unified.get("is_peak_hour", pd.Series([1] * len(unified))).astype(bool)
    qb = q_before[peak_mask].mean() if peak_mask.sum() > 0 else q_before.mean()
    qa = q_after[peak_mask].mean()  if peak_mask.sum() > 0 else q_after.mean()
    q_reduction_pct = (qb - qa) / qb * 100 if qb > 0 else 0

    results.update(queue_before_peak=qb, queue_after_peak=qa,
                   queue_reduction=qb - qa, queue_reduction_pct=q_reduction_pct)
    logger.info(f"  Queue before: {qb:.4f}  After: {qa:.4f}  Δ: {q_reduction_pct:+.1f}%")

    # ── 5. Customer Response Rate ─────────────────────────────────────────
    logger.info("\n[5/6] Customer Response Rate (ACN)...")
    if "acn_sessions_count" in merged.columns:
        sessions = merged["acn_sessions_count"]
        shift    = elasticity * (baseline_price - dynamic_prices) / baseline_price
        delta    = shift * sessions
        response = (delta.abs() / sessions * 100).replace([np.inf, -np.inf], 0).fillna(0).mean()
        shifted  = delta.abs().sum()
    else:
        response = shifted = 0

    results.update(customer_response_rate=response, sessions_shifted_total=shifted)
    logger.info(f"  Mean shift: {response:.2f}%  Total sessions shifted: {shifted:.0f}")

    # ── 6. Pricing Efficiency ─────────────────────────────────────────────
    logger.info("\n[6/6] Pricing Efficiency Score (ACN)...")
    total_kwh  = (session_count * ACN_AVG_KWH).sum()
    eff_dyn    = new_rev / total_kwh if total_kwh > 0 else 0
    eff_base   = baseline_price
    eff_impr   = (eff_dyn - eff_base) / eff_base * 100

    results.update(pricing_efficiency_baseline=eff_base, pricing_efficiency_dynamic=eff_dyn,
                   pricing_efficiency_improvement_pct=eff_impr)
    logger.info(f"  Baseline: ₹{eff_base:.2f}/kWh  Dynamic: ₹{eff_dyn:.2f}/kWh  Δ: {eff_impr:+.2f}%")

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"1. Revenue Gain:             {results['revenue_gain_pct']:+.2f}%")
    logger.info(f"2. Utilization Change:        {results['utilization_change_pct']:+.2f}%")
    logger.info(f"3. Off-Peak Uplift:           {results['off_peak_uplift_pct']:+.1f}%")
    logger.info(f"4. Wait Time Reduction:       {results['queue_reduction_pct']:+.1f}%")
    logger.info(f"5. Customer Response Rate:    {results['customer_response_rate']:.2f}%")
    logger.info(f"6. Pricing Efficiency Gain:   {results['pricing_efficiency_improvement_pct']:+.2f}%")

    pd.DataFrame([results]).to_csv("outputs/evaluation_metrics.csv", index=False)
    logger.info("✓ Saved to outputs/evaluation_metrics.csv")
    return results


if __name__ == "__main__":
    evaluate_demand_agent()
    evaluate_all_metrics()
