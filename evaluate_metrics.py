#!/usr/bin/env python3
"""
PROBLEM 7 FIX: Evaluation metrics computed on correct datasets.

Computes all 6 key metrics using proper data sources:
- Revenue Gain % → ACN
- Charger Utilization Rate → UrbanEV
- Off-Peak Uplift → UrbanEV
- Avg Waiting Time Reduction → UrbanEV
- Customer Response Rate → ACN + elasticity
- Pricing Efficiency Score → ACN
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_all_metrics(
    agentic_outcomes_path: str = "outputs/agentic_outcomes.csv",
    unified_base_path: str = "data/processed/unified_analytical_base.csv",
    baseline_price: float = 15.0,
    elasticity: float = 0.25
) -> dict:
    """
    Evaluate all metrics using correct datasets.
    
    Args:
        agentic_outcomes_path: Path to agentic outcomes CSV with dynamic prices
        unified_base_path: Path to unified analytical base with ACN and UrbanEV data
        baseline_price: Baseline tariff (₹15/kWh)
        elasticity: Demand elasticity for customer response rate
    
    Returns:
        Dict with all 6 metrics
    """
    logger.info("=" * 70)
    logger.info("EVALUATION METRICS - CORRECT DATASET ASSIGNMENT")
    logger.info("=" * 70)
    
    # Load data
    outcomes = pd.read_csv(agentic_outcomes_path)
    unified = pd.read_csv(unified_base_path)
    
    # Align outcomes with unified base (assuming same time_step order)
    if 'time_step' in outcomes.columns and 'time_step' in unified.columns:
        merged = outcomes.merge(unified, on='time_step', how='left', suffixes=('_out', '_base'))
    else:
        # Fallback: assume row alignment
        merged = pd.concat([outcomes.reset_index(drop=True), unified.reset_index(drop=True)], axis=1)
    
    logger.info(f"\nData loaded:")
    logger.info(f"  Outcomes: {len(outcomes)} steps")
    logger.info(f"  Unified base: {len(unified)} rows")
    logger.info(f"  Merged: {len(merged)} rows")
    
    results = {}
    
    # =========================
    # METRIC 1: Revenue Gain %
    # =========================
    logger.info("\n[1/6] Computing Revenue Gain % (ACN)...")
    
    # PROBLEM 7a FIX: Apply dynamic tariffs to ACN hourly kWh volumes
    # Extract ACN kWh from unified base (or use acn_total_kwh if in merged)
    if 'acn_total_kwh_base' in merged.columns:
        acn_kwh = merged['acn_total_kwh_base']
    elif 'acn_total_kwh' in merged.columns:
        acn_kwh = merged['acn_total_kwh']
    else:
        logger.error("ACN kWh column not found")
        acn_kwh = pd.Series([0] * len(merged))
    
    # Dynamic prices from outcomes
    dynamic_prices = merged['p_new'] if 'p_new' in merged.columns else outcomes['p_new']
    
    # Calculate revenues
    new_revenue = (dynamic_prices * acn_kwh).sum()
    old_revenue = (baseline_price * acn_kwh).sum()
    revenue_gain_pct = (new_revenue - old_revenue) / old_revenue * 100 if old_revenue > 0 else 0
    
    results['revenue_gain_pct'] = revenue_gain_pct
    results['new_revenue'] = new_revenue
    results['old_revenue'] = old_revenue
    
    logger.info(f"  Old revenue (₹15/kWh): ₹{old_revenue:.2f}")
    logger.info(f"  New revenue (dynamic): ₹{new_revenue:.2f}")
    logger.info(f"  Revenue gain: {revenue_gain_pct:+.2f}%")
    
    # =========================
    # METRIC 2: Charger Utilization Rate
    # =========================
    logger.info("\n[2/6] Computing Charger Utilization Rate (UrbanEV)...")
    
    # PROBLEM 7b FIX: Before = mean UrbanEV utilization
    # After = simulate demand response with elasticity
    if 'urban_mean_utilization_base' in merged.columns:
        util_col = 'urban_mean_utilization_base'
    elif 'urban_mean_utilization' in merged.columns:
        util_col = 'urban_mean_utilization'
    else:
        logger.error("UrbanEV utilization column not found")
        util_col = None
    
    if util_col:
        util_before = merged[util_col].mean()
        
        # Simulate after: higher price reduces demand, lower price increases
        price_delta_pct = (dynamic_prices - baseline_price) / baseline_price
        demand_response = -elasticity * price_delta_pct  # Negative elasticity
        util_after_simulated = merged[util_col] * (1 + demand_response)
        util_after_simulated = util_after_simulated.clip(0, 1.0)  # Cap at 100%
        util_after = util_after_simulated.mean()
        
        results['utilization_before'] = util_before
        results['utilization_after'] = util_after
        results['utilization_change_pct'] = (util_after - util_before) / util_before * 100 if util_before > 0 else 0
        
        logger.info(f"  Before: {util_before:.1%}")
        logger.info(f"  After (simulated with ε={elasticity}): {util_after:.1%}")
        logger.info(f"  Change: {results['utilization_change_pct']:+.2f}%")
    else:
        results['utilization_before'] = 0
        results['utilization_after'] = 0
        results['utilization_change_pct'] = 0
    
    # =========================
    # METRIC 3: Off-Peak Uplift
    # =========================
    logger.info("\n[3/6] Computing Off-Peak Uplift (UrbanEV)...")
    
    # PROBLEM 7c FIX: Count UrbanEV sessions/pile-hours where utilization <30%
    off_peak_threshold = 0.30
    
    if util_col:
        off_peak_mask_before = merged[util_col] < off_peak_threshold
        off_peak_count_before = off_peak_mask_before.sum()
        
        off_peak_mask_after = util_after_simulated < off_peak_threshold
        off_peak_count_after = off_peak_mask_after.sum()
        
        off_peak_uplift = off_peak_count_after - off_peak_count_before
        off_peak_uplift_pct = (off_peak_uplift / off_peak_count_before * 100) if off_peak_count_before > 0 else 0
        
        results['off_peak_before'] = off_peak_count_before
        results['off_peak_after'] = off_peak_count_after
        results['off_peak_uplift'] = off_peak_uplift
        results['off_peak_uplift_pct'] = off_peak_uplift_pct
        
        logger.info(f"  Off-peak timesteps before (<{off_peak_threshold:.0%}): {off_peak_count_before}")
        logger.info(f"  Off-peak timesteps after: {off_peak_count_after}")
        logger.info(f"  Uplift: {off_peak_uplift:+d} ({off_peak_uplift_pct:+.1f}%)")
    else:
        results['off_peak_before'] = 0
        results['off_peak_after'] = 0
        results['off_peak_uplift'] = 0
        results['off_peak_uplift_pct'] = 0
    
    # =========================
    # METRIC 4: Avg Waiting Time Reduction
    # =========================
    logger.info("\n[4/6] Computing Avg Waiting Time Reduction (UrbanEV)...")
    
    # PROBLEM 7d FIX: Use UrbanEV queue_length_proxy at peak hours
    # For now, compute queue proxy as: (utilization - 0.5) * 10 for util > 50%
    if util_col:
        queue_proxy_before = merged[util_col].apply(lambda u: max(0, (u - 0.5) * 10))
        queue_proxy_after = util_after_simulated.apply(lambda u: max(0, (u - 0.5) * 10))
        
        # Focus on peak hours (is_peak_hour == 1)
        if 'is_peak_hour' in merged.columns:
            peak_mask = (merged['is_peak_hour'] == 1).values
        elif 'is_peak_hour_base' in merged.columns:
            peak_mask = (merged['is_peak_hour_base'] == 1).values
        else:
            peak_mask = None
        
        if peak_mask is not None:
            queue_before_peak = queue_proxy_before.iloc[peak_mask].mean()
            queue_after_peak = queue_proxy_after.iloc[peak_mask].mean()
        else:
            queue_before_peak = queue_proxy_before.mean()
            queue_after_peak = queue_proxy_after.mean()
        
        queue_reduction = queue_before_peak - queue_after_peak
        queue_reduction_pct = (queue_reduction / queue_before_peak * 100) if queue_before_peak > 0 else 0
        
        results['queue_before_peak'] = queue_before_peak
        results['queue_after_peak'] = queue_after_peak
        results['queue_reduction'] = queue_reduction
        results['queue_reduction_pct'] = queue_reduction_pct
        
        logger.info(f"  Queue proxy before (peak hours): {queue_before_peak:.2f}")
        logger.info(f"  Queue proxy after (peak hours): {queue_after_peak:.2f}")
        logger.info(f"  Reduction: {queue_reduction:+.2f} ({queue_reduction_pct:+.1f}%)")
    else:
        results['queue_before_peak'] = 0
        results['queue_after_peak'] = 0
        results['queue_reduction'] = 0
        results['queue_reduction_pct'] = 0
    
    # =========================
    # METRIC 5: Customer Response Rate
    # =========================
    logger.info("\n[5/6] Computing Customer Response Rate (ACN + elasticity)...")
    
    # PROBLEM 7e FIX: For each ACN hour, compute demand response
    # delta_sessions = ε * (baseline_price - dynamic_price) / baseline_price * baseline_sessions
    if 'acn_sessions_count' in merged.columns:
        baseline_sessions = merged['acn_sessions_count']
        price_change_pct = (baseline_price - dynamic_prices) / baseline_price
        delta_sessions = elasticity * price_change_pct * baseline_sessions
        
        # Mean absolute % shift in session volume
        session_shift_pct = (delta_sessions.abs() / baseline_sessions * 100).replace([np.inf, -np.inf], 0).fillna(0)
        mean_response_rate = session_shift_pct.mean()
        
        results['customer_response_rate'] = mean_response_rate
        results['sessions_shifted_total'] = delta_sessions.abs().sum()
        
        logger.info(f"  Mean absolute session shift: {mean_response_rate:.2f}%")
        logger.info(f"  Total sessions shifted: {results['sessions_shifted_total']:.0f}")
    else:
        results['customer_response_rate'] = 0
        results['sessions_shifted_total'] = 0
        logger.info("  ACN sessions not found")
    
    # =========================
    # METRIC 6: Pricing Efficiency Score
    # =========================
    logger.info("\n[6/6] Computing Pricing Efficiency Score (ACN)...")
    
    # PROBLEM 7f FIX: ACN total dynamic revenue / ACN total kWh delivered
    total_kwh = acn_kwh.sum()
    pricing_efficiency_dynamic = (new_revenue / total_kwh) if total_kwh > 0 else 0
    pricing_efficiency_baseline = baseline_price  # Baseline is constant ₹15/kWh
    
    efficiency_improvement_pct = (pricing_efficiency_dynamic - pricing_efficiency_baseline) / pricing_efficiency_baseline * 100
    
    results['pricing_efficiency_baseline'] = pricing_efficiency_baseline
    results['pricing_efficiency_dynamic'] = pricing_efficiency_dynamic
    results['pricing_efficiency_improvement_pct'] = efficiency_improvement_pct
    
    logger.info(f"  Baseline: ₹{pricing_efficiency_baseline:.2f}/kWh")
    logger.info(f"  Dynamic: ₹{pricing_efficiency_dynamic:.2f}/kWh")
    logger.info(f"  Improvement: {efficiency_improvement_pct:+.2f}%")
    
    # =========================
    # Summary
    # =========================
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"\n1. Revenue Gain: {results['revenue_gain_pct']:+.2f}%")
    logger.info(f"2. Charger Utilization Change: {results['utilization_change_pct']:+.2f}%")
    logger.info(f"3. Off-Peak Uplift: {results['off_peak_uplift_pct']:+.1f}%")
    logger.info(f"4. Avg Waiting Time Reduction: {results['queue_reduction_pct']:+.1f}%")
    logger.info(f"5. Customer Response Rate: {results['customer_response_rate']:.2f}%")
    logger.info(f"6. Pricing Efficiency Improvement: {results['pricing_efficiency_improvement_pct']:+.2f}%")
    
    # Export results
    results_df = pd.DataFrame([results])
    output_path = "outputs/evaluation_metrics.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"\n✓ Metrics saved to {output_path}")
    
    return results


if __name__ == "__main__":
    results = evaluate_all_metrics()
