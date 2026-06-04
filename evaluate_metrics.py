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


def simulate_price_for_row(u_pred: float, baseline: float, alpha: float, beta: float, 
                           lower: float = 10.0, upper: float = 25.0) -> float:
    """
    Simulate dynamic price using learned parameters and deterministic pricing logic.
    Mirrors the PricingAgent._compute_price_from_regime logic.
    
    Args:
        u_pred: Predicted utilization
        baseline: Baseline price (₹15/kWh)
        alpha: Learned surge pricing parameter
        beta: Learned discount pricing parameter
        lower: Lower price bound
        upper: Upper price bound
    
    Returns:
        Simulated dynamic price
    """
    if u_pred > 0.80:
        # Surge regime
        p_new = baseline * (1 + alpha * (u_pred - 0.80))
    elif u_pred < 0.30:
        # Discount regime (using elasticity approximation)
        epsilon_approx = 0.25  # Use default elasticity for discount calculation
        p_new = baseline - (0.30 - u_pred) * beta * epsilon_approx
    else:
        # Neutral regime
        p_new = baseline + (u_pred - 0.55) * 8.0
    
    return np.clip(p_new, lower, upper)


def evaluate_demand_agent_metrics(
    agentic_outcomes_path: str = "outputs/agentic_outcomes.csv"
) -> dict:
    """
    Evaluate Demand Agent prediction accuracy using u_pred vs u_actual.
    
    Args:
        agentic_outcomes_path: Path to agentic outcomes CSV with predictions
    
    Returns:
        Dict with RMSE, MAE, and R² metrics
    """
    logger.info("=" * 70)
    logger.info("DEMAND AGENT METRICS")
    logger.info("=" * 70)
    
    # Load outcomes
    outcomes = pd.read_csv(agentic_outcomes_path)
    
    # Extract predictions and actuals
    u_pred = outcomes['u_pred'].values
    u_actual = outcomes['u_actual'].values
    
    # Compute metrics
    n = len(u_pred)
    
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((u_pred - u_actual) ** 2))
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(u_pred - u_actual))
    
    # R² (Coefficient of Determination)
    ss_res = np.sum((u_actual - u_pred) ** 2)
    ss_tot = np.sum((u_actual - np.mean(u_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_samples': n
    }
    
    logger.info(f"\n[0/6] Demand Agent Prediction Accuracy")
    logger.info(f"  Samples: {n}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    
    return results


def evaluate_all_metrics(
    agentic_outcomes_path: str = "outputs/agentic_outcomes.csv",
    unified_base_path: str = "data/processed/unified_analytical_base.csv",  # Fixed real data
    baseline_price: float = 15.0,
    elasticity: float = 0.25,
    learned_epsilon: float = None,
    learned_alpha: float = None,
    learned_beta: float = None
) -> dict:
    """
    Evaluate all metrics using correct datasets.
    
    Args:
        agentic_outcomes_path: Path to agentic outcomes CSV with dynamic prices
        unified_base_path: Path to unified analytical base with ACN and UrbanEV data
        baseline_price: Baseline tariff (₹15/kWh)
        elasticity: Demand elasticity for customer response rate
        learned_epsilon: Final learned elasticity parameter (if None, extracted from outcomes)
        learned_alpha: Final learned alpha parameter (if None, extracted from outcomes)
        learned_beta: Final learned beta parameter (if None, extracted from outcomes)
    
    Returns:
        Dict with all 6 metrics
    """
    logger.info("=" * 70)
    logger.info("EVALUATION METRICS - CORRECT DATASET ASSIGNMENT")
    logger.info("=" * 70)
    
    # Load data
    outcomes = pd.read_csv(agentic_outcomes_path)
    unified = pd.read_csv(unified_base_path)
    
    # Extract final learned parameters from last row of outcomes if not provided
    if learned_epsilon is None or learned_alpha is None or learned_beta is None:
        last_row = outcomes.iloc[-1]
        learned_epsilon = last_row['epsilon_est'] if 'epsilon_est' in outcomes.columns else 0.232
        learned_alpha = last_row['alpha_est'] if 'alpha_est' in outcomes.columns else 2.569
        learned_beta = last_row['beta_est'] if 'beta_est' in outcomes.columns else 2.480
    
    logger.info(f"\nLearned parameters (final):")
    logger.info(f"  ε = {learned_epsilon:.3f}")
    logger.info(f"  α = {learned_alpha:.3f}")
    logger.info(f"  β = {learned_beta:.3f}")
    
    # TASK 1 FIX: Merge on step/time_step to get only test rows
    # outcomes has 'step', unified has 'time_step'
    merged = outcomes.merge(unified, left_on='step', right_on='time_step', how='inner', suffixes=('', '_base'))
    
    logger.info(f"\nData loaded:")
    logger.info(f"  Outcomes: {len(outcomes)} steps")
    logger.info(f"  Unified base (full dataset): {len(unified)} rows")
    logger.info(f"  Merged (test set only): {len(merged)} rows")
    
    results = {}
    
    # =========================
    # METRIC 1: Revenue Gain %
    # =========================
    logger.info("\n[1/6] Computing Revenue Gain % (ACN)...")
    
    # TASK 1: Extract ACN kWh and dynamic prices from merged test rows
    acn_kwh = merged['acn_total_kwh']
    dynamic_prices = merged['p_new']
    
    # TASK 1: Calculate revenues over same 34 test rows
    old_revenue = (acn_kwh * baseline_price).sum()
    new_revenue = (acn_kwh * dynamic_prices).sum()
    revenue_gain_pct = (new_revenue - old_revenue) / old_revenue * 100 if old_revenue > 0 else 0
    
    results['revenue_gain_pct'] = revenue_gain_pct
    results['new_revenue'] = new_revenue
    results['old_revenue'] = old_revenue
    
    logger.info(f"  Test set rows: {len(merged)}")
    logger.info(f"  Total kWh (test set): {acn_kwh.sum():.2f}")
    logger.info(f"  Old revenue (₹15/kWh, test set only): ₹{old_revenue:.2f}")
    logger.info(f"  New revenue (dynamic, test set only): ₹{new_revenue:.2f}")
    logger.info(f"  Revenue gain: {revenue_gain_pct:+.2f}%")
    
    # =========================
    # METRIC 2: Charger Utilization Rate
    # =========================
    logger.info("\n[2/6] Computing Charger Utilization Rate (UrbanEV)...")
    
    # TASK 2 FIX: Use actual dynamic price per row with elasticity formula
    # u_after = u_before * (1 - ε * (price - 15) / 15) per row, then average
    if 'urban_mean_utilization_base' in merged.columns:
        util_col = 'urban_mean_utilization_base'
    elif 'urban_mean_utilization' in merged.columns:
        util_col = 'urban_mean_utilization'
    else:
        logger.error("UrbanEV utilization column not found")
        util_col = None
    
    if util_col:
        util_before = merged[util_col].mean()
        
        # Apply per-row elasticity formula using actual dynamic prices
        util_after_simulated = merged[util_col] * (1 - elasticity * (dynamic_prices - baseline_price) / baseline_price)
        util_after_simulated = util_after_simulated.clip(0, 1.0)  # Cap at 100%
        util_after = util_after_simulated.mean()
        
        results['utilization_before'] = util_before
        results['utilization_after'] = util_after
        results['utilization_change_pct'] = (util_after - util_before) / util_before * 100 if util_before > 0 else 0
        
        logger.info(f"  Before: {util_before:.1%}")
        logger.info(f"  After (using actual dynamic prices per row, ε={elasticity}): {util_after:.1%}")
        logger.info(f"  Change: {results['utilization_change_pct']:+.2f}%")
    else:
        results['utilization_before'] = 0
        results['utilization_after'] = 0
        results['utilization_change_pct'] = 0
    
    # =========================
    # METRIC 3: Off-Peak Uplift
    # =========================
    logger.info("\n[3/6] Computing Off-Peak Uplift (UrbanEV - ALL 168 ROWS)...")
    
    # TASK 3 FIX: Compute over all 168 rows using simulated prices
    off_peak_threshold = 0.30
    
    # Determine utilization column in full unified dataset
    if 'urban_mean_utilization' in unified.columns:
        util_col_full = 'urban_mean_utilization'
    else:
        logger.error("UrbanEV utilization column not found in unified dataset")
        util_col_full = None
    
    if util_col_full:
        # Before: baseline state across all 168 rows
        util_before_full = unified[util_col_full]
        
        # Simulate dynamic prices for all 168 rows
        simulated_prices_full = unified[util_col_full].apply(
            lambda u: simulate_price_for_row(u, baseline_price, learned_alpha, learned_beta)
        )
        
        # After: apply elasticity formula per row using simulated prices
        util_after_full = util_before_full * (1 - learned_epsilon * (simulated_prices_full - baseline_price) / baseline_price)
        util_after_full = util_after_full.clip(0, 1.0)
        
        # Count off-peak timesteps
        off_peak_mask_before = util_before_full < off_peak_threshold
        off_peak_count_before = off_peak_mask_before.sum()
        
        off_peak_mask_after = util_after_full < off_peak_threshold
        off_peak_count_after = off_peak_mask_after.sum()
        
        off_peak_uplift = off_peak_count_after - off_peak_count_before
        off_peak_uplift_pct = (off_peak_uplift / off_peak_count_before * 100) if off_peak_count_before > 0 else 0
        
        results['off_peak_before'] = off_peak_count_before
        results['off_peak_after'] = off_peak_count_after
        results['off_peak_uplift'] = off_peak_uplift
        results['off_peak_uplift_pct'] = off_peak_uplift_pct
        
        logger.info(f"  Dataset: All {len(unified)} rows")
        logger.info(f"  Off-peak timesteps before (<{off_peak_threshold:.0%}): {off_peak_count_before}")
        logger.info(f"  Off-peak timesteps after (simulated): {off_peak_count_after}")
        logger.info(f"  Uplift: {off_peak_uplift:+d} ({off_peak_uplift_pct:+.1f}%)")
    else:
        results['off_peak_before'] = 0
        results['off_peak_after'] = 0
        results['off_peak_uplift'] = 0
        results['off_peak_uplift_pct'] = 0
    
    # =========================
    # METRIC 4: Avg Waiting Time Reduction
    # =========================
    logger.info("\n[4/6] Computing Avg Waiting Time Reduction (UrbanEV - ALL 168 ROWS)...")
    
    # TASK 4 FIX: Compute over all 168 rows using simulated prices
    if util_col_full:
        # Queue proxy: (utilization - 0.5) * 10 for util > 50%
        queue_proxy_before_full = util_before_full.apply(lambda u: max(0, (u - 0.5) * 10))
        queue_proxy_after_full = util_after_full.apply(lambda u: max(0, (u - 0.5) * 10))
        
        # Focus on peak hours
        if 'is_peak_hour' in unified.columns:
            peak_mask = (unified['is_peak_hour'] == 1).values
        else:
            peak_mask = None
        
        if peak_mask is not None and peak_mask.sum() > 0:
            queue_before_peak = queue_proxy_before_full[peak_mask].mean()
            queue_after_peak = queue_proxy_after_full[peak_mask].mean()
            peak_count = peak_mask.sum()
        else:
            queue_before_peak = queue_proxy_before_full.mean()
            queue_after_peak = queue_proxy_after_full.mean()
            peak_count = len(unified)
        
        queue_reduction = queue_before_peak - queue_after_peak
        queue_reduction_pct = (queue_reduction / queue_before_peak * 100) if queue_before_peak > 0 else 0
        
        results['queue_before_peak'] = queue_before_peak
        results['queue_after_peak'] = queue_after_peak
        results['queue_reduction'] = queue_reduction
        results['queue_reduction_pct'] = queue_reduction_pct
        
        logger.info(f"  Dataset: All {len(unified)} rows ({peak_count} peak hours)")
        logger.info(f"  Queue proxy before (peak hours): {queue_before_peak:.2f}")
        logger.info(f"  Queue proxy after (peak hours, simulated): {queue_after_peak:.2f}")
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
    # First evaluate Demand Agent metrics
    demand_results = evaluate_demand_agent_metrics()
    
    # Then evaluate all other metrics
    results = evaluate_all_metrics()
