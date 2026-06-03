#!/usr/bin/env python3
"""
Prototype runner for EV Tariff Optimization System.

This script demonstrates the full agentic workflow using synthetic data.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.config import SystemConfig
from src.data_loader import load_dataset, train_test_split
from src.orchestrator import SystemOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_hours: int = 200) -> pd.DataFrame:
    """
    Generate synthetic unified analytical base for testing.
    
    Args:
        n_hours: Number of hourly records to generate
        
    Returns:
        DataFrame with required schema
    """
    logger.info(f"Generating {n_hours} hours of synthetic data...")
    
    np.random.seed(42)
    
    hours = []
    for i in range(n_hours):
        hour_of_day = i % 24
        day_of_week = (i // 24) % 7
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if hour_of_day in [7, 8, 9, 17, 18, 19] else 0
        
        # Simulate patterns
        base_utilization = 0.50
        if is_peak_hour:
            base_utilization += 0.25
        if is_weekend:
            base_utilization -= 0.10
        
        # Add noise
        utilization = np.clip(base_utilization + np.random.normal(0, 0.10), 0.1, 0.95)
        
        # Sessions correlated with utilization
        sessions = int(utilization * 50) + np.random.randint(-5, 5)
        sessions = max(1, sessions)
        
        # kWh delivered
        kwh = sessions * (20 + np.random.normal(0, 5))
        kwh = max(10, kwh)
        
        # Queue increases with utilization
        queue = max(0, (utilization - 0.5) * 10 + np.random.normal(0, 1))
        
        hours.append({
            'time_step': i,
            'hourly_timestamp': f"2024-01-{1 + i//24:02d}T{hour_of_day:02d}:00:00Z",
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak_hour,
            'acn_sessions_count': sessions,
            'acn_total_kwh': kwh,
            'acn_avg_kwh_per_session': kwh / sessions,
            'acn_base_revenue': kwh * 15.0,  # baseline tariff
            'acn_revenue_per_session': (kwh / sessions) * 15.0,
            'acn_energy_cost_per_kwh': 15.0,
            'urban_mean_utilization': utilization,
            'urban_peak_queue': queue,
            'urban_total_volume': kwh,
            'station_cluster_id': i % 5  # 5 clusters
        })
    
    df = pd.DataFrame(hours)
    logger.info(f"Generated synthetic data: {len(df)} rows")
    return df


def main():
    """Run the prototype optimization."""
    logger.info("=== EV Tariff Optimization Prototype ===")
    
    # Step 1: Generate synthetic data
    df = generate_synthetic_data(n_hours=200)
    
    # Ensure output directory exists
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/processed/synthetic_unified.csv", index=False)
    logger.info("Synthetic data saved to data/processed/synthetic_unified.csv")
    
    # Step 2: Load configuration
    config = SystemConfig(
        llm_provider="openai",
        baseline_tariff_per_kwh=15.0,
        pricing_bounds=(10.0, 22.0),
        theta_init=(1.5, 2.5, 2.5),
        random_seed=42,
        max_iterations=100,  # Limited for prototype
        convergence_window=20  # Faster convergence for demo
    )
    logger.info(f"Configuration loaded: baseline={config.baseline_tariff_per_kwh}, theta={config.theta_init}")
    
    # Step 3: Split data
    train_df, test_df = train_test_split(df, train_ratio=0.80)
    logger.info(f"Data split: train={len(train_df)}, test={len(test_df)}")
    
    # Step 4: Initialize orchestrator
    orchestrator = SystemOrchestrator(config)
    
    # Step 5: Train demand agent
    orchestrator.train_demand_agent(train_df)
    
    # Step 6: Prepare test set
    orchestrator.prepare_test_set(test_df)
    
    # Step 7: Run optimization loop
    logger.info("\n" + "="*60)
    logger.info("Starting Optimization Loop")
    logger.info("="*60 + "\n")
    
    outcomes_df = orchestrator.run_optimization()
    
    # Step 8: Export results
    Path("outputs").mkdir(parents=True, exist_ok=True)
    output_path = "outputs/agentic_outcomes.csv"
    orchestrator.export_results(outcomes_df, output_path)
    
    # Step 9: Display summary
    logger.info("\n" + "="*60)
    logger.info("Optimization Summary")
    logger.info("="*60)
    logger.info(f"Total steps: {len(outcomes_df)}")
    logger.info(f"Mean revenue gain: {outcomes_df['revenue_gain_pct'].mean():.2f}%")
    logger.info(f"Mean reward: {outcomes_df['reward'].mean():.2f}")
    logger.info(f"Final theta: ε={outcomes_df['epsilon'].iloc[-1]:.3f}, "
                f"α={outcomes_df['alpha'].iloc[-1]:.3f}, β={outcomes_df['beta'].iloc[-1]:.3f}")
    logger.info(f"Regime distribution:")
    logger.info(outcomes_df['regime'].value_counts())
    logger.info(f"\nResults saved to {output_path}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
