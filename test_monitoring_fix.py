#!/usr/bin/env python3
"""Quick test to verify monitoring agent JSON parsing fix."""
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from src.config import SystemConfig
from src.data_loader import load_dataset, train_test_split
from src.orchestrator import SystemOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n_hours: int = 50) -> pd.DataFrame:
    """Generate minimal synthetic data."""
    np.random.seed(42)
    hours = []
    
    for i in range(n_hours):
        hour_of_day = i % 24
        is_weekend = (i // 24) % 7 >= 5
        
        base_util = 0.45 + np.random.uniform(-0.15, 0.25)
        if 8 <= hour_of_day <= 18 and not is_weekend:
            base_util += 0.15
        
        hours.append({
            'hour_of_day': hour_of_day,
            'is_weekend': int(is_weekend),
            'acn_total_kwh': np.random.uniform(200, 800),
            'urban_mean_utilization': np.clip(base_util, 0.1, 0.9),
            'urban_peak_queue': max(0, np.random.normal(base_util * 5, 1.5)),
        })
    
    return pd.DataFrame(hours)

def main():
    logger.info("Testing monitoring agent JSON parsing fix...")
    
    # Generate small dataset
    df = generate_synthetic_data(50)
    
    # Initialize config
    config = SystemConfig(
        llm_provider="ollama",
        llm_model="llama3.2:3b",
        max_iterations=10  # Only 10 steps
    )
    
    # Split data
    train_df, test_df = train_test_split(df, config.train_ratio)
    logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # Initialize system
    orch = SystemOrchestrator(config, use_llm=True)
    orch.train_demand_agent(train_df)
    orch.prepare_test_set(test_df)
    
    # Run optimization
    logger.info("Running 10-step optimization...")
    outcomes = orch.run_optimization()
    
    # Check results
    logger.info(f"✓ Completed {len(outcomes)} steps")
    logger.info(f"Price range: {outcomes['p_new'].min():.2f} - {outcomes['p_new'].max():.2f}")
    logger.info(f"Unique prices: {len(outcomes['p_new'].unique())}")
    logger.info(f"Theta changes: epsilon {outcomes['epsilon'].iloc[0]:.4f} → {outcomes['epsilon'].iloc[-1]:.4f}")
    
    # Check for monitoring agent errors
    monitoring_stats = orch.monitoring_agent.get_stats()
    logger.info(f"Monitoring agent stats: {monitoring_stats}")
    
    if monitoring_stats['llm_success'] > 0:
        logger.info("✅ Monitoring agent LLM working!")
    else:
        logger.warning("⚠️ Monitoring agent only used fallback")
    
    return outcomes

if __name__ == "__main__":
    main()
