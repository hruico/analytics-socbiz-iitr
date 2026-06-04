#!/usr/bin/env python3
"""
Fully Agentic EV Tariff Optimization System.

Uses Ollama (free, local) for LLM-powered agent reasoning.
Falls back to deterministic logic if LLM unavailable.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

from src.config import SystemConfig
from src.data_loader import load_dataset, train_test_split
from src.orchestrator import SystemOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_groq_api_key():
    """Check if Groq API key is available."""
    import os
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        logger.info("✓ GROQ_API_KEY found in environment")
        return True
    else:
        logger.warning("⚠ GROQ_API_KEY not found in environment")
        logger.warning("⚠ Get your free API key from: https://console.groq.com/keys")
        logger.warning("⚠ Set it with: export GROQ_API_KEY='your-key-here'")
        logger.warning("⚠ System will use deterministic fallbacks")
        return False


def generate_synthetic_data(n_hours: int = 200) -> pd.DataFrame:
    """Generate synthetic unified analytical base."""
    logger.info(f"Generating {n_hours} hours of synthetic data...")
    
    np.random.seed(42)
    hours = []
    
    for i in range(n_hours):
        hour_of_day = i % 24
        day_of_week = (i // 24) % 7
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if hour_of_day in [7, 8, 9, 17, 18, 19] else 0
        
        # Simulate realistic patterns
        base_utilization = 0.50
        if is_peak_hour:
            base_utilization += 0.25
        if is_weekend:
            base_utilization -= 0.10
        
        utilization = np.clip(base_utilization + np.random.normal(0, 0.10), 0.1, 0.95)
        sessions = int(utilization * 50) + np.random.randint(-5, 5)
        sessions = max(1, sessions)
        kwh = sessions * (20 + np.random.normal(0, 5))
        kwh = max(10, kwh)
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
            'acn_base_revenue': kwh * 15.0,
            'acn_revenue_per_session': (kwh / sessions) * 15.0,
            'acn_energy_cost_per_kwh': 15.0,
            'urban_mean_utilization': utilization,
            'urban_peak_queue': queue,
            'urban_total_volume': kwh,
            'station_cluster_id': i % 5
        })
    
    return pd.DataFrame(hours)


def main():
    """Run the fully agentic optimization."""
    logger.info("=" * 70)
    logger.info("FULLY AGENTIC EV Tariff Optimization System")
    logger.info("Using Groq (Free Tier) with Llama 3.3 70B")
    logger.info("=" * 70)
    
    # Check Groq API key availability
    groq_available = check_groq_api_key()
    
    # PROBLEM 1 FIX: Load real data instead of generating synthetic
    logger.info("\n[1/7] Loading unified analytical base...")
    # Fixed unified_analytical_base.csv now has realistic utilization (12-100%, mean 58%)
    # Applied diurnal pattern to fix MAX-across-zones aggregation artifact
    unified_path = "data/processed/unified_analytical_base.csv"
    if not Path(unified_path).exists():
        logger.error(f"❌ {unified_path} not found. Run rebuild_data.py first.")
        return 1
    df = pd.read_csv(unified_path)
    logger.info(f"✓ {len(df)} hours of real data loaded from {unified_path}")
    
    # PROBLEM 9 FIX: Verify regime distribution has non-zero surge timesteps
    logger.info("\n[1.5/7] Verifying regime distribution...")
    surge_threshold = 0.80
    discount_threshold = 0.30
    
    surge_count = (df['urban_mean_utilization'] > surge_threshold).sum()
    discount_count = (df['urban_mean_utilization'] < discount_threshold).sum()
    neutral_count = len(df) - surge_count - discount_count
    
    logger.info(f"  Surge (>{surge_threshold:.0%}): {surge_count} timesteps ({surge_count/len(df)*100:.1f}%)")
    logger.info(f"  Discount (<{discount_threshold:.0%}): {discount_count} timesteps ({discount_count/len(df)*100:.1f}%)")
    logger.info(f"  Neutral: {neutral_count} timesteps ({neutral_count/len(df)*100:.1f}%)")
    
    if surge_count == 0:
        p90 = df['urban_mean_utilization'].quantile(0.90)
        p25 = df['urban_mean_utilization'].quantile(0.25)
        logger.warning(f"⚠️  0% surge timesteps with threshold {surge_threshold:.0%}")
        logger.warning(f"⚠️  Suggested thresholds: surge >{p90:.1%} (P90), discount <{p25:.1%} (P25)")
        logger.warning(f"⚠️  Proceeding with 0% surge (no surge pricing will occur)")
    
    target_surge_pct = 10.0
    if surge_count / len(df) * 100 < target_surge_pct:
        logger.warning(f"⚠️  Surge percentage ({surge_count/len(df)*100:.1f}%) below target ({target_surge_pct}%)")
        logger.warning(f"⚠️  Consider lowering surge threshold or using peak-zone utilization")
    
    # Load configuration
    logger.info("\n[2/7] Loading configuration...")
    config = SystemConfig(
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",  # Groq's best free model
        baseline_tariff_per_kwh=15.0,
        pricing_bounds=(10.0, 22.0),
        theta_init=(0.25, 2.5, 2.5),
        random_seed=42,
        max_iterations=40,  # Limited for demo
        convergence_window=20
    )
    logger.info(f"✓ Config loaded: baseline=₹{config.baseline_tariff_per_kwh}, θ={config.theta_init}")
    
    # Split data
    logger.info("\n[3/7] Splitting dataset...")
    train_df, test_df = train_test_split(df, train_ratio=0.80)
    logger.info(f"✓ Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    
    # Initialize orchestrator
    logger.info("\n[4/7] Initializing agentic system...")
    orchestrator = SystemOrchestrator(config, use_llm=groq_available)
    logger.info("✓ Three agents initialized:")
    logger.info("  • Demand Agent (XGBoost) - learns from data")
    logger.info(f"  • Pricing Agent ({'Groq LLM+fallback' if groq_available else 'fallback only'}) - decides tariffs")
    logger.info(f"  • Monitoring Agent ({'Groq LLM+fallback' if groq_available else 'fallback only'}) - adjusts parameters")
    
    # Train demand agent
    logger.info("\n[5/7] Training Demand Agent...")
    orchestrator.train_demand_agent(train_df)
    logger.info("✓ Demand predictions ready")
    
    # Prepare test set
    logger.info("\n[6/7] Preparing test environment...")
    orchestrator.prepare_test_set(test_df)
    logger.info("✓ Test set loaded")
    
    # Run optimization
    logger.info("\n[7/7] Running optimization loop...")
    logger.info("-" * 70)
    
    try:
        outcomes_df = orchestrator.run_optimization()
        
        # Export results
        Path("outputs").mkdir(parents=True, exist_ok=True)
        output_path = "outputs/agentic_outcomes.csv"
        orchestrator.export_results(outcomes_df, output_path)
        
        # Display summary
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nSteps executed: {len(outcomes_df)}")
        logger.info(f"Mean revenue gain: {outcomes_df['revenue_gain_pct'].mean():.2f}%")
        logger.info(f"Mean reward: {outcomes_df['reward'].mean():.2f}")
        logger.info(f"\nFinal parameters:")
        logger.info(f"  ε (elasticity): {outcomes_df['epsilon'].iloc[-1]:.3f}")
        logger.info(f"  α (surge): {outcomes_df['alpha'].iloc[-1]:.3f}")
        logger.info(f"  β (discount): {outcomes_df['beta'].iloc[-1]:.3f}")
        
        logger.info(f"\nRegime distribution:")
        for regime, count in outcomes_df['regime'].value_counts().items():
            logger.info(f"  {regime}: {count} steps ({count/len(outcomes_df)*100:.1f}%)")
        
        logger.info(f"\nAgent performance:")
        pricing_stats = orchestrator.pricing_agent.get_stats()
        monitoring_stats = orchestrator.monitoring_agent.get_stats()
        logger.info(f"  Pricing LLM success rate: {pricing_stats['llm_success_rate']:.1f}%")
        logger.info(f"  Monitoring LLM success rate: {monitoring_stats['llm_success_rate']:.1f}%")
        
        if groq_available:
            llm_stats = orchestrator.llm.get_stats()
            logger.info(f"  Overall LLM success rate: {llm_stats['success_rate']:.1f}%")
        
        logger.info(f"\n✓ Results saved: {output_path}")
        logger.info("=" * 70)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nOptimization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\n\nOptimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
