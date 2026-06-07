#!/usr/bin/env python3
"""Main entry point for the agentic EV tariff optimization system."""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

from src.config import SystemConfig
from src.data_loader import load_dataset, train_test_split
from src.orchestrator import SystemOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_groq_available() -> bool:
    """Return True if at least one Groq API key is configured."""
    from src.utils.llm_provider import _load_keys_from_env
    keys = _load_keys_from_env()
    if keys:
        logger.info(f"✓ {len(keys)} Groq API key(s) found")
        return True
    logger.warning("⚠ No Groq API keys found — running in deterministic fallback mode")
    logger.warning("⚠ Add GROQ_API_KEY_1 … GROQ_API_KEY_4 to .env")
    return False


def main():
    logger.info("=" * 70)
    logger.info("Agentic EV Tariff Optimization  |  Groq / Llama 3.3 70B")
    logger.info("=" * 70)

    groq_available = check_groq_available()

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("\n[1/7] Loading analytical base...")
    data_path = "data/processed/unified_analytical_base.csv"
    if not Path(data_path).exists():
        logger.error(f"❌ {data_path} not found. Run rebuild_data.py first.")
        return 1

    df = pd.read_csv(data_path)
    logger.info(f"✓ {len(df)} rows loaded")

    # ── Regime check ─────────────────────────────────────────────────────────
    logger.info("\n[1.5/7] Regime distribution...")
    surge_thr    = 0.48
    discount_thr = 0.36
    n_surge    = (df["urban_mean_utilization"] > surge_thr).sum()
    n_discount = (df["urban_mean_utilization"] < discount_thr).sum()
    n_neutral  = len(df) - n_surge - n_discount
    logger.info(f"  Surge (>{surge_thr:.0%}):    {n_surge} ({n_surge/len(df)*100:.1f}%)")
    logger.info(f"  Discount (<{discount_thr:.0%}): {n_discount} ({n_discount/len(df)*100:.1f}%)")
    logger.info(f"  Neutral:           {n_neutral} ({n_neutral/len(df)*100:.1f}%)")

    # ── Config ───────────────────────────────────────────────────────────────
    logger.info("\n[2/7] Loading configuration...")
    config = SystemConfig(
        llm_provider="groq",
        llm_model="llama-3.3-70b-versatile",
        baseline_tariff_per_kwh=15.0,
        pricing_bounds=(10.0, 22.0),
        surge_threshold=surge_thr,
        discount_threshold=discount_thr,
        theta_init=(0.25, 4.0, 4.0),
        random_seed=42,
        max_iterations=168,
        convergence_window=40,
    )
    logger.info(f"✓ baseline=₹{config.baseline_tariff_per_kwh}, θ={config.theta_init}")

    # ── Split ────────────────────────────────────────────────────────────────
    logger.info("\n[3/7] Splitting dataset...")
    train_df, test_df = train_test_split(df, train_ratio=0.60)
    logger.info(f"✓ Train: {len(train_df)}, Test: {len(test_df)}")

    # ── Init ─────────────────────────────────────────────────────────────────
    logger.info("\n[4/7] Initializing agents...")
    orchestrator = SystemOrchestrator(config, use_llm=groq_available)
    llm_label = "Groq LLM + fallback" if groq_available else "fallback only"
    logger.info(f"  • Demand Agent    (XGBoost)")
    logger.info(f"  • Pricing Agent   ({llm_label})")
    logger.info(f"  • Monitoring Agent({llm_label})")

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("\n[5/7] Training Demand Agent...")
    orchestrator.train_demand_agent(train_df)
    logger.info("✓ Done")

    # ── Prepare ───────────────────────────────────────────────────────────────
    logger.info("\n[6/7] Preparing test set...")
    orchestrator.prepare_test_set(test_df)

    # ── Optimize ──────────────────────────────────────────────────────────────
    logger.info("\n[7/7] Running optimization loop...")
    logger.info("-" * 70)

    try:
        outcomes_df = orchestrator.run_optimization()

        Path("outputs").mkdir(parents=True, exist_ok=True)
        output_path = "outputs/agentic_outcomes.csv"
        orchestrator.export_results(outcomes_df, output_path)

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Steps:             {len(outcomes_df)}")
        logger.info(f"Mean revenue gain: {outcomes_df['revenue_gain_pct'].mean():.2f}%")
        logger.info(f"Mean reward:       {outcomes_df['reward'].mean():.2f}")
        logger.info(f"Final θ:  ε={outcomes_df['epsilon'].iloc[-1]:.3f}  "
                    f"α={outcomes_df['alpha'].iloc[-1]:.3f}  "
                    f"β={outcomes_df['beta'].iloc[-1]:.3f}")
        logger.info("Regime distribution:")
        for regime, count in outcomes_df["regime"].value_counts().items():
            logger.info(f"  {regime}: {count} ({count/len(outcomes_df)*100:.1f}%)")
        ps = orchestrator.pricing_agent.get_stats()
        ms = orchestrator.monitoring_agent.get_stats()
        logger.info(f"Pricing LLM:    {ps['llm_success_rate']:.1f}% success")
        logger.info(f"Monitoring LLM: {ms['llm_success_rate']:.1f}% success")
        logger.info(f"\n✓ Results: {output_path}")
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
