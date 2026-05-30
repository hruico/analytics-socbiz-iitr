# =============================================================================
# orchestrator.py — AgenticOrchestrator CLI entry point
#
# Drives the closed three-agent feedback loop:
#   DemandPredictionAgent → TariffPricingAgent → MonitoringLearningAgent
#
# Usage:
#   python orchestrator.py --csv data/processed/unified_analytical_base.csv \
#       --steps 100 --verbose 10
# =============================================================================

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from src.config import (
    PROCESSED_BASE_PATH,
    OUTPUTS_DIR,
    EDA_OUTPUTS_DIR,
    EPSILON_INIT,
)
from src.agents.demand import DemandPredictionAgent
from src.agents.pricing import TariffPricingAgent
from src.agents.monitoring import MonitoringLearningAgent
from src.utils.logging_utils import configure_logging, log_dependency_versions

logger = logging.getLogger("ev_agentic.orchestrator")


class AgenticOrchestrator:
    """
    Drives the closed three-agent feedback loop over the test set.

    Parameters
    ----------
    csv_path : str
        Path to unified_analytical_base.csv.
    epsilon_init, alpha_init, beta_init : float
        Initial theta values for TariffPricingAgent.
    lr : float
        Initial learning rate for MonitoringLearningAgent.
    lr_decay : float
        Learning rate decay coefficient.
    api_delay : float
        Seconds to sleep between consecutive Gemini API calls.
    use_lightgbm : bool
        Whether to also train a LightGBM backend in DemandPredictionAgent.
    """

    def __init__(
        self,
        csv_path: str = PROCESSED_BASE_PATH,
        epsilon_init: float = EPSILON_INIT,
        alpha_init: float = 4.0,
        beta_init: float = 4.0,
        lr: float = 0.8,
        lr_decay: float = 0.002,
        api_delay: float = 1.0,
        use_lightgbm: bool = False,
    ) -> None:
        # ── Validate CSV exists ───────────────────────────────────────────────
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"Input CSV not found: '{csv_path}'\n"
                "Run the preprocessing pipeline first:\n"
                "  python -m src.pipeline.preprocess"
            )

        # ── Validate GROQ_API_KEY ─────────────────────────────────────────────
        import os
        if not os.environ.get("GROQ_API_KEY", ""):
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Export it before running:\n"
                "  export GROQ_API_KEY='your-groq-key-here'\n"
                "Get a free key at: https://console.groq.com"
            )

        self._csv_path = csv_path
        self._api_delay = api_delay

        # ── Create output directories ─────────────────────────────────────────
        Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)
        Path(EDA_OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

        # ── Construct agents ──────────────────────────────────────────────────
        logger.info("Initialising DemandPredictionAgent …")
        self.demand_agent = DemandPredictionAgent(
            csv_path=csv_path,
            use_lightgbm=use_lightgbm,
        )

        logger.info("Initialising TariffPricingAgent …")
        self.pricing_agent = TariffPricingAgent(
            epsilon_init=epsilon_init,
            alpha_init=alpha_init,
            beta_init=beta_init,
        )

        logger.info("Initialising MonitoringLearningAgent …")
        self.monitor_agent = MonitoringLearningAgent(
            pricing_agent=self.pricing_agent,
            lr=lr,
            lr_decay=lr_decay,
        )

        self._use_lightgbm = use_lightgbm
        logger.info("AgenticOrchestrator ready. Test-set rows: %d", len(self.demand_agent))

    def run(
        self,
        max_steps: int | None = None,
        verbose_every: int = 10,
    ) -> pd.DataFrame:
        """
        Execute the three-agent feedback loop.

        Parameters
        ----------
        max_steps : int or None
            Number of steps to run. Defaults to the full test set.
        verbose_every : int
            Log a progress line every N steps.

        Returns
        -------
        pd.DataFrame
            Episode summary from MonitoringLearningAgent.

        Requirements: 7.1, 7.7
        """
        n_steps = min(max_steps or len(self.demand_agent), len(self.demand_agent))
        logger.info("Starting agentic loop: %d steps", n_steps)

        for i in range(n_steps):
            # Step 1: Demand prediction
            state = self.demand_agent.predict_state(i)

            # Step 2: Tariff pricing (Gemini call 1)
            decision = self.pricing_agent.compute_tariff(state)

            # Step 3: Monitoring and learning (Gemini call 2 → Δθ)
            update = self.monitor_agent.step(state, decision)

            if verbose_every > 0 and (i + 1) % verbose_every == 0:
                logger.info(
                    "Step %4d/%d | regime=%-8s p=₹%5.2f | "
                    "rev_gain=%+6.2f%% | reward=%+.4f",
                    i + 1, n_steps,
                    decision.regime, decision.p_new,
                    update.revenue_gain_pct, update.reward,
                )

            # Rate-limit guard between Gemini calls
            time.sleep(self._api_delay)

        df = self.monitor_agent.summary()
        logger.info("Agentic loop complete. %d steps logged.", len(df))
        return df

    def _print_final_report(self, df: pd.DataFrame) -> None:
        """
        Print the final OP'26 evaluation report to stdout.

        Requirements: 7.3
        """
        print("\n" + "=" * 65)
        print("  OP'26 — FINAL EVALUATION REPORT")
        print("=" * 65)

        # Demand prediction metrics
        metrics = self.demand_agent.evaluation_metrics()
        print("\n── Demand Prediction (XGBoost) ──────────────────────────────")
        for target, m in metrics.items():
            print(f"  {target}")
            print(f"    RMSE : {m['RMSE']:.6f}")
            print(f"    MAE  : {m['MAE']:.6f}")
            print(f"    R²   : {m['R2']:.6f}")

        if df.empty:
            print("\n  No episode data available.")
            print("=" * 65)
            return

        # Revenue and pricing metrics
        print("\n── Revenue & Pricing ────────────────────────────────────────")
        print(f"  Revenue Gain %          : {df['revenue_gain_pct'].mean():+.2f}%")
        print(f"  Pricing Efficiency      : Rs{df['pricing_efficiency'].mean():.2f}/kWh")

        # Regime distribution
        regime_counts = df['regime'].value_counts()
        total = len(df)
        print(f"  Surge steps             : {regime_counts.get('surge', 0)} ({regime_counts.get('surge', 0)/total*100:.1f}%)")
        print(f"  Neutral steps           : {regime_counts.get('neutral', 0)} ({regime_counts.get('neutral', 0)/total*100:.1f}%)")
        print(f"  Discount steps          : {regime_counts.get('discount', 0)} ({regime_counts.get('discount', 0)/total*100:.1f}%)")

        # Charger utilisation
        print("\n── Charger Utilisation ──────────────────────────────────────")
        print(f"  Mean Charger Utilisation: {df['charger_utilisation'].mean():.4f}")

        # Off-peak uplift
        try:
            baseline_df = pd.read_csv(self._csv_path)
            uplift = self.monitor_agent.off_peak_uplift(baseline_df)
            print(f"  Off_Peak_Uplift         : {uplift:+.2f}%")
        except Exception:
            print("  Off_Peak_Uplift         : N/A")

        # Wait reduction
        print("\n── Queue & Wait ─────────────────────────────────────────────")
        print(f"  Avg Wait Reduction      : {df['avg_wait_reduction'].mean():.4f}")

        # Customer Response Rate (demand elasticity proxy — required metric)
        print("\n── Customer Response (Demand Elasticity) ────────────────────")
        if 'customer_response_rate' in df.columns:
            crr = df['customer_response_rate'].mean()
            print(f"  Customer Response Rate  : {crr:+.2f}%")
            print(f"  (negative = demand fell due to surge pricing)")

        # Reward convergence
        print("\n── Learning ─────────────────────────────────────────────────")
        print(f"  Mean Reward             : {df['reward'].mean():+.4f}")
        print(f"  Final epsilon           : {df['epsilon_after'].iloc[-1]:.4f}")
        print(f"  Final alpha             : {df['alpha_after'].iloc[-1]:.4f}")
        print(f"  Final beta              : {df['beta_after'].iloc[-1]:.4f}")

        print("=" * 65 + "\n")

    def export(self, path: str = f"{OUTPUTS_DIR}/agentic_outcomes.csv") -> None:
        """
        Export episode outcomes to CSV.

        Requirements: 7.4
        """
        df = self.monitor_agent.summary()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Agentic outcomes exported → %s  (%d rows)", path, len(df))

    def export_predictions(
        self, path: str = f"{OUTPUTS_DIR}/predictions.csv"
    ) -> None:
        """
        Export actual vs predicted demand values for the full test set.

        Uses a single batched model.predict() call for efficiency.

        Requirements: 7.5
        """
        agent = self.demand_agent
        test = agent.test_df

        # Batch predict — one call instead of len(test) calls
        X_test = test[agent.feature_cols].values
        preds = agent.model.predict(X_test)  # shape (n, 2)

        df = pd.DataFrame({
            "hourly_timestamp": test.get("hourly_timestamp", pd.RangeIndex(len(test))).astype(str),
            "hour_of_day": test["hour_of_day"].astype(int),
            "is_weekend": test["is_weekend"].astype(int),
            "actual_urban_mean_utilization": test["urban_mean_utilization"].astype(float),
            "actual_urban_peak_queue": test["urban_peak_queue"].astype(float),
            "pred_urban_mean_utilization": preds[:, 0].clip(0.0, 1.0).astype(float),
            "pred_urban_peak_queue": preds[:, 1].clip(0.0).astype(float),  # floor negatives
        })
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Predictions exported → %s  (%d rows)", path, len(df))

    def export_model_comparison(
        self, path: str = f"{OUTPUTS_DIR}/model_comparison.csv"
    ) -> None:
        """
        Export XGBoost vs LightGBM comparison (when LightGBM is enabled).

        Requirements: 8.5
        """
        if not self._use_lightgbm:
            logger.warning("export_model_comparison: LightGBM not enabled — skipping.")
            return
        df = self.demand_agent.compare_backends()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Model comparison exported → %s", path)

    def export_sensitivity_analysis(
        self,
        path: str = f"{OUTPUTS_DIR}/sensitivity_analysis.csv",
        epsilon_values: list[float] | None = None,
    ) -> None:
        """
        Export sensitivity analysis sweeping ε ∈ {0.5, 1.0, 1.5, 2.0}.

        Requirements: 8.6
        """
        if epsilon_values is None:
            epsilon_values = [0.5, 1.0, 1.5, 2.0]
        df = self.pricing_agent.run_sensitivity_analysis(
            self.demand_agent.test_df, epsilon_values
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Sensitivity analysis exported → %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse all CLI arguments.

    Requirements: 7.2
    """
    parser = argparse.ArgumentParser(
        description="OP'26 Agentic EV Charging Tariff Optimisation Orchestrator"
    )
    parser.add_argument(
        "--csv", default=PROCESSED_BASE_PATH,
        help="Path to unified_analytical_base.csv",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Number of episode steps (default: full test set)",
    )
    parser.add_argument(
        "--verbose", type=int, default=10,
        help="Log progress every N steps",
    )
    parser.add_argument(
        "--lr", type=float, default=0.8,
        help="Initial learning rate η₀",
    )
    parser.add_argument(
        "--decay", type=float, default=0.002,
        help="Learning rate decay coefficient",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds between consecutive Gemini API calls",
    )
    parser.add_argument(
        "--epsilon", type=float, default=EPSILON_INIT,
        help="Initial price-elasticity parameter epsilon (default: 0.3 — inelastic EV demand)",
    )
    parser.add_argument(
        "--alpha", type=float, default=4.0,
        help="Initial surge-sensitivity parameter α",
    )
    parser.add_argument(
        "--beta", type=float, default=4.0,
        help="Initial discount-sensitivity parameter β",
    )
    parser.add_argument(
        "--out", default=f"{OUTPUTS_DIR}/agentic_outcomes.csv",
        help="Output path for agentic_outcomes.csv",
    )
    parser.add_argument(
        "--predictions", default=f"{OUTPUTS_DIR}/predictions.csv",
        help="Output path for predictions.csv",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--lightgbm", action="store_true",
        help="Also train a LightGBM backend for comparison",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    configure_logging(args.log_level)
    log_dependency_versions()

    orchestrator = AgenticOrchestrator(
        csv_path=args.csv,
        epsilon_init=args.epsilon,
        alpha_init=args.alpha,
        beta_init=args.beta,
        lr=args.lr,
        lr_decay=args.decay,
        api_delay=args.delay,
        use_lightgbm=args.lightgbm,
    )

    df = orchestrator.run(max_steps=args.steps, verbose_every=args.verbose)

    orchestrator._print_final_report(df)
    orchestrator.export(args.out)
    orchestrator.export_predictions(args.predictions)
    orchestrator.export_sensitivity_analysis()

    if args.lightgbm:
        orchestrator.export_model_comparison()


if __name__ == "__main__":
    main()
