"""System orchestrator for optimization loop with LLM agents."""
import pandas as pd
import numpy as np
from typing import List, Optional
import logging

from src.config import SystemConfig
from src.agents.demand import DemandAgent
from src.agents.pricing import PricingAgent
from src.agents.monitoring import MonitoringAgent, StepMetrics
from src.utils.metrics import MetricsEngine
from src.utils.convergence import ConvergenceChecker
from src.utils.llm_provider import LLMProviderWrapper

logger = logging.getLogger(__name__)


class SystemOrchestrator:
    """Orchestrates the three-agent optimization loop with LLM integration."""
    
    def __init__(self, config: SystemConfig, use_llm: bool = True):
        self.config = config
        self.use_llm = use_llm
        
        # Initialize LLM provider if enabled
        self.llm = None
        if use_llm:
            try:
                self.llm = LLMProviderWrapper(
                    provider=config.llm_provider,
                    model=config.llm_model,
                    max_retries=2,
                    timeout=30
                )
                logger.info(f"LLM provider initialized ({config.llm_provider}/{config.llm_model})")
            except Exception as e:
                logger.warning(f"LLM initialization failed: {e}. Using fallback mode.")
                self.llm = None
        
        self.demand_agent = DemandAgent(config.random_seed)
        self.pricing_agent = None  # Initialized after training
        self.monitoring_agent = MonitoringAgent(llm_provider=self.llm)
        self.metrics_engine = MetricsEngine(config.reward_weights)
        self.convergence_checker = ConvergenceChecker(config)
        
        self.test_df = None
        self.outcomes = []
        self.history = []
    
    def train_demand_agent(self, train_df: pd.DataFrame):
        """Train the demand prediction model."""
        logger.info("Training demand agent...")
        metrics = self.demand_agent.train(train_df)
        logger.info(f"Demand agent trained: {metrics}")
    
    def prepare_test_set(self, test_df: pd.DataFrame):
        """Prepare test dataset and initialize pricing agent with LLM."""
        self.test_df = test_df
        
        # Initialize pricing agent with LLM support
        self.pricing_agent = PricingAgent(
            baseline=self.config.baseline_tariff_per_kwh,
            bounds=self.config.pricing_bounds,
            theta=np.array(self.config.theta_init),
            llm_provider=self.llm
        )
        
        # Set baseline queue for convergence
        # PROBLEM 2c FIX: Queue derived from UrbanEV utilization if not available
        if 'urban_peak_queue' in test_df.columns:
            baseline_queue = test_df['urban_peak_queue'].mean()
        else:
            # Compute queue proxy from utilization: queue ≈ 10 * (util - 0.5) when > 50%
            queue_proxy = test_df['urban_mean_utilization'].apply(lambda u: max(0, 10 * (u - 0.5)))
            baseline_queue = queue_proxy.mean()
            logger.info(f"urban_peak_queue not in data, using utilization-based proxy (mean={baseline_queue:.2f})")
        
        self.convergence_checker.set_baseline_queue(baseline_queue)
        
        logger.info(f"Test set prepared: {len(test_df)} rows, LLM={'enabled' if self.llm else 'disabled'}")
    
    def run_optimization(self) -> pd.DataFrame:
        """Run the optimization loop."""
        logger.info("Starting optimization loop...")
        
        # Predict demand for all test rows
        u_pred, q_pred, congestion_prob = self.demand_agent.predict(self.test_df)
        
        for step in range(min(len(self.test_df), self.config.max_iterations)):
            row = self.test_df.iloc[step]
            
            # Fix queue actual fallback: use q_pred when data is zero or missing
            if 'urban_peak_queue' in row.index:
                q_actual_raw = row['urban_peak_queue']
                q_actual = q_actual_raw if q_actual_raw > 0 else q_pred[step]
            else:
                # PROBLEM 2c FIX: Compute queue proxy from utilization
                q_actual = max(0, 10 * (row['urban_mean_utilization'] - 0.5))
            
            # Get pricing decision
            decision = self.pricing_agent.compute_tariff(
                u_pred[step], q_pred[step],
                row['hour_of_day'], row['is_weekend'], congestion_prob[step]
            )
            
            # Compute metrics
            metrics = self.metrics_engine.compute_step_metrics(
                p_new=decision.p_new,
                kwh=row['acn_total_kwh'],
                u_actual=row['urban_mean_utilization'],
                q_actual=q_actual,  # Use fixed q_actual
                epsilon=self.pricing_agent.theta[0],
                baseline=self.config.baseline_tariff_per_kwh,
                q_baseline_mean=self.convergence_checker.baseline_queue_mean,
                regime=decision.regime  # Pass regime for safety bounds
            )
            
            # Update convergence checker
            self.convergence_checker.update(
                metrics['revenue_gain_pct'],
                self.pricing_agent.theta,
                metrics['utilization_new'],
                q_actual  # Use fixed q_actual
            )
            
            # Check convergence
            conv_result = self.convergence_checker.check()
            
            # Store outcome
            self.outcomes.append({
                'step': step,
                'regime': decision.regime,
                'p_new': decision.p_new,
                'u_pred': u_pred[step],
                'u_actual': row['urban_mean_utilization'],
                'q_pred': q_pred[step],
                'q_actual': q_actual,  # Use fixed q_actual
                **metrics,
                'epsilon': self.pricing_agent.theta[0],
                'alpha': self.pricing_agent.theta[1],
                'beta': self.pricing_agent.theta[2],
                'fallback_used': decision.fallback_used,
                # PROBLEM 10 FIX: Add enhanced columns
                'data_source': 'unified',  # 'ACN' for revenue metrics, 'UrbanEV' for utilization
                'utilization_value': row['urban_mean_utilization'],
                'price_applied': decision.p_new,
                'revenue_this_step': metrics['revenue_new'],
                'hour_of_day': row['hour_of_day'],
                'is_peak_hour': row['is_peak_hour']
            })
            
            # Build history for monitoring agent
            step_metric = StepMetrics(
                step=step,
                regime=decision.regime,
                revenue_gain_pct=metrics['revenue_gain_pct'],
                u_actual=row['urban_mean_utilization'],
                u_pred=u_pred[step]
            )
            self.history.append(step_metric)
            
            # Get parameter update
            if len(self.history) >= 3:
                # FIX 4: Rate limit management - add 1.5 second sleep between pricing and monitoring LLM calls
                # At 2 calls/step × 40 steps = 80 calls with 1.5s spacing → ~53 calls/min (under 60 req/min limit)
                import time
                if self.llm:
                    time.sleep(1.5)
                
                update = self.monitoring_agent.evaluate_and_propose(
                    step, metrics['revenue_gain_pct'],
                    row['urban_mean_utilization'], u_pred[step],
                    decision.regime, self.history[-5:],
                    self.pricing_agent.theta
                )
                
                # Apply learning rate decay
                eta = self.config.learning_rate_init / (1 + self.config.learning_rate_decay * step)
                delta = np.array([update.delta_epsilon, update.delta_alpha, update.delta_beta])
                
                # Log delta and eta for debugging
                logger.info(f"Step {step}: eta={eta:.6f}, delta_before_scaling={delta}, delta_after_scaling={eta * delta}")
                
                self.pricing_agent.apply_update(eta * delta)
            
            # Check if converged
            if conv_result['converged']:
                logger.info(f"Converged at step {step}")
                break
            
            # PROBLEM 5 FIX: Log reward decomposition every 10 steps
            if (step + 1) % 10 == 0:
                logger.info(f"Step {step + 1}: revenue_gain={metrics['revenue_gain_pct']:.2f}%, "
                           f"reward={metrics['reward']:.2f} "
                           f"[rev:{metrics['reward_revenue_component']:.1f}, "
                           f"util:{metrics['reward_utilization_component']:.1f}, "
                           f"cong:{metrics['reward_congestion_component']:.1f}]")
        
        outcomes_df = pd.DataFrame(self.outcomes)
        logger.info(f"Optimization complete: {len(outcomes_df)} steps")
        
        # Log agent statistics
        if self.llm:
            logger.info(f"LLM stats: {self.llm.get_stats()}")
        logger.info(f"Pricing agent: {self.pricing_agent.get_stats()}")
        logger.info(f"Monitoring agent: {self.monitoring_agent.get_stats()}")
        
        return outcomes_df
    
    def export_results(self, outcomes_df: pd.DataFrame, path: str):
        """Export optimization results."""
        outcomes_df.to_csv(path, index=False)
        logger.info(f"Results exported to {path}")
