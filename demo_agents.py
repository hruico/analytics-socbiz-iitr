#!/usr/bin/env python3
"""
Demo script showing the three agents in action.

This demonstrates how the agents work together in the optimization loop.
"""
import numpy as np
import pandas as pd
from src.agents.demand import DemandAgent
from src.agents.pricing import PricingAgent
from src.agents.monitoring import MonitoringAgent, StepMetrics
from src.utils.metrics import MetricsEngine

print("=" * 70)
print("EV Tariff Optimization - Agent Demo")
print("=" * 70)

# Generate tiny synthetic dataset
print("\n1. Generating synthetic training data...")
np.random.seed(42)
train_data = []
for i in range(50):
    hour = i % 24
    train_data.append({
        'acn_sessions_count': 20 + np.random.randint(-5, 5),
        'acn_total_kwh': 400 + np.random.normal(0, 50),
        'acn_avg_kwh_per_session': 20 + np.random.normal(0, 3),
        'hour_of_day': hour,
        'day_of_week': (i // 24) % 7,
        'is_weekend': 1 if ((i // 24) % 7) >= 5 else 0,
        'is_peak_hour': 1 if hour in [7,8,9,17,18,19] else 0,
        'urban_mean_utilization': 0.5 + np.random.normal(0, 0.1),
        'urban_peak_queue': max(0, np.random.normal(2, 1))
    })
train_df = pd.DataFrame(train_data)
print(f"   Created {len(train_df)} training samples")

# Initialize agents
print("\n2. Initializing three agents...")
demand_agent = DemandAgent(random_seed=42)
pricing_agent = PricingAgent(
    baseline=15.0,
    bounds=(10.0, 22.0),
    theta=np.array([1.5, 2.5, 2.5])
)
monitoring_agent = MonitoringAgent()
metrics_engine = MetricsEngine(reward_weights=(1.0, 0.5, 0.3))
print("   ✓ Demand Agent (XGBoost)")
print("   ✓ Pricing Agent (deterministic fallback)")
print("   ✓ Monitoring Agent (parameter updates)")

# Train demand agent
print("\n3. Training Demand Agent...")
train_result = demand_agent.train(train_df)
print(f"   Trained on {train_result['samples']} samples")

# Run a few optimization steps
print("\n4. Running optimization loop (5 steps)...")
print("\n" + "-" * 70)

test_data = []
for i in range(5):
    hour = (i + 10) % 24
    test_data.append({
        'acn_sessions_count': 22,
        'acn_total_kwh': 420.0,
        'acn_avg_kwh_per_session': 19.1,
        'hour_of_day': hour,
        'day_of_week': 1,
        'is_weekend': 0,
        'is_peak_hour': 1 if hour in [7,8,9,17,18,19] else 0,
        'urban_mean_utilization': 0.65,
        'urban_peak_queue': 2.5
    })
test_df = pd.DataFrame(test_data)

history = []

for step in range(5):
    print(f"\nStep {step + 1}:")
    
    row = test_df.iloc[step]
    
    # Agent 1: Demand prediction
    u_pred, q_pred, congestion_prob = demand_agent.predict(test_df.iloc[[step]])
    print(f"  [Demand Agent]")
    print(f"    Predicted: utilization={u_pred[0]:.2%}, queue={q_pred[0]:.2f}, congestion={congestion_prob[0]:.2%}")
    
    # Agent 2: Pricing decision
    decision = pricing_agent.compute_tariff(
        u_pred[0], q_pred[0],
        row['hour_of_day'], row['is_weekend'], congestion_prob[0]
    )
    print(f"  [Pricing Agent]")
    print(f"    Decision: price=₹{decision.p_new:.2f}, regime={decision.regime}")
    print(f"    Rationale: {decision.rationale}")
    
    # Compute metrics
    metrics = metrics_engine.compute_step_metrics(
        p_new=decision.p_new,
        kwh=row['acn_total_kwh'],
        u_actual=row['urban_mean_utilization'],
        q_actual=row['urban_peak_queue'],
        epsilon=pricing_agent.theta[0],
        baseline=15.0,
        q_baseline_mean=2.5
    )
    
    print(f"  [Metrics]")
    print(f"    Revenue gain: {metrics['revenue_gain_pct']:+.2f}%")
    print(f"    Reward: {metrics['reward']:.2f}")
    
    # Agent 3: Monitoring and parameter update
    step_metric = StepMetrics(
        step=step,
        regime=decision.regime,
        revenue_gain_pct=metrics['revenue_gain_pct'],
        u_actual=row['urban_mean_utilization'],
        u_pred=u_pred[0]
    )
    history.append(step_metric)
    
    if len(history) >= 3:
        update = monitoring_agent.evaluate_and_propose(
            step, metrics['revenue_gain_pct'],
            row['urban_mean_utilization'], u_pred[0],
            decision.regime, history[-3:]
        )
        print(f"  [Monitoring Agent]")
        print(f"    Parameter updates: Δε={update.delta_epsilon:+.3f}, "
              f"Δα={update.delta_alpha:+.3f}, Δβ={update.delta_beta:+.3f}")
        
        # Apply update with learning rate
        eta = 0.1
        delta = np.array([update.delta_epsilon, update.delta_alpha, update.delta_beta])
        pricing_agent.apply_update(eta * delta)
        print(f"    New theta: ε={pricing_agent.theta[0]:.3f}, "
              f"α={pricing_agent.theta[1]:.3f}, β={pricing_agent.theta[2]:.3f}")

print("\n" + "-" * 70)
print("\n5. Summary")
print(f"   Initial theta: [1.500, 2.500, 2.500]")
print(f"   Final theta:   [{pricing_agent.theta[0]:.3f}, {pricing_agent.theta[1]:.3f}, {pricing_agent.theta[2]:.3f}]")
print(f"   Theta changed: {np.any(np.abs(pricing_agent.theta - np.array([1.5, 2.5, 2.5])) > 0.001)}")
print("\n" + "=" * 70)
print("Demo complete! The three agents worked together to optimize pricing.")
print("=" * 70)
