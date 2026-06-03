# Bugfix Requirements Document

## Introduction

This document addresses critical bugs in the agentic EV tariff optimization system that prevent the learning loop from functioning. Analysis of `agentic_outcomes.csv` reveals that the system is stuck in a non-learning state where:
- Revenue gain percentages are mechanically locked to discrete values (-6.5%, 1.5%, 3.5%)
- Elasticity parameters Θ = [ε, α, β] never update despite the monitoring agent running
- Regime classification is always "neutral" regardless of utilization levels
- Queue actuals frequently collapse to zero

The system produces cosmetic outputs but performs no actual optimization or learning.

## Bug Analysis

### Current Behavior (Defect)

**Revenue Calculation Bugs:**

1.1 WHEN the pricing agent selects p_new=13.5, p_new=14.5, or p_new=16.5 THEN revenue_gain_pct is mechanically locked to exactly 3.5%, 1.5%, or -6.5% respectively regardless of actual demand conditions

1.2 WHEN p_new is calculated, the corresponding demand_shift is deterministically set to 0.15, 0.05, or -0.15 based solely on the price tier, creating a mechanical relationship where revenue outcomes cannot vary within a price tier

**Parameter Learning Bugs:**

1.3 WHEN the monitoring agent proposes parameter updates with non-zero deltas (delta_epsilon, delta_alpha, delta_beta) AND the orchestrator calls `pricing_agent.apply_update(eta * delta)` THEN the theta parameters [ε, α, β] remain frozen at [1.5, 2.5, 2.5] for all 40 steps

1.4 WHEN the learning loop executes over multiple steps THEN epsilon, alpha, and beta values in the output CSV remain constant at their initialization values, indicating zero learning

**Regime Classification Bugs:**

1.5 WHEN actual utilization u_actual exceeds 80% (e.g., steps 2, 15, 17 with u_actual ≈ 0.85-0.86) THEN the regime is classified as "neutral" instead of "surge"

1.6 WHEN predicted utilization u_pred exceeds 80% (e.g., step 15 with u_pred=0.845) THEN the regime is still classified as "neutral" even though this should trigger surge pricing

1.7 WHEN utilization is below 30% THEN the regime remains "neutral" instead of being classified as "discount"

1.8 WHEN the system runs for 40 steps THEN all 40 steps are classified as "neutral" (100% neutral, 0% surge, 0% discount)

**Queue Simulation Bugs:**

1.9 WHEN q_pred is non-zero (e.g., steps 0, 4, 7, 9 with q_pred > 0) THEN q_actual frequently equals 0.0, indicating the queue simulation is not computing or persisting actual queue states

1.10 WHEN queue predictions indicate congestion THEN the actual queue measurement does not reflect the predicted state

### Expected Behavior (Correct)

**Revenue Calculation Fixes:**

2.1 WHEN the pricing agent selects any p_new value THEN revenue_gain_pct SHALL be computed from the actual price elasticity formula using continuous mathematics, not discrete tier-based outcomes

2.2 WHEN demand_shift is calculated THEN it SHALL use the formula: `demand_shift = -ε × (p_new - baseline) / baseline` with the current epsilon value, allowing continuous variation based on the actual price difference

2.3 WHEN revenue is calculated THEN revenue_new SHALL equal `p_new × kwh × (1 + demand_shift)` and revenue_gain_pct SHALL equal `(revenue_new - revenue_baseline) / revenue_baseline × 100`, producing varying outcomes even within the same price tier

**Parameter Learning Fixes:**

2.4 WHEN the monitoring agent proposes parameter updates with deltas [Δε, Δα, Δβ] AND the orchestrator applies `pricing_agent.apply_update(eta * delta)` THEN the theta parameters SHALL update according to: θ_new = θ_old + η × Δθ with proper bounds enforcement

2.5 WHEN learning occurs over multiple steps THEN epsilon, alpha, and beta values SHALL change in response to revenue outcomes and utilization patterns, demonstrating actual learning

2.6 WHEN the monitoring agent detects declining revenue for 3+ consecutive steps THEN epsilon SHALL decrease (Δε < 0) to reduce price sensitivity

2.7 WHEN high utilization (>80%) persists in surge regime THEN alpha SHALL increase (Δα > 0) to strengthen surge pricing

**Regime Classification Fixes:**

2.8 WHEN u_pred > 0.80 THEN the regime SHALL be classified as "surge" and the pricing agent SHALL compute p_new above the baseline

2.9 WHEN u_pred < 0.30 THEN the regime SHALL be classified as "discount" and the pricing agent SHALL compute p_new below the baseline

2.10 WHEN 0.30 ≤ u_pred ≤ 0.80 THEN the regime SHALL be classified as "neutral" and the pricing agent SHALL compute p_new near the baseline

2.11 WHEN the system runs over 40 steps with varying utilization THEN the regime distribution SHALL reflect actual utilization patterns (e.g., if 5% of steps have u_pred > 0.80, approximately 5% of regimes should be "surge")

**Queue Simulation Fixes:**

2.12 WHEN q_pred is computed as non-zero THEN q_actual SHALL be computed from the demand simulation or baseline queue data and SHALL NOT default to 0.0

2.13 WHEN the queue simulation runs THEN it SHALL produce non-zero q_actual values that correlate with q_pred and utilization levels

### Unchanged Behavior (Regression Prevention)

**Preserved Core Mechanics:**

3.1 WHEN the demand agent predicts utilization and queue THEN predictions SHALL CONTINUE TO be made using the trained ML model without modification

3.2 WHEN the metrics engine computes utilization_new THEN it SHALL CONTINUE TO use the formula: `u_new = clip(u_actual + demand_shift × 0.1, 0.0, 1.0)`

3.3 WHEN the reward is calculated THEN it SHALL CONTINUE TO use the multi-objective formula: `reward = w1×revenue_gain + w2×utilization_improvement - w3×queue_penalty`

**Preserved Fallback Behavior:**

3.4 WHEN the LLM fails to respond or returns invalid JSON THEN the pricing agent SHALL CONTINUE TO fall back to deterministic pricing formulas

3.5 WHEN the LLM is disabled or unavailable THEN the monitoring agent SHALL CONTINUE TO use deterministic parameter update rules

**Preserved Bounds and Constraints:**

3.6 WHEN theta parameters are updated THEN they SHALL CONTINUE TO be clipped to valid ranges: ε ∈ [0.1, 5.0], α ∈ [1.0, 10.0], β ∈ [1.0, 10.0]

3.7 WHEN p_new is computed THEN it SHALL CONTINUE TO be clipped to the configured pricing bounds [lower, upper]

3.8 WHEN parameter deltas are proposed THEN they SHALL CONTINUE TO be constrained to maximum magnitudes: |Δε| ≤ 0.05, |Δα| ≤ 0.10, |Δβ| ≤ 0.10

**Preserved Logging and Statistics:**

3.9 WHEN agents complete their operations THEN they SHALL CONTINUE TO track and report statistics (LLM success rates, fallback counts)

3.10 WHEN the orchestrator exports results THEN it SHALL CONTINUE TO save the outcomes dataframe to CSV with all current columns


## Bug Condition Analysis

### Bug Condition Functions

**BC1: Revenue Calculation Stuck**
```pascal
FUNCTION isRevenueCalculationBuggy(step_data)
  INPUT: step_data of type StepData
  OUTPUT: boolean
  
  // Revenue gain is mechanically determined by price tier
  IF step_data.p_new = 13.5 THEN
    RETURN step_data.revenue_gain_pct = 3.5
  ELSE IF step_data.p_new = 14.5 THEN
    RETURN step_data.revenue_gain_pct = 1.5
  ELSE IF step_data.p_new = 16.5 THEN
    RETURN step_data.revenue_gain_pct = -6.5
  END IF
  
  RETURN false
END FUNCTION
```

**BC2: Parameter Learning Frozen**
```pascal
FUNCTION isParameterLearningBuggy(outcomes_data)
  INPUT: outcomes_data of type List[StepData]
  OUTPUT: boolean
  
  // Theta parameters never change across all steps
  first_theta ← [outcomes_data[0].epsilon, outcomes_data[0].alpha, outcomes_data[0].beta]
  
  FOR EACH step IN outcomes_data DO
    current_theta ← [step.epsilon, step.alpha, step.beta]
    IF current_theta ≠ first_theta THEN
      RETURN false
    END IF
  END FOR
  
  RETURN true  // All theta values identical
END FUNCTION
```

**BC3: Regime Classification Broken**
```pascal
FUNCTION isRegimeClassificationBuggy(step_data)
  INPUT: step_data of type StepData
  OUTPUT: boolean
  
  // Regime should be surge when u_pred > 0.80, discount when < 0.30
  IF step_data.u_pred > 0.80 AND step_data.regime ≠ "surge" THEN
    RETURN true
  END IF
  
  IF step_data.u_pred < 0.30 AND step_data.regime ≠ "discount" THEN
    RETURN true
  END IF
  
  RETURN false
END FUNCTION
```

**BC4: Queue Simulation Collapsed**
```pascal
FUNCTION isQueueSimulationBuggy(step_data)
  INPUT: step_data of type StepData
  OUTPUT: boolean
  
  // Queue actual is zero when prediction is non-zero
  RETURN (step_data.q_pred > 0.0) AND (step_data.q_actual = 0.0)
END FUNCTION
```

### Properties for Fix Checking

**Property P1: Revenue Varies Continuously**
```pascal
// Fix Checking: Revenue calculation uses continuous elasticity
FOR ALL step WHERE isRevenueCalculationBuggy(step) DO
  // After fix, demand_shift should vary continuously
  demand_shift' ← compute_demand_shift'(step.epsilon, step.p_new, baseline)
  revenue_gain' ← compute_revenue_gain'(step.p_new, step.kwh, demand_shift', baseline)
  
  ASSERT demand_shift' = -step.epsilon × (step.p_new - baseline) / baseline
  ASSERT revenue_gain' is NOT mechanically locked to {-6.5, 1.5, 3.5}
  ASSERT revenue_gain' varies based on actual epsilon and price values
END FOR
```

**Property P2: Parameters Update Over Time**
```pascal
// Fix Checking: Theta parameters learn from feedback
FOR ALL outcomes WHERE isParameterLearningBuggy(outcomes) DO
  // After fix, theta should change when updates are applied
  initial_theta ← [outcomes[0].epsilon, outcomes[0].alpha, outcomes[0].beta]
  
  changes_detected ← 0
  FOR step ← 3 TO length(outcomes) - 1 DO
    current_theta ← [outcomes[step].epsilon, outcomes[step].alpha, outcomes[step].beta]
    IF current_theta ≠ initial_theta THEN
      changes_detected ← changes_detected + 1
    END IF
  END FOR
  
  ASSERT changes_detected > 0  // At least some learning occurred
END FOR
```

**Property P3: Regime Reflects Utilization**
```pascal
// Fix Checking: Regime classification follows utilization thresholds
FOR ALL step WHERE isRegimeClassificationBuggy(step) DO
  regime' ← classify_regime'(step.u_pred)
  
  IF step.u_pred > 0.80 THEN
    ASSERT regime' = "surge"
  ELSE IF step.u_pred < 0.30 THEN
    ASSERT regime' = "discount"
  ELSE
    ASSERT regime' = "neutral"
  END IF
END FOR
```

**Property P4: Queue Computed Correctly**
```pascal
// Fix Checking: Queue actuals reflect predictions
FOR ALL step WHERE isQueueSimulationBuggy(step) DO
  q_actual' ← compute_queue_actual'(step)
  
  ASSERT q_actual' > 0.0 OR step.q_pred = 0.0
  ASSERT q_actual' correlates with step.q_pred
END FOR
```

### Preservation Property

**Property P0: Non-Buggy Behavior Unchanged**
```pascal
// Preservation Checking: Core mechanics preserved
FOR ALL step WHERE NOT (isRevenueCalculationBuggy(step) OR 
                        isRegimeClassificationBuggy(step) OR 
                        isQueueSimulationBuggy(step)) DO
  
  // Ensure unchanged behavior for correct operations
  ASSERT demand_prediction'(step) = demand_prediction(step)
  ASSERT bounds_enforcement'(step.p_new) = bounds_enforcement(step.p_new)
  ASSERT reward_computation'(step) = reward_computation(step)
  ASSERT fallback_logic'(step) = fallback_logic(step)
END FOR

// For parameter learning, preservation applies to:
// - Valid range constraints
// - Learning rate decay
// - Convergence checking logic
```

### Key Definitions

- **F**: Original (unfixed) system - current code with frozen parameters and mechanical revenue
- **F'**: Fixed system - code with working learning loop and continuous calculations
- **Counterexamples**: See `outputs/agentic_outcomes.csv` rows 0-39 where:
  - All rows show frozen Θ = [1.5, 2.5, 2.5]
  - Rows with p_new ∈ {13.5, 14.5, 16.5} show mechanical revenue_gain_pct
  - Row 15: u_pred=0.845 but regime="neutral" (should be surge)
  - Row 0: q_pred=1.106 but q_actual=0.0
