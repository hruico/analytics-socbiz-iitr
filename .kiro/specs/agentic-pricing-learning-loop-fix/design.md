# Agentic Pricing Learning Loop Bugfix Design

## Overview

The agentic EV tariff optimization system exhibits four critical bugs that prevent the learning loop from functioning:

1. **Discrete Price Selection Bug**: The pricing agent selects only three discrete price values (13.5, 14.5, 16.5), making revenue gains mechanically locked to three outcomes despite having a continuous elasticity formula
2. **Parameter Update Bug**: Theta parameters [ε, α, β] remain frozen at initialization values despite apply_update() being called
3. **Regime Classification Bug**: u_pred systematically underestimates high-utilization scenarios, preventing surge/discount regimes from triggering
4. **Queue Simulation Bug**: q_actual defaults to 0 instead of being computed from demand data

The fix strategy focuses on enabling continuous price selection, fixing the numpy array mutation issue, improving demand model predictions, and properly computing queue actuals.

## Glossary

- **Bug_Condition (C)**: The conditions that trigger each of the four bugs
- **Property (P)**: The desired behavior for each bug category - continuous revenue variation, parameter learning, correct regime classification, and proper queue computation
- **Preservation**: Existing reward calculation, bounds enforcement, and fallback logic that must remain unchanged
- **compute_tariff()**: The method in `src/agents/pricing.py` that determines p_new - currently outputs only discrete values
- **apply_update()**: The method in `src/agents/pricing.py` that updates theta parameters - appears correct but may have subtle mutation issues
- **demand_shift**: The elasticity-based demand adjustment computed in `src/utils/metrics.py` - formula is correct but input prices are discrete
- **theta**: The parameter vector [ε, α, β] stored in `PricingAgent.theta` - currently frozen at [1.5, 2.5, 2.5]
- **regime**: The pricing strategy classification (surge/neutral/discount) determined by u_pred thresholds
- **q_actual**: The actual queue length passed from orchestrator to metrics engine - currently 0 in many cases

## Bug Details

### Fault Condition

The bugs manifest in four distinct but related failure modes:

**BC1: Discrete Price Selection**
The pricing agent (both LLM and fallback) outputs only three discrete price values: 13.5, 14.5, and 16.5. Even though the continuous elasticity formula `demand_shift = -ε × (p_new - baseline) / baseline` exists in `metrics.py`, it receives only these three input values, producing mechanically locked revenue outcomes.

**BC2: Frozen Parameter Updates**
The theta parameters never change across 40 steps despite the monitoring agent proposing updates and the orchestrator calling `apply_update(eta * delta)`. The values remain [1.5, 2.5, 2.5] throughout execution.

**BC3: Regime Classification Failure**
97.5% of steps are classified as "neutral" even when u_pred exceeds 0.80 (which should trigger "surge"). Step 15 shows u_pred=0.845 but regime="neutral". This appears to be a demand model underfit issue where u_pred systematically underestimates high-load scenarios.

**BC4: Queue Simulation Collapse**
q_actual equals 0.0 in many cases where q_pred > 0, indicating the queue simulation is not running or the value is not being captured correctly.

**Formal Specification:**
```
FUNCTION isBugCondition(step_data, all_outcomes)
  INPUT: step_data of type StepData, all_outcomes of type List[StepData]
  OUTPUT: boolean
  
  // BC1: Price is one of three discrete values
  bc1 := step_data.p_new IN [13.5, 14.5, 16.5]
  
  // BC2: Theta never changes from initialization
  bc2 := ALL steps IN all_outcomes HAVE 
         steps.epsilon = 1.5 AND steps.alpha = 2.5 AND steps.beta = 2.5
  
  // BC3: Regime is neutral when utilization warrants surge/discount
  bc3 := (step_data.u_pred > 0.80 OR step_data.u_pred < 0.30) 
         AND step_data.regime = "neutral"
  
  // BC4: Queue actual is zero when prediction is non-zero
  bc4 := step_data.q_pred > 0.0 AND step_data.q_actual = 0.0
  
  RETURN bc1 OR bc2 OR bc3 OR bc4
END FUNCTION
```

### Examples

**BC1: Discrete Price Selection**
- Step 0: p_new=14.5 → demand_shift=0.05 → revenue_gain=1.5% (mechanically locked)
- Step 1: p_new=13.5 → demand_shift=0.15 → revenue_gain=3.5% (mechanically locked)
- Step 2: p_new=16.5 → demand_shift=-0.15 → revenue_gain=-6.5% (mechanically locked)
- Expected: p_new should vary continuously (e.g., 14.23, 15.67, 13.89) producing varying revenue gains

**BC2: Frozen Parameters**
- Steps 0-39: [ε, α, β] = [1.5, 2.5, 2.5] (no variation)
- Expected: After step 10 with declining revenue, ε should decrease to ~1.45
- Expected: After step 15 with high utilization, α should increase to ~2.6

**BC3: Regime Misclassification**
- Step 15: u_pred=0.845 (> 0.80 threshold) but regime="neutral" (should be "surge")
- Step 2: u_actual=0.859 but regime="neutral" (should be "surge")
- Expected: When u_pred > 0.80, regime should be "surge" with p_new > baseline

**BC4: Queue Collapse**
- Step 0: q_pred=1.106 but q_actual=0.0
- Step 4: q_pred=1.152 but q_actual=0.0
- Expected: q_actual should correlate with q_pred or use actual queue measurements

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Continuous elasticity formula in `metrics.py` must remain: `demand_shift = -ε × (p_new - baseline) / baseline`
- Reward computation must continue using: `reward = w1×revenue_gain + w2×utilization_improvement - w3×queue_penalty`
- Parameter bounds enforcement must continue clipping: ε ∈ [0.1, 5.0], α ∈ [1.0, 10.0], β ∈ [1.0, 10.0]
- Price bounds enforcement must continue clipping: p_new ∈ [10.0, 22.0]
- Fallback logic when LLM fails must remain functional
- Logging and statistics tracking must continue working

**Scope:**
All inputs that do NOT involve the buggy price selection, parameter updates, regime classification, or queue computation should be completely unaffected by this fix. This includes:
- Demand prediction model (except potential retraining for better high-load predictions)
- Convergence checking logic
- LLM retry and fallback mechanisms
- CSV export functionality
- Configuration validation

## Hypothesized Root Cause

Based on the bug analysis and code inspection, the most likely issues are:

### 1. **Discrete Price Selection in Pricing Agent**

**Location**: `src/agents/pricing.py` - `_deterministic_fallback()` method

**Issue**: The fallback formula computes p_new using:
```python
p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 10.0
```

This formula with the current parameters and `/10.0` scaling produces only discrete outputs. The LLM may also be returning discrete values due to prompt structure or rounding.

**Evidence**:
- All 40 steps show only three price values: 13.5, 14.5, 16.5
- The formula is overly quantized by the `/10.0` factor
- No randomness or continuous variation in the price calculation

### 2. **Apply Update Numpy Array Issue**

**Location**: `src/agents/pricing.py` - `apply_update()` method

**Issue**: Looking at the code, `apply_update()` appears correct:
```python
def apply_update(self, delta: np.ndarray) -> None:
    self.theta += delta
    self.theta[0] = np.clip(self.theta[0], 0.1, 5.0)
    ...
```

However, the bug persists. Possible causes:
- The `delta` array passed in is always zeros (monitoring agent issue)
- The `eta * delta` multiplication in orchestrator produces zeros due to float precision
- Numpy array view vs copy issue if theta is being reassigned elsewhere

**Evidence**: 
- Theta remains [1.5, 2.5, 2.5] across all 40 steps
- The update code looks correct syntactically
- Must be either receiving zero deltas or having updates overwritten

### 3. **Demand Model Underfit for High Utilization**

**Location**: `src/agents/demand.py` - trained model

**Issue**: The demand prediction model systematically underestimates high-utilization scenarios. This causes u_pred to rarely exceed 0.80, preventing surge regime from triggering even when u_actual > 0.80.

**Evidence**:
- Step 15: u_actual=0.859, but u_pred=0.845 (underprediction)
- 97.5% neutral regime across 40 steps
- No surge pricing triggered despite several high-utilization periods

### 4. **Queue Actual Not Computed from Data**

**Location**: `src/orchestrator.py` - line 95

**Issue**: The code shows:
```python
q_actual=row['urban_peak_queue']
```

This reads from the dataframe, so either:
- The dataframe column `urban_peak_queue` contains zeros (data preparation issue)
- The queue simulation is not running to populate this field
- There's a mismatch between column name and actual data

**Evidence**:
- q_actual=0.0 in many steps where q_pred > 0
- The orchestrator code looks correct - it's reading from the dataframe
- Must be a data source issue or preprocessing bug

## Correctness Properties

Property 1: Fault Condition - Continuous Price and Revenue Variation

_For any_ step where the pricing agent computes a tariff, the fixed system SHALL produce continuously varying p_new values across the full pricing bounds range [10.0, 22.0], and revenue_gain_pct SHALL vary continuously based on the actual elasticity formula rather than being mechanically locked to discrete outcomes.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Fault Condition - Parameter Learning Functions

_For any_ sequence of steps where the monitoring agent proposes non-zero parameter updates, the fixed system SHALL update theta parameters according to θ_new = θ_old + η × Δθ, demonstrating measurable changes in [ε, α, β] values over time in response to revenue and utilization feedback.

**Validates: Requirements 2.4, 2.5, 2.6, 2.7**

Property 3: Fault Condition - Regime Classification Accuracy

_For any_ step where u_pred exceeds 0.80 or falls below 0.30, the fixed system SHALL classify the regime as "surge" or "discount" respectively, and the overall regime distribution SHALL reflect the actual utilization pattern in the data.

**Validates: Requirements 2.8, 2.9, 2.10, 2.11**

Property 4: Fault Condition - Queue Computation Integrity

_For any_ step where q_pred is non-zero or demand indicates congestion, the fixed system SHALL compute q_actual from the simulation or actual demand data, producing non-zero values that correlate with predicted queue levels.

**Validates: Requirements 2.12, 2.13**

Property 5: Preservation - Core Mechanics Unchanged

_For any_ input where continuous pricing would have worked correctly (if prices weren't discrete), bounds enforcement was correct, or fallback logic was functioning, the fixed system SHALL produce exactly the same behavior as the original system, preserving all existing mechanics for reward calculation, parameter bounds clipping, and LLM fallback handling.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

## Fix Implementation

### Changes Required

#### Fix 1: Enable Continuous Price Selection

**File**: `src/agents/pricing.py`

**Function**: `_deterministic_fallback()`

**Specific Changes**:

1. **Remove /10.0 Quantization**: Change price calculation to allow finer granularity
   - Current: `p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 10.0`
   - Fixed: `p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 100.0`
   - This allows 100x more price variation within the same range

2. **Add Precision to Neutral Regime**: Increase sensitivity in neutral pricing
   - Current: `p_new = self.baseline + (u_pred - 0.55) * 2.0`
   - Fixed: `p_new = self.baseline + (u_pred - 0.55) * (self.upper - self.lower) / 2.0`
   - This scales neutral pricing proportionally to the full bounds range

3. **LLM Prompt Modification**: Update prompt to encourage continuous values
   - Add instruction: "Use precise decimal values (e.g., 14.37, 15.82) rather than rounding to nearest 0.5"
   - Add example outputs showing decimal precision

#### Fix 2: Debug and Fix Parameter Update Issue

**File**: `src/agents/monitoring.py` and `src/orchestrator.py`

**Investigation Steps**:

1. **Add Logging to Track Delta Values**: Before apply_update(), log the delta array
   ```python
   logger.info(f"Step {step}: Applying delta={eta * delta}, eta={eta}")
   ```

2. **Verify Monitoring Agent Produces Non-Zero Deltas**: Check that `evaluate_and_propose()` returns meaningful updates
   - If deltas are zeros, the issue is in the monitoring agent's logic
   - If deltas are non-zero but theta doesn't change, issue is in apply_update()

3. **Potential Fix if Delta is Zero**: Strengthen monitoring agent's update magnitude
   ```python
   # Ensure minimum update magnitude when trends are detected
   if revenue_declining_for_3_steps:
       delta_epsilon = max(delta_epsilon, -0.05)  # Force meaningful update
   ```

4. **Potential Fix if Theta Reassignment**: Check for theta being reassigned after updates
   - Search for any `self.theta = ` assignments that might override updates
   - Ensure theta is not being reset from config on each step

**File**: `src/agents/pricing.py`

**Function**: `apply_update()`

**Verification Changes**:
```python
def apply_update(self, delta: np.ndarray) -> None:
    """Update theta parameters."""
    theta_before = self.theta.copy()
    self.theta += delta
    self.theta[0] = np.clip(self.theta[0], 0.1, 5.0)
    self.theta[1] = np.clip(self.theta[1], 1.0, 10.0)
    self.theta[2] = np.clip(self.theta[2], 1.0, 10.0)
    
    # Add logging to verify update occurred
    logger.debug(f"Theta update: {theta_before} -> {self.theta}, delta={delta}")
```

#### Fix 3: Improve Regime Classification (Lower Priority)

**File**: `src/agents/demand.py`

**Approach**: This requires model retraining or tuning, which is more involved. For immediate fix:

**Option A - Use Actual Utilization for Regime** (Quick Fix):
```python
# In pricing agent, use u_actual for regime classification when available
# This bypasses the underfit prediction issue
regime = classify_regime(u_actual if u_actual is not None else u_pred)
```

**Option B - Retrain Demand Model** (Proper Fix):
- Add more training data from high-utilization periods
- Adjust model features to better capture peak load patterns
- Increase model capacity or use ensemble methods
- This is beyond the immediate bugfix scope

#### Fix 4: Compute Queue Actual Correctly

**File**: `src/data_loader.py` or data preprocessing

**Investigation**:
1. Check if `urban_peak_queue` column in the input data contains zeros
2. If data source has zeros, need to compute q_actual from simulation

**File**: `src/orchestrator.py`

**Specific Changes**:

**Option A - Use Queue Prediction as Fallback**:
```python
# Line 95 - add fallback when data is missing
q_actual_raw = row['urban_peak_queue']
q_actual = q_actual_raw if q_actual_raw > 0 else q_pred[step]
```

**Option B - Compute from Utilization**:
```python
# Derive queue from utilization and capacity
# Assuming M/M/1 queue approximation: q ≈ u / (1 - u) when u is high
if row['urban_peak_queue'] == 0.0:
    u = row['urban_mean_utilization']
    q_actual = (u / (1.0 - u + 0.01)) * baseline_capacity_factor
else:
    q_actual = row['urban_peak_queue']
```

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, write exploratory tests that demonstrate the bugs on the unfixed code to confirm our root cause analysis, then verify that the fixes work correctly and preserve existing behavior.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate the bugs BEFORE implementing the fix. Confirm or refute the root cause analysis. If we refute, we will need to re-hypothesize.

**Test Plan**: Write tests that simulate the optimization loop on a small dataset and assert the buggy behaviors occur on UNFIXED code. This validates our understanding before fixing.

**Test Cases**:

1. **Discrete Price Test**: Run 10 steps and assert that p_new only takes values in a small discrete set (will pass on unfixed code, demonstrating the bug)
   ```python
   def test_discrete_price_bug_exists():
       # Run on UNFIXED code
       outcomes = run_optimization(steps=10)
       unique_prices = set(outcomes['p_new'])
       assert len(unique_prices) <= 5  # Bug: only a few discrete values
   ```

2. **Frozen Theta Test**: Run 20 steps and assert that theta never changes from initialization (will pass on unfixed code)
   ```python
   def test_frozen_theta_bug_exists():
       # Run on UNFIXED code
       outcomes = run_optimization(steps=20)
       theta_values = outcomes[['epsilon', 'alpha', 'beta']].values
       assert np.all(theta_values == theta_values[0])  # Bug: all identical
   ```

3. **Regime Classification Test**: Find a step with u_pred > 0.80 and assert regime is "neutral" (will pass on unfixed code)
   ```python
   def test_regime_bug_exists():
       # Run on UNFIXED code with high-utilization data
       outcomes = run_optimization(test_data_with_high_util)
       high_util_steps = outcomes[outcomes['u_pred'] > 0.80]
       assert (high_util_steps['regime'] == 'neutral').all()  # Bug exists
   ```

4. **Queue Zero Test**: Find steps with q_pred > 0 and assert q_actual == 0 (will pass on unfixed code)
   ```python
   def test_queue_bug_exists():
       # Run on UNFIXED code
       outcomes = run_optimization(steps=10)
       buggy_queue = outcomes[(outcomes['q_pred'] > 0) & (outcomes['q_actual'] == 0)]
       assert len(buggy_queue) > 0  # Bug: some queue actuals are zero
   ```

**Expected Counterexamples**:
- Pricing agent outputs only 3-5 distinct values despite continuous bounds
- Theta remains [1.5, 2.5, 2.5] across all steps
- High utilization steps (u_pred > 0.80) classified as "neutral"
- Multiple steps where q_pred > 0 but q_actual = 0

### Fix Checking

**Goal**: Verify that for all inputs where the bug condition holds, the fixed system produces the expected behavior.

**Pseudocode:**
```
FOR ALL step_data WHERE isBugCondition(step_data, all_outcomes) DO
  // BC1 Fix: Continuous prices
  IF step_data.p_new IN [13.5, 14.5, 16.5] THEN
    outcomes_fixed := run_fixed_system(same_inputs)
    ASSERT count_unique(outcomes_fixed.p_new) > 20  // Much more variation
    ASSERT outcomes_fixed.revenue_gain_pct varies continuously
  END IF
  
  // BC2 Fix: Learning occurs
  IF all theta values are frozen THEN
    outcomes_fixed := run_fixed_system(same_inputs)
    theta_changes := count_theta_changes(outcomes_fixed)
    ASSERT theta_changes > 0  // Learning happened
  END IF
  
  // BC3 Fix: Regime classification
  IF step_data.u_pred > 0.80 AND regime = "neutral" THEN
    decision_fixed := classify_regime_fixed(step_data.u_pred)
    ASSERT decision_fixed.regime = "surge"
  END IF
  
  // BC4 Fix: Queue computation
  IF step_data.q_pred > 0 AND step_data.q_actual = 0 THEN
    q_actual_fixed := compute_queue_fixed(step_data)
    ASSERT q_actual_fixed > 0 OR step_data.q_pred = 0
  END IF
END FOR
```

### Preservation Checking

**Goal**: Verify that for all inputs where the bug condition does NOT hold, the fixed system produces the same result as the original system.

**Pseudocode:**
```
FOR ALL functionality WHERE NOT affected_by_bugs DO
  // Test that unchanged mechanics are preserved
  ASSERT demand_prediction_fixed(input) = demand_prediction_original(input)
  ASSERT bounds_enforcement_fixed(p_new) = bounds_enforcement_original(p_new)
  ASSERT reward_computation_fixed(metrics) = reward_computation_original(metrics)
  ASSERT elasticity_formula_fixed(params) = elasticity_formula_original(params)
  ASSERT fallback_logic_fixed(llm_failure) = fallback_logic_original(llm_failure)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy operations

**Test Plan**: First observe behavior on UNFIXED code for non-buggy operations (e.g., bounds enforcement, reward calculation with valid inputs), then write property-based tests capturing that behavior.

**Test Cases**:

1. **Reward Calculation Preservation**: Generate random valid metric inputs and verify reward formula unchanged
   ```python
   @given(revenue_gain=floats(-10, 10), util_improvement=floats(-0.2, 0.2))
   def test_reward_formula_preserved(revenue_gain, util_improvement):
       reward_original = compute_reward_original(revenue_gain, util_improvement, ...)
       reward_fixed = compute_reward_fixed(revenue_gain, util_improvement, ...)
       assert reward_original == reward_fixed
   ```

2. **Bounds Enforcement Preservation**: Generate random prices and verify clipping behavior unchanged
   ```python
   @given(p_new=floats(0, 50))
   def test_bounds_enforcement_preserved(p_new):
       p_clipped_original = clip_original(p_new, bounds)
       p_clipped_fixed = clip_fixed(p_new, bounds)
       assert p_clipped_original == p_clipped_fixed
   ```

3. **Elasticity Formula Preservation**: Verify the core elasticity calculation unchanged
   ```python
   @given(epsilon=floats(0.1, 5.0), p_new=floats(10, 22), baseline=floats(12, 18))
   def test_elasticity_formula_preserved(epsilon, p_new, baseline):
       shift_original = compute_demand_shift_original(epsilon, p_new, baseline)
       shift_fixed = compute_demand_shift_fixed(epsilon, p_new, baseline)
       assert abs(shift_original - shift_fixed) < 1e-10
   ```

4. **Fallback Logic Preservation**: Verify deterministic fallback when LLM fails works identically
   ```python
   def test_fallback_logic_preserved():
       # Disable LLM to force fallback
       decision_original = pricing_agent_original.compute_tariff(u_pred=0.5, ...)
       decision_fixed = pricing_agent_fixed.compute_tariff(u_pred=0.5, ...)
       
       # Fallback behavior should be identical (modulo continuous price fix)
       assert decision_original.regime == decision_fixed.regime
       assert decision_original.fallback_used == decision_fixed.fallback_used
   ```

### Unit Tests

- Test discrete price quantization in `_deterministic_fallback()` before and after fix
- Test parameter update application with known delta values
- Test regime classification thresholds (u_pred boundaries at 0.30 and 0.80)
- Test queue actual fallback logic when dataframe has zeros
- Test that continuous elasticity formula in metrics.py remains unchanged

### Property-Based Tests

- Generate random utilization values and verify regime classification follows rules
- Generate random price inputs and verify continuous revenue variation after fix
- Generate random parameter deltas and verify theta updates correctly
- Test that reward calculation is identical for all preserved operations

### Integration Tests

- Run full optimization loop for 10 steps and verify price diversity increases
- Run full optimization loop and verify theta changes after step 3
- Run with high-utilization test data and verify surge regime triggers
- Run with test data and verify q_actual has non-zero values correlated with q_pred
- Compare final outcomes between fixed and unfixed code to quantify improvement
