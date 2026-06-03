# Parametric Formula Fix Bugfix Design

## Overview

The agentic EV pricing system contains three hardcoded formula issues that reduce parametric design and maintainability. This bugfix systematically replaces magic numbers with config-derived values:

1. **Surge formula inconsistency**: LLM path uses multiplicative formula while fallback uses additive, both with hardcoded 0.80 threshold
2. **Epsilon floor magic number**: Uses hardcoded 1.0 floor instead of deriving from revenue breakeven at neutral midpoint
3. **Neutral formula inconsistency**: LLM path uses hardcoded 0.55 midpoint and 8.0 slope while fallback computes from bounds

The fix creates a fully parametric system where all formulas derive values from config constants (baseline, bounds, thresholds) and learned parameters (alpha, beta, epsilon).

## Glossary

- **Bug_Condition (C)**: The condition where hardcoded values are used instead of config-derived parameters
- **Property (P)**: The desired behavior where formulas use parametric values consistently across LLM and fallback paths
- **Preservation**: Existing regime classification, parameter learning, and price behavior that must remain unchanged
- **surge_threshold**: The utilization threshold above which surge pricing applies (0.80 from classification rules)
- **discount_threshold**: The utilization threshold below which discount pricing applies (0.30 from classification rules)
- **midpoint_util**: The center of the neutral regime computed as (surge_threshold + discount_threshold) / 2 = 0.55
- **epsilon_floor**: Revenue breakeven threshold below which epsilon reduction is skipped, derived from baseline / expected_neutral_price
- **cooldown counter**: 5-step window during which epsilon reductions are blocked after a reduction occurs

## Bug Details

### Fault Condition

The bug manifests when pricing formulas are executed in `PricingAgent._compute_price_from_regime()` and `PricingAgent._deterministic_fallback()`, and when monitoring agent evaluates epsilon reduction in `MonitoringAgent._deterministic_fallback()`. These functions use hardcoded literal values instead of deriving them from configuration constants.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type PricingContext {regime: str, u_pred: float, path: str}
  OUTPUT: boolean
  
  RETURN (input.regime == "surge" AND input.path IN ["llm", "fallback"] AND uses_hardcoded_threshold(0.80))
         OR (input.regime == "neutral" AND input.path IN ["llm", "fallback"] AND uses_hardcoded_midpoint(0.55))
         OR (epsilon_reduction_check() AND uses_hardcoded_floor(1.0))
END FUNCTION
```

### Examples

**Issue 1: Surge Formula Inconsistency**
- **LLM path**: `p_new = self.baseline * (1 + alpha * (u_pred - 0.80))` with hardcoded 0.80
- **Fallback path**: `p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 100.0` with additive formula
- **Expected**: Both paths use multiplicative formula with surge_threshold derived from config

**Issue 2: Epsilon Floor Magic Number**
- **Current**: `epsilon_above_floor = current_epsilon > 1.0` with hardcoded 1.0
- **Expected**: `epsilon_floor = baseline / baseline = 1.0` but derived, not hardcoded, enabling future config changes

**Issue 3: Neutral Formula Inconsistency**
- **LLM path**: `p_new = self.baseline + (u_pred - 0.55) * 8.0` with hardcoded 0.55 and 8.0
- **Fallback path**: `p_new = self.baseline + (u_pred - 0.55) * (self.upper - self.lower) / 2.0` with hardcoded 0.55 but computed slope
- **Expected**: Both paths use `midpoint_util = 0.55` and `price_slope = (17.0 - 13.0) / (0.80 - 0.30) = 8.0` derived from config constants

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Regime classification thresholds (u_pred > 0.80 for surge, u_pred < 0.30 for discount) remain as system rules
- Price values at specific utilization levels should be identical before and after fix
- Parameter learning logic (alpha, beta, epsilon updates) remains unchanged
- Reward formula computation remains unchanged
- LLM prompt structure and response parsing remains unchanged
- Bounds clipping [lower, upper] remains unchanged
- Monitoring agent evaluation criteria remain unchanged

**Scope:**
All inputs that do NOT involve the three specific formula calculations should be completely unaffected by this fix. This includes:
- Discount regime pricing (uses different formula)
- XGBoost demand predictions
- CSV export metrics
- Convergence detection logic
- Learning rate decay schedule

## Hypothesized Root Cause

Based on the code analysis, the root causes are:

1. **Incremental Development**: Formulas were hardcoded during initial implementation without considering config-driven design

2. **Path Divergence**: LLM and fallback paths evolved independently, causing formula inconsistencies (multiplicative vs additive surge, different slope calculations for neutral)

3. **Magic Number Anti-Pattern**: The epsilon floor of 1.0 was hardcoded without documenting that it represents revenue breakeven at neutral midpoint

4. **Missing Abstraction**: No shared constants or derived values enforced consistency between LLM and fallback pricing paths

## Correctness Properties

Property 1: Fault Condition - Parametric Formula Consistency

_For any_ pricing context where formulas are computed, the system SHALL derive all threshold, midpoint, slope, and floor values from configuration constants (baseline, bounds, regime thresholds) rather than using hardcoded literals, ensuring consistency across LLM and fallback paths.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7**

Property 2: Preservation - Price Value Equivalence

_For any_ utilization value and regime combination, the fixed code SHALL produce exactly the same price output as the original code for the specific config values (baseline=15.0, bounds=[10.0, 22.0], surge_threshold=0.80, discount_threshold=0.30), preserving all existing pricing behavior.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File**: `src/agents/pricing.py`

**Function**: `PricingAgent.__init__()`

**Specific Changes**:
1. **Add Config Constants**: Add class constants for regime thresholds and neutral pricing parameters
   - `SURGE_THRESHOLD = 0.80` (derived from regime classification rules)
   - `DISCOUNT_THRESHOLD = 0.30` (derived from regime classification rules)
   - Compute `self.midpoint_util = (SURGE_THRESHOLD + DISCOUNT_THRESHOLD) / 2` → 0.55
   - Add `UPPER_NEUTRAL = 17.0` and `LOWER_NEUTRAL = 13.0` as config constants
   - Compute `self.price_slope = (UPPER_NEUTRAL - LOWER_NEUTRAL) / (SURGE_THRESHOLD - DISCOUNT_THRESHOLD)` → 8.0

2. **Replace Hardcoded Surge Threshold**: In `_compute_price_from_regime()` line 73
   - Change: `p_new = self.baseline * (1 + alpha * (u_pred - 0.80))`
   - To: `p_new = self.baseline * (1 + alpha * (u_pred - self.SURGE_THRESHOLD))`

3. **Replace Hardcoded Neutral Formula**: In `_compute_price_from_regime()` line 71
   - Change: `p_new = self.baseline + (u_pred - 0.55) * 8.0`
   - To: `p_new = self.baseline + (u_pred - self.midpoint_util) * self.price_slope`

4. **Unify Fallback Surge Formula**: In `_deterministic_fallback()` line 176
   - Change: `p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 100.0`
   - To: `p_new = self.baseline * (1 + alpha * (u_pred - self.SURGE_THRESHOLD))`
   - Remove surge_scalar calculation since it's no longer needed for pricing (only for tracking)

5. **Replace Fallback Neutral Formula**: In `_deterministic_fallback()` line 199
   - Change: `p_new = self.baseline + (u_pred - 0.55) * (self.upper - self.lower) / 2.0`
   - To: `p_new = self.baseline + (u_pred - self.midpoint_util) * self.price_slope`

**File**: `src/agents/monitoring.py`

**Function**: `MonitoringAgent.__init__()`

**Specific Changes**:
1. **Add Epsilon Floor Calculation**: Add instance variable in `__init__`
   - Compute `self.epsilon_floor = baseline / baseline = 1.0` (but derived, not hardcoded)
   - This represents revenue breakeven: at neutral midpoint, price = baseline, so epsilon_floor ensures epsilon doesn't drop below this ratio

2. **Replace Hardcoded Floor Check**: In `_deterministic_fallback()` line 198
   - Change: `epsilon_above_floor = current_epsilon > 1.0`
   - To: `epsilon_above_floor = current_epsilon > self.epsilon_floor`

3. **Update Floor Logging**: In `_deterministic_fallback()` line 207
   - Change: `logger.info(f"Epsilon reduction skipped: floor reached (ε={current_epsilon:.3f} ≤ 1.0)")`
   - To: `logger.info(f"Epsilon reduction skipped: floor reached (ε={current_epsilon:.3f} ≤ {self.epsilon_floor:.3f})")`

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate formula inconsistencies on unfixed code, then verify the fix produces identical price values while using parametric formulas.

### Exploratory Fault Condition Checking

**Goal**: Surface counterexamples that demonstrate hardcoded values BEFORE implementing the fix. Confirm the inconsistencies between LLM and fallback paths.

**Test Plan**: Write tests that compare formula outputs between LLM and fallback paths, and verify epsilon floor is hardcoded. Run these tests on the UNFIXED code to observe discrepancies.

**Test Cases**:
1. **Surge Formula Comparison**: Compare LLM surge formula vs fallback surge formula at u_pred=0.90 (will show different formulas on unfixed code)
2. **Neutral Formula Comparison**: Compare LLM neutral formula vs fallback neutral formula at u_pred=0.55 (will show different slope calculations on unfixed code)
3. **Epsilon Floor Hardcoding**: Verify epsilon floor is 1.0 literal rather than computed from baseline (will show hardcoded value on unfixed code)
4. **Threshold Hardcoding**: Verify 0.80 threshold appears as literal in formulas (will show hardcoded value on unfixed code)

**Expected Counterexamples**:
- LLM surge uses multiplicative formula, fallback surge uses additive formula
- LLM neutral uses hardcoded 8.0 slope, fallback neutral computes slope as (upper - lower) / 2.0 = 6.0
- Epsilon floor check uses literal 1.0 instead of baseline / baseline

### Fix Checking

**Goal**: Verify that for all pricing contexts where formulas are used, the fixed code derives values from config constants and produces consistent results across LLM and fallback paths.

**Pseudocode:**
```
FOR ALL pricing_context WHERE isBugCondition(pricing_context) DO
  price_llm := compute_price_llm_path(pricing_context)
  price_fallback := compute_price_fallback_path(pricing_context)
  
  ASSERT uses_parametric_values(price_llm)
  ASSERT uses_parametric_values(price_fallback)
  ASSERT price_llm == price_fallback (for same regime and u_pred)
END FOR
```

### Preservation Checking

**Goal**: Verify that for all utilization values and regime combinations, the fixed function produces the same price output as the original function.

**Pseudocode:**
```
FOR ALL u_pred IN [0.0, 1.0] AND regime IN ["surge", "neutral", "discount"] DO
  price_original := compute_price_original(u_pred, regime)
  price_fixed := compute_price_fixed(u_pred, regime)
  
  ASSERT abs(price_original - price_fixed) < 0.01  # numerical tolerance
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the utilization domain [0.0, 1.0]
- It catches edge cases at regime boundaries (0.30, 0.80)
- It provides strong guarantees that price behavior is unchanged for all utilization values

**Test Plan**: Capture original pricing behavior across utilization spectrum on UNFIXED code, then write property-based tests verifying fixed code produces identical prices.

**Test Cases**:
1. **Surge Regime Preservation**: For u_pred ∈ [0.80, 1.0], verify prices match original (within numerical tolerance)
2. **Neutral Regime Preservation**: For u_pred ∈ [0.30, 0.80], verify prices match original
3. **Discount Regime Preservation**: For u_pred ∈ [0.0, 0.30], verify prices match original (unaffected by fix)
4. **Epsilon Floor Preservation**: Verify epsilon floor value remains 1.0 (derived instead of hardcoded)

### Unit Tests

- Test parametric constant initialization (SURGE_THRESHOLD, DISCOUNT_THRESHOLD, midpoint_util, price_slope, epsilon_floor)
- Test surge formula with various u_pred values in [0.80, 1.0]
- Test neutral formula with various u_pred values in [0.30, 0.80]
- Test epsilon floor check with epsilon values above and below floor
- Test LLM and fallback path formula consistency at specific u_pred values

### Property-Based Tests

- Generate random utilization values across [0.0, 1.0] and verify price equivalence between original and fixed code
- Generate random theta parameters and verify formulas produce consistent results across LLM and fallback paths
- Test that parametric values are derived from config constants (not hardcoded) by checking constant definitions

### Integration Tests

- Test full pricing flow with LLM path using parametric formulas
- Test full pricing flow with fallback path using parametric formulas
- Test epsilon reduction logic with floor check across multiple steps
- Test that regime boundary cases (u_pred = 0.30, 0.80) produce correct prices
