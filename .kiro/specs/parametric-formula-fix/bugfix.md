# Bugfix Requirements Document

## Introduction

The agentic EV pricing system currently contains three hardcoded formula issues that use magic numbers instead of deriving values from existing configuration parameters. These hardcoded values reduce the system's parametric nature and make it less maintainable. This bugfix addresses all three issues to ensure formulas are fully derived from config values and learned parameters.

## Bug Analysis

### Current Behavior (Defect)

**Issue 1: Surge Formula - Hardcoded Calculation**

1.1 WHEN u_pred > 0.80 in LLM pricing path THEN the system uses `p_new = self.baseline * (1 + alpha * (u_pred - 0.80))` with hardcoded 0.80 threshold value

1.2 WHEN u_pred > 0.80 in fallback path THEN the system uses `p_new = self.baseline + surge_scalar * alpha * (self.upper - self.baseline) / 100.0` instead of the multiplicative formula, creating inconsistency between LLM and fallback pricing

**Issue 2: Epsilon Floor - Magic Number**

1.3 WHEN monitoring agent proposes epsilon reduction THEN the system applies the reduction without checking if epsilon is above a revenue-breakeven floor, risking unprofitable pricing

1.4 WHEN epsilon reduction occurs THEN the system does not enforce a cooldown period, allowing rapid successive reductions that can destabilize pricing

**Issue 3: Neutral Formula - Hardcoded Literals**

1.5 WHEN u_pred is in neutral regime (0.30 to 0.80) in LLM pricing path THEN the system uses `p_new = self.baseline + (u_pred - 0.55) * 8.0` with hardcoded 0.55 midpoint and 8.0 slope values

1.6 WHEN u_pred is in neutral regime in fallback path THEN the system uses `p_new = self.baseline + (u_pred - 0.55) * (self.upper - self.lower) / 2.0` with different slope calculation, creating inconsistency

### Expected Behavior (Correct)

**Issue 1: Surge Formula - Parametric Formula**

2.1 WHEN u_pred > surge_threshold in both LLM and fallback paths THEN the system SHALL use the multiplicative formula `p_new = baseline * (1 + alpha * (u_pred - surge_threshold))` where surge_threshold is derived from config (0.80)

**Issue 2: Epsilon Floor - Derived from Revenue Breakeven**

2.2 WHEN monitoring agent proposes epsilon reduction THEN the system SHALL compute epsilon_floor = baseline / expected_neutral_price where expected_neutral_price = baseline + (midpoint_util - midpoint_util) * price_slope (evaluates to baseline at midpoint)

2.3 WHEN self.theta[0] <= epsilon_floor THEN the system SHALL skip epsilon reduction for that step and log the skip reason

2.4 WHEN epsilon reduction is applied THEN the system SHALL set a 5-step cooldown counter to prevent further epsilon reductions

2.5 WHEN cooldown counter > 0 THEN the system SHALL decrement the counter each step and skip any proposed epsilon reductions

**Issue 3: Neutral Formula - Config-Derived Constants**

2.6 WHEN u_pred is in neutral regime in both LLM and fallback paths THEN the system SHALL compute midpoint_util = (surge_threshold + discount_threshold) / 2 and price_slope = (upper_neutral - lower_neutral) / (surge_threshold - discount_threshold)

2.7 WHEN computing neutral pricing THEN the system SHALL use `p_new = baseline + (u_pred - midpoint_util) * price_slope` where upper_neutral=17.0 and lower_neutral=13.0 are config constants

### Unchanged Behavior (Regression Prevention)

3.1 WHEN regime classification logic executes THEN the system SHALL CONTINUE TO use thresholds u_pred > 0.80 for surge and u_pred < 0.30 for discount

3.2 WHEN XGBoost demand model makes predictions THEN the system SHALL CONTINUE TO operate unchanged

3.3 WHEN reward formula is computed THEN the system SHALL CONTINUE TO use existing reward weights and calculation logic

3.4 WHEN CSV export is performed THEN the system SHALL CONTINUE TO export all existing metrics unchanged

3.5 WHEN LLM call structure is invoked THEN the system SHALL CONTINUE TO use the existing prompt structure and response parsing

3.6 WHEN bounds clipping is applied THEN the system SHALL CONTINUE TO clip prices to [lower, upper] bounds from config

3.7 WHEN alpha and beta parameter updates are applied THEN the system SHALL CONTINUE TO use existing learning rate decay and parameter update logic

3.8 WHEN monitoring agent evaluates performance THEN the system SHALL CONTINUE TO use existing history window and evaluation criteria
