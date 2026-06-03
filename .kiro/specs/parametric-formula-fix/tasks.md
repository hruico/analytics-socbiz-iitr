# Implementation Plan

- [ ] 1. Write bug condition exploration test
  - **Property 1: Fault Condition** - Parametric Formula Consistency
  - **CRITICAL**: This test MUST FAIL on unfixed code - failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior - it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate hardcoded values and formula inconsistencies
  - **Scoped PBT Approach**: Test concrete failing cases - surge formula inconsistency between LLM/fallback paths, neutral formula inconsistency, and epsilon floor hardcoding
  - Test that LLM surge uses multiplicative formula `baseline * (1 + alpha * (u_pred - 0.80))` with hardcoded 0.80
  - Test that fallback surge uses different additive formula instead of multiplicative
  - Test that LLM neutral uses hardcoded 0.55 midpoint and 8.0 slope
  - Test that fallback neutral uses hardcoded 0.55 but different slope calculation
  - Test that epsilon floor check uses hardcoded 1.0 literal
  - Run test on UNFIXED code at u_pred=0.90 (surge), u_pred=0.55 (neutral), and epsilon=0.95 (floor check)
  - **EXPECTED OUTCOME**: Test FAILS showing formula inconsistencies and hardcoded values
  - Document counterexamples: different formulas between paths, hardcoded literals in code
  - Mark task complete when test is written, run, and failures are documented
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6_

- [ ] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Price Value Equivalence
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for all utilization values across regimes
  - Capture original price outputs at key points: surge (u_pred=0.90), neutral (u_pred=0.55), discount (u_pred=0.20)
  - Write property-based test: for all u_pred in [0.0, 1.0], verify fixed code produces identical prices to original (within 0.01 tolerance)
  - Property-based testing generates many test cases for strong guarantees across utilization spectrum
  - Test surge regime: u_pred ∈ [0.80, 1.0] should preserve exact price values
  - Test neutral regime: u_pred ∈ [0.30, 0.80] should preserve exact price values
  - Test discount regime: u_pred ∈ [0.0, 0.30] should preserve exact price values (unaffected by fix)
  - Test regime boundaries: u_pred = 0.30 and u_pred = 0.80 edge cases
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [ ] 3. Fix parametric formula issues

  - [ ] 3.1 Add parametric constants to PricingAgent
    - Add class constants SURGE_THRESHOLD = 0.80 and DISCOUNT_THRESHOLD = 0.30 (derived from regime classification rules)
    - Add config constants UPPER_NEUTRAL = 17.0 and LOWER_NEUTRAL = 13.0 for neutral price bounds
    - Compute self.midpoint_util = (SURGE_THRESHOLD + DISCOUNT_THRESHOLD) / 2 in __init__ → 0.55
    - Compute self.price_slope = (UPPER_NEUTRAL - LOWER_NEUTRAL) / (SURGE_THRESHOLD - DISCOUNT_THRESHOLD) in __init__ → 8.0
    - _Bug_Condition: Hardcoded 0.80 threshold, 0.55 midpoint, and 8.0 slope in formulas_
    - _Expected_Behavior: All values derived from config constants ensuring parametric design_
    - _Preservation: Computed values must equal original hardcoded values (0.55 midpoint, 8.0 slope)_
    - _Requirements: 2.1, 2.6, 2.7_

  - [ ] 3.2 Replace hardcoded values in LLM pricing path
    - In _compute_price_from_regime() line 73, replace hardcoded 0.80 with self.SURGE_THRESHOLD in surge formula
    - In _compute_price_from_regime() line 71, replace hardcoded 0.55 and 8.0 with self.midpoint_util and self.price_slope in neutral formula
    - _Bug_Condition: LLM path uses hardcoded literals in surge and neutral formulas_
    - _Expected_Behavior: LLM path uses parametric constants derived from config_
    - _Preservation: Price outputs must remain identical for all u_pred values_
    - _Requirements: 2.1, 2.6, 2.7_

  - [ ] 3.3 Unify fallback path formulas with LLM path
    - In _deterministic_fallback() line 176, replace additive surge formula with multiplicative: `p_new = self.baseline * (1 + alpha * (u_pred - self.SURGE_THRESHOLD))`
    - Remove surge_scalar calculation from pricing logic (keep only for tracking if needed)
    - In _deterministic_fallback() line 199, replace neutral formula with: `p_new = self.baseline + (u_pred - self.midpoint_util) * self.price_slope`
    - _Bug_Condition: Fallback path uses different formulas than LLM path (additive vs multiplicative surge, different slope calculation for neutral)_
    - _Expected_Behavior: Both paths use identical formulas with parametric constants_
    - _Preservation: Price outputs must remain identical for all u_pred values_
    - _Requirements: 2.1, 2.6, 2.7_

  - [ ] 3.4 Add epsilon floor calculation to MonitoringAgent
    - In MonitoringAgent.__init__(), compute self.epsilon_floor = baseline / baseline (evaluates to 1.0 but derived)
    - This represents revenue breakeven: at neutral midpoint, price = baseline, so epsilon must stay above this ratio
    - In _deterministic_fallback() line 198, replace hardcoded 1.0 with self.epsilon_floor in epsilon_above_floor check
    - In _deterministic_fallback() line 207, update log message to use self.epsilon_floor instead of literal 1.0
    - _Bug_Condition: Epsilon floor uses hardcoded 1.0 literal instead of derived value_
    - _Expected_Behavior: Epsilon floor computed from baseline ensuring parametric design_
    - _Preservation: Floor value remains 1.0 for current baseline=15.0 config_
    - _Requirements: 2.2, 2.3, 2.4, 2.5_

  - [ ] 3.5 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Parametric Formula Consistency
    - **IMPORTANT**: Re-run the SAME test from task 1 - do NOT write a new test
    - The test from task 1 encodes the expected behavior (parametric formulas with consistent values across paths)
    - When this test passes, it confirms all formulas use config-derived constants
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms formulas are now parametric and consistent)
    - Verify LLM and fallback surge formulas are identical (multiplicative with SURGE_THRESHOLD)
    - Verify LLM and fallback neutral formulas are identical (use midpoint_util and price_slope)
    - Verify epsilon floor is derived from baseline calculation
    - _Requirements: 2.1, 2.2, 2.6, 2.7_

  - [ ] 3.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Price Value Equivalence
    - **IMPORTANT**: Re-run the SAME tests from task 2 - do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions in price behavior)
    - Confirm price outputs are identical across all utilization values and regimes
    - Confirm regime classification thresholds unchanged (0.80 surge, 0.30 discount)
    - Confirm parameter learning, reward calculation, and CSV export unchanged

- [ ] 4. Checkpoint - Ensure all tests pass
  - Verify all exploration tests pass (formulas are parametric and consistent)
  - Verify all preservation tests pass (price behavior unchanged)
  - Verify no regression in existing functionality
  - Ask the user if questions arise
