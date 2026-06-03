# Implementation Plan: Agentic EV Tariff Optimization System Rebuild

## Overview

This plan implements a three-agent autonomous optimization system (Demand_Agent with XGBoost, Pricing_Agent with LLM, Monitoring_Agent with LLM) that continuously learns to optimize EV charging tariffs through convergence-based execution. The system includes complete data preprocessing (JSON→CSV conversion, dataset fusion with station-level spatial features), all required features (Occupancy Density, Energy Cost per kWh, spatial clustering), three-output demand prediction (utilization, queue, congestion probability), LangGraph prompt templates, comprehensive EDA (weekday/weekend, volatility analysis, pricing efficiency trends), presentation asset generation, model selection/comparison, LLM cost budgeting, and causal inference safeguards.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create src/ directory with __init__.py and subdirectories: preprocessing/, agents/, utils/
  - Create tests/ directory for unit and property tests
  - Create data/ directory for raw/ and processed/ subdirectories
  - Create outputs/ directory for results (eda/, checkpoints/, figures/, presentation/)
  - Set up requirements.txt with dependencies: xgboost, pandas, numpy, pydantic, langchain, langchain-openai, langchain-anthropic, langchain-ollama, langgraph, hypothesis, pytest, pytest-cov, pytest-xdist (parallel tests), matplotlib, seaborn, scikit-learn, lightgbm, shap (for feature importance)
  - Create .gitignore for outputs/, data/raw/, __pycache__, .pytest_cache, .coverage, .hypothesis/
  - Create pytest.ini with hypothesis profiles: ci (max_examples=50, workers=4), local (max_examples=100), and hypothesis database caching enabled
  - _Requirements: 12.1, 2.3, 2.1, 2.2_

- [x] 2. Implement ACN-Data JSON to CSV conversion
  - [x] 2.1 Create src/preprocessing/acn_parser.py with ACNDataParser class
    - Parse raw ACN-Data JSON format with fields: connectionTime, disconnectTime, kWhDelivered, sessionID, stationID
    - Normalize timestamps to UTC hourly format
    - Aggregate sessions to hourly granularity by stationID and timestamp
    - Compute derived metrics: acn_sessions_count, acn_total_kwh, acn_avg_kwh_per_session
    - Compute acn_base_revenue = SUM(kWhDelivered * baseline_tariff) per hour
    - Compute acn_revenue_per_session = AVG(kWhDelivered * baseline_tariff) per hour
    - Compute acn_energy_cost_per_kwh = baseline_tariff (constant, from config)
    - Export to data/processed/acn_hourly.csv with columns: hourly_timestamp, stationID, acn_sessions_count, acn_total_kwh, acn_avg_kwh_per_session, acn_base_revenue, acn_revenue_per_session, acn_energy_cost_per_kwh
    - _Requirements: 7.1, 7.8_
  
  - [x] 2.2 Write unit tests for ACN parser
    - Test JSON parsing with valid session data (multiple sessions per hour)
    - Test hourly aggregation correctness (verify counts and sums)
    - Test handling of missing fields in JSON (sessionID, kWhDelivered)
    - Test timestamp normalization and UTC conversion
    - Test edge case: empty JSON input
    - _Requirements: 7.1_

- [x] 3. Implement UrbanEV dataset preprocessing with spatial features
  - [x] 3.1 Create src/preprocessing/urbanev_parser.py with UrbanEVParser class
    - Load UrbanEV (ST-EVCDP) CSV with fields: start_time, end_time, charging_volume, waiting_time, station_utilization, station_id, latitude, longitude
    - Normalize timestamps to UTC hourly format
    - Aggregate to hourly granularity by station_id and timestamp
    - Compute urban_mean_utilization = AVG(station_utilization) per hour (already in [0,1] range)
    - Compute urban_peak_queue = MAX(waiting_time / avg_session_duration) per hour
    - Compute urban_total_volume = SUM(charging_volume) per hour
    - Preserve station metadata: station_id, latitude, longitude for spatial analysis
    - Export to data/processed/urbanev_hourly.csv with columns: hourly_timestamp, station_id, latitude, longitude, urban_mean_utilization, urban_peak_queue, urban_total_volume
    - _Requirements: 7.1, 7.2, 7.3, 7.8_
  
  - [x] 3.2 Write unit tests for UrbanEV parser
    - Test CSV loading and schema validation
    - Test hourly aggregation across multiple stations (24,798 charging piles)
    - Test utilization clipping to [0, 1]
    - Test queue calculation with zero session duration
    - Test preservation of spatial coordinates
    - _Requirements: 7.2, 7.3_

- [ ] 4. Implement dataset alignment and fusion with spatial clustering
  - [x] 4.1 Create src/preprocessing/dataset_fusion.py with DatasetFusion class
    - Load acn_hourly.csv and urbanev_hourly.csv
    - Map station IDs between ACN-Data and UrbanEV schemas (handle different ID formats)
    - Align datasets by normalized hourly timestamp (outer join to preserve all timestamps)
    - Merge on timestamp to produce unified rows with all ACN + UrbanEV features
    - Add time_step column (sequential ordering 0, 1, 2, ...) for chronological operations
    - Add temporal features: hour_of_day (0-23), day_of_week (0-6), is_weekend (0 or 1)
    - Apply K-means clustering (k=5) on station coordinates (latitude, longitude) to create station_cluster_id feature for spatial demand modeling
    - Export to data/processed/unified_analytical_base.csv with columns: hourly_timestamp, time_step, acn_sessions_count, acn_total_kwh, acn_avg_kwh_per_session, acn_base_revenue, acn_revenue_per_session, acn_energy_cost_per_kwh, urban_mean_utilization, urban_peak_queue, urban_total_volume, hour_of_day, day_of_week, is_weekend, station_cluster_id
    - _Requirements: 7.1, 7.8, 7.9_
  
  - [ ]* 4.2 Write property test for dataset alignment
    - **Property: Timestamp Alignment Correctness**
    - **Validates: All rows in unified_analytical_base.csv have matching timestamps from both sources or explicit nulls**
    - **Requirements: 7.8, 7.9**
  
  - [ ] 4.3 Write unit tests for dataset fusion
    - Test timestamp matching between datasets with different time ranges
    - Test handling of non-overlapping time ranges (outer join behavior)
    - Test station ID mapping logic with mismatched schemas
    - Test K-means spatial clustering produces 5 distinct cluster IDs
    - Test temporal feature engineering (hour_of_day, is_weekend)
    - Test time_step sequential ordering
    - _Requirements: 7.8, 7.9_

- [ ] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [-] 6. Implement SystemConfig with validation, serialization, and cost controls
  - [x] 6.1 Create src/config.py with SystemConfig Pydantic model
    - Define all configuration fields with types and constraints (see design doc)
    - Implement field validators for pricing_bounds, baseline_tariff_per_kwh, theta_init (epsilon [0.1, 5.0], alpha [1.0, 10.0], beta [1.0, 10.0])
    - Add LLM provider enum: openai, anthropic, ollama
    - Add llm_cost_budget_usd field (default: 10.0) for API cost ceiling
    - Add max_llm_calls_per_step field (default: 5) to rate-limit LLM invocations
    - Add llm_token_budget field (default: 100000) to track cumulative token usage
    - Include convergence thresholds, learning rate schedule, reward weights
    - Validate baseline_tariff_per_kwh is within pricing_bounds
    - _Requirements: 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 13.5, 13.6, 13.7_
  
  - [ ]* 6.2 Write property test for configuration validation rules
    - **Property 35: Configuration Validation Rules**
    - **Validates: Requirements 12.7, 13.5, 13.6, 13.7**
  
  - [ ]* 6.3 Write property test for invalid configuration rejection
    - **Property 36: Invalid Configuration Rejection**
    - **Validates: Requirements 12.8, 13.2**
  
  - [ ]* 6.4 Write property test for configuration round-trip preservation
    - **Property 37: Configuration Round-Trip Preservation**
    - **Validates: Requirements 13.1, 13.4**
  
  - [ ] 6.5 Implement ConfigParser class with parse() and serialize() methods
    - Load JSON from file path and validate with SystemConfig
    - Write SystemConfig to JSON with pretty formatting (2-space indent)
    - Handle validation errors with descriptive messages identifying constraint violations
    - _Requirements: 13.1, 13.2, 13.3, 13.4_
  
  - [ ] 6.6 Write unit tests for configuration edge cases
    - Test specific invalid configs (negative bounds, theta out of range, baseline outside bounds)
    - Test missing required fields
    - Test round-trip with Indian market config (₹15 baseline, [10, 22] bounds)
    - _Requirements: 12.8, 13.2_

- [ ] 7. Implement data loading and feature engineering with Occupancy Density
  - [ ] 7.1 Create src/data_loader.py with load_dataset() function
    - Load unified_analytical_base.csv with schema validation
    - Check for required columns from ACN-Data and UrbanEV
    - Verify acn_energy_cost_per_kwh is present (Energy Cost per kWh from ACN-Data preprocessing)
    - Engineer is_peak_hour feature (hour_of_day in [7,8,9,17,18,19])
    - Engineer revenue_per_kwh = acn_base_revenue / acn_total_kwh when acn_total_kwh > 0
    - Engineer Occupancy_Density = acn_sessions_count / urban_total_volume when urban_total_volume > 0 (sessions per kWh volume, measures congestion relative to charging capacity)
    - Use station_cluster_id directly (already computed in fusion step) as spatial feature
    - Handle missing values: forward-fill utilization/queue, drop rows with missing revenue
    - Sort by time_step to preserve chronological order
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9_
  
  - [ ]* 7.2 Write property test for peak hour engineering
    - **Property 19: Peak Hour Engineering**
    - **Validates: Requirements 7.4**
  
  - [ ]* 7.3 Write property test for revenue per kWh calculation
    - **Property 20: Revenue Per kWh Calculation**
    - **Validates: Requirements 7.5**
  
  - [ ]* 7.4 Write property test for missing value handling
    - **Property 21: Missing Value Handling Strategy**
    - **Validates: Requirements 7.6**
  
  - [ ] 7.5 Implement train_test_split() function
    - Split at 80% boundary based on chronological time_step
    - No shuffling to preserve temporal dependencies
    - Return (train_df, test_df) tuple
    - Verify test set has at least 10 rows (raise InsufficientData error otherwise)
    - _Requirements: 7.7, 8.2_
  
  - [ ]* 7.6 Write property test for chronological train-test split
    - **Property 22: Chronological Train-Test Split**
    - **Validates: Requirements 7.7, 8.2**
  
  - [ ] 7.7 Write unit tests for data loading edge cases
    - Test handling of missing columns (MissingColumns error)
    - Test empty dataframe (InsufficientData error)
    - Test dataset with insufficient rows for split (<10 rows)
    - Test Occupancy_Density calculation with zero volume (should handle division by zero)
    - Test spatial clustering feature presence (station_cluster_id)
    - _Requirements: 6.1, 7.1_

- [ ] 8. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 9. Implement Demand_Agent with XGBoost MultiOutput (3 targets: utilization, queue, congestion probability)
  - [ ] 9.1 Create src/agents/demand.py with DemandAgent class
    - Define input features: acn_sessions_count, acn_total_kwh, acn_avg_kwh_per_session, hour_of_day, day_of_week, is_weekend, is_peak_hour, station_cluster_id, Occupancy_Density
    - Define 3 target variables: urban_mean_utilization (continuous [0,1]), urban_peak_queue (continuous [0,∞)), congestion_probability (continuous [0,1], derived from percentile of urban_peak_queue during training)
    - Compute congestion_probability target during training: if urban_peak_queue > 75th percentile, then 1.0, else 0.0 (binary classification label for MultiOutputRegressor)
    - Use XGBoost MultiOutputRegressor with base estimator: n_estimators=600, learning_rate=0.04, max_depth=6, subsample=0.80, colsample_bytree=0.75, reg_alpha=0.1, reg_lambda=1.5, tree_method="hist", random_state from config
    - Train on chronological first 80% (by time_step)
    - Clip predictions: u_pred to [0,1], q_pred to [0,∞), congestion_prob to [0,1]
    - _Requirements: 8.1, 8.2, 8.4, 8.6_
  
  - [ ] 9.2 Implement model selection and comparison
    - Compare XGBoost against LightGBM and RandomForestRegressor baselines
    - Perform 5-fold time-series cross-validation on training set (chronological splits)
    - Tune hyperparameters for XGBoost: grid search over n_estimators [400,600,800], learning_rate [0.03,0.04,0.05], max_depth [5,6,7]
    - Log comparison table: model name, RMSE (utilization), RMSE (queue), RMSE (congestion), R² (utilization), training time
    - Select best model based on average RMSE across all 3 targets
    - _Requirements: 8.1, 16.1_
  
  - [ ] 9.3 Implement evaluate() method
    - Compute RMSE, MAE, R² for all 3 prediction targets on test set
    - Log evaluation metrics with target-specific breakdowns
    - _Requirements: 8.5_
  
  - [ ] 9.4 Implement get_feature_importance() method
    - Extract feature importance from trained XGBoost model
    - Rank features and return top 5 with importance scores
    - Optionally compute SHAP values for better interpretability
    - Log feature importance table
    - _Requirements: 8.7_
  
  - [ ]* 9.5 Write property test for prediction clipping
    - **Property 23: Prediction Clipping**
    - **Validates: Requirements 8.6**
  
  - [ ] 9.6 Write unit tests for Demand_Agent
    - Test training with synthetic data (verify no errors)
    - Test prediction output shape (3 values per row)
    - Test utilization clipping to [0,1]
    - Test queue clipping to [0,∞) (no negative queues)
    - Test congestion probability output range [0,1]
    - Test feature importance extraction returns 5 features
    - _Requirements: 8.1, 8.6_

- [ ] 10. Implement LLM provider abstraction with cost tracking
  - [ ] 10.1 Create src/utils/llm_provider.py with LLMProviderWrapper class
    - Implement provider factory for OpenAI, Anthropic, Ollama (see design doc)
    - Validate API credentials at instantiation (check environment variables)
    - Implement invoke_with_retry() with exponential backoff (2^attempt * backoff_seconds)
    - Classify errors as transient (rate limit, timeout, 503, 429) or permanent (invalid key, 401, 404)
    - Track cumulative token usage across all invocations in shared counter
    - Check token usage against llm_token_budget before each call, raise QuotaExceededError if exceeded
    - Log all LLM API calls: provider, model, tokens_used, latency_ms, success_status, cumulative_tokens
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.7, 6.4, 6.5, 14.2_
  
  - [ ]* 10.2 Write property test for error classification
    - **Property 17: Error Classification**
    - **Validates: Requirements 6.4**
  
  - [ ]* 10.3 Write property test for agent retry with exponential backoff
    - **Property 11: Agent Retry with Exponential Backoff**
    - **Validates: Requirements 3.6, 6.5**
  
  - [ ] 10.4 Write unit tests for LLM provider
    - Test provider factory creates correct provider for each enum value
    - Test missing API key raises EnvironmentError
    - Test retry logic with mock transient failures (3 attempts)
    - Test permanent error raises immediately without retry
    - Test token budget enforcement (QuotaExceededError when budget exceeded)
    - Test cumulative token tracking across multiple calls
    - _Requirements: 2.7, 6.1, 6.4_

- [ ] 11. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 12. Implement Pricing_Agent with LangGraph and explicit prompt templates
  - [ ] 12.1 Create src/agents/pricing.py with PricingAgent class
    - Define ForecastState input model: timestamp, u_pred, q_pred, u_actual, q_actual, kwh_delivered, hour_of_day, is_weekend, congestion_prob
    - Define PricingDecision output model: p_new, regime (surge|neutral|discount), surge_scalar, discount_scalar, elasticity_used, rationale
    - Implement LangGraph state machine with 3 nodes: analyse_state, compute_price, validate
    - _Requirements: 3.2, 5.1, 5.2_
  
  - [ ] 12.2 Implement analyse_state node with explicit prompt template
    - **Input contract**: ForecastState, current theta [ε, α, β], pricing_bounds, baseline_tariff
    - **Prompt template**: "You are analyzing demand conditions for EV charging tariff optimization. Current utilization prediction: {u_pred:.2%}, queue prediction: {q_pred:.1f}, congestion probability: {congestion_prob:.2%}, time: {hour_of_day}h ({'weekend' if is_weekend else 'weekday'}). Elasticity parameter: {epsilon:.2f}. Baseline tariff: ₹{baseline:.2f}/kWh. Based on these conditions, what pricing regime is appropriate? Consider: surge pricing when utilization exceeds 80% to manage congestion, discount pricing when utilization is below 30% to attract demand, neutral pricing otherwise. Provide strategic reasoning in 2-3 sentences."
    - **Output**: Structured reasoning text with regime recommendation
    - Store reasoning in state.analysis_reasoning
    - _Requirements: 5.1, 5.2_
  
  - [ ] 12.3 Implement compute_price node with explicit prompt template
    - **Input contract**: ForecastState with analysis_reasoning, current theta, pricing_bounds, baseline_tariff
    - **Prompt template**: "Based on your analysis: '{analysis_reasoning}', determine the optimal per-kWh tariff. Constraints: price must be in [₹{lower:.2f}, ₹{upper:.2f}]. Baseline is ₹{baseline:.2f}/kWh. Current parameters: alpha (surge intensity) = {alpha:.2f}, beta (discount intensity) = {beta:.2f}. If surge regime: price should exceed baseline. If discount regime: price should be below baseline. Respond with JSON: {{\"p_new\": <float>, \"regime\": \"surge\"|\"neutral\"|\"discount\", \"surge_scalar\": <float 0-1>, \"discount_scalar\": <float 0-1>, \"rationale\": \"<1-2 sentences>\"}}. Do not include any other text."
    - **Output**: JSON string conforming to PricingDecision schema
    - Parse JSON, handle parsing errors with structured exception
    - _Requirements: 5.1, 5.2, 3.2_
  
  - [ ] 12.4 Implement validate node
    - Check format validation: JSON schema compliance with PricingDecision
    - Check bounds validation: pricing_bounds[0] ≤ p_new ≤ pricing_bounds[1]
    - Check regime consistency: surge requires u_pred > 0.80 AND p_new > baseline, discount requires u_pred < 0.30 AND p_new < baseline, neutral requires u_pred in [0.30, 0.80]
    - Return (validated_decision, success=True) or (None, success=False)
    - _Requirements: 3.3, 15.5_
  
  - [ ] 12.5 Implement deterministic_pricing_fallback()
    - Use fallback formula from design doc (surge/discount/neutral scalars)
    - Log fallback invocation with reason (format failure, validation failure, business logic failure)
    - Return PricingDecision with fallback_used=True flag
    - _Requirements: 3.7_
  
  - [ ]* 12.6 Write property test for pricing decision bounds and regime consistency
    - **Property 9: Pricing Decision Bounds and Regime Consistency**
    - **Validates: Requirements 3.3, 15.5**
  
  - [ ]* 12.7 Write property test for fallback on validation failure
    - **Property 12: Fallback on Validation Failure**
    - **Validates: Requirements 3.7**
  
  - [ ] 12.8 Write unit tests for Pricing_Agent
    - Test surge regime assignment (u_pred=0.85 → regime="surge", p_new > baseline)
    - Test discount regime assignment (u_pred=0.25 → regime="discount", p_new < baseline)
    - Test neutral regime assignment (u_pred=0.55 → regime="neutral")
    - Test bounds enforcement (clip to [lower, upper])
    - Test LLM failure triggers fallback (mock invalid JSON response)
    - Test congestion probability influences surge decision
    - _Requirements: 3.2, 3.3, 3.7_


- [ ] 13. Implement Monitoring_Agent with ground-truth outcome evaluation
  - [ ] 13.1 Create src/agents/monitoring.py with MonitoringAgent class
    - Define MonitoringInput model: step, pricing_decision, state, revenue_baseline, revenue_new, demand_shift, utilization_baseline, utilization_new, queue_baseline, queue_new, current_theta, recent_history (last 3-5 steps)
    - Define LearningUpdate output model: delta_epsilon, delta_alpha, delta_beta, reward, revenue_gain_pct, charger_utilisation, avg_wait_reduction, pricing_efficiency, demand_shift, reflection
    - _Requirements: 3.4, 4.1, 9.7_
  
  - [ ] 13.2 Implement evaluate_and_propose() method with ground-truth collection
    - Collect ground-truth outcomes from test dataset row (u_actual, q_actual, kwh_delivered) vs. predictions (u_pred, q_pred)
    - Compute realized metrics deterministically in Python (not via LLM): revenue_gain_pct, demand_shift, utilization_new (elasticity-adjusted), pricing_efficiency, avg_wait_reduction
    - Provide LLM with summary of realized outcomes, pricing decision, and recent trend (last 3 steps)
    - Request LLM to propose parameter adjustments [Δε, Δα, Δβ] with reasoning
    - Prompt: "You are evaluating a pricing decision. Price: ₹{p_new:.2f}/kWh (regime: {regime}). Outcomes: revenue gain {revenue_gain_pct:+.1f}%, actual utilization {u_actual:.2%} (predicted {u_pred:.2%}), actual queue {q_actual:.1f} (predicted {q_pred:.1f}). Recent trend (last 3 steps): {trend_summary}. Current parameters: epsilon={eps:.2f}, alpha={alpha:.2f}, beta={beta:.2f}. Should we adjust parameters? If high utilization persists (>80%), increase alpha. If low utilization persists (<30%), increase beta. If revenue is declining, reduce epsilon. Respond with JSON: {{\"delta_epsilon\": <float>, \"delta_alpha\": <float>, \"delta_beta\": <float>, \"reflection\": \"<2-3 sentences>\"}}. Deltas must satisfy |Δε|≤0.05, |Δα|≤0.10, |Δβ|≤0.10."
    - Parse JSON, validate business logic (directional correctness)
    - _Requirements: 3.4, 5.3, 5.4, 5.5_
  
  - [ ] 13.3 Implement business logic validation for parameter updates
    - Check directional correctness: if u_actual > 0.80 for 3+ consecutive steps, delta_alpha must be > 0 or 0
    - Check directional correctness: if u_actual < 0.30 for 3+ consecutive steps, delta_beta must be > 0 or 0
    - Check directional correctness: if revenue_gain_pct < 0 for 3+ consecutive steps, delta_epsilon must be ≤ 0
    - Check magnitude constraints: |delta_epsilon| ≤ 0.05, |delta_alpha| ≤ 0.10, |delta_beta| ≤ 0.10
    - Return validation result with pass/fail and reason
    - _Requirements: 3.4, 4.4, 4.5, 4.6_
  
  - [ ] 13.4 Implement compute_fallback_update() with deterministic rules
    - Analyze recent_history (last 3 steps)
    - If revenue_gain_pct < 0 for all 3 steps: delta_epsilon = -0.02
    - If u_actual > 0.80 during surge for all 3 steps: delta_alpha = +0.10
    - If u_actual < 0.30 during discount for all 3 steps: delta_beta = +0.10
    - Otherwise: return zeros (no change)
    - Log fallback reasoning
    - _Requirements: 4.4, 4.5, 4.6_
  
  - [ ]* 13.5 Write property test for monitoring parameter update validation
    - **Property 10: Monitoring Parameter Update Validation**
    - **Validates: Requirements 3.4**
  
  - [ ]* 13.6 Write property test for parameter update precedence
    - **Property 13: Parameter Update Precedence**
    - **Validates: Requirements 4.1**
  
  - [ ] 13.7 Write unit tests for Monitoring_Agent
    - Test valid agent proposal is accepted (directionally correct deltas)
    - Test invalid proposal triggers fallback (delta_alpha negative when high util persists)
    - Test fallback computation (3 consecutive negative revenue → delta_epsilon = -0.02)
    - Test magnitude constraint enforcement (|Δε| ≤ 0.05)
    - Test ground-truth collection from test dataset row
    - _Requirements: 3.4, 4.1, 4.3_


- [ ] 14. Implement MetricsComputation engine with causal inference safeguards
  - [ ] 14.1 Create src/utils/metrics.py with MetricsEngine class
    - Implement compute_step_metrics() with all formulas from design doc Appendix B
    - Compute demand_shift = -epsilon * (p_new - baseline) / baseline
    - Compute adjusted_demand_factor = max(0.05, 1 + demand_shift)
    - Compute revenue_new = p_new * kwh_delivered * adjusted_demand_factor
    - Compute revenue_gain_pct = (revenue_new - revenue_baseline) / revenue_baseline * 100
    - Compute utilization_new = clip(u_actual + demand_shift * 0.1, 0.0, 1.0)
    - Compute utilization_improvement = (utilization_new - u_actual) * 100
    - Compute pricing_efficiency = revenue_new / kwh_delivered
    - Compute customer_response_rate = demand_shift * 100
    - Compute avg_wait_reduction = -demand_shift * q_actual
    - Compute queue_penalty = max(0, q_actual - baseline_mean_queue) * 100
    - Compute reward = w1 * revenue_gain_pct + w2 * utilization_improvement - w3 * queue_penalty
    - Return StepMetrics object
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 5.5_
  
  - [ ] 14.2 Add causal_claim_detector() function
    - Scan metric logs and evaluation summary for causal language: "causes", "due to", "because of", "results in", "leads to"
    - Flag detected causal claims with WARNING log: "Causal claim detected: '{claim_text}'. Causal inference is not justified without controlled experiments."
    - Insert disclaimer in evaluation summary: "All reported associations are correlational. Causal claims are not supported by this observational study design."
    - _Requirements: Problem statement warning_
  
  - [ ]* 14.3 Write property tests for metric formulas
    - **Property 24: Revenue Gain Percentage Formula**
    - **Property 25: Demand Shift Elasticity Formula**
    - **Property 26: Utilization Elasticity Adjustment**
    - **Property 27: Pricing Efficiency Calculation**
    - **Property 28: Customer Response Rate**
    - **Property 29: Average Wait Reduction**
    - **Property 30: Reward Function Computation**
    - **Validates: Requirements 9.1-9.7**
  
  - [ ] 14.4 Write unit tests for metrics engine
    - Test revenue calculation with elasticity adjustment
    - Test demand_shift floor at 5% (extreme price increase scenario)
    - Test reward computation with default weights [1.0, 0.5, 0.3]
    - Test causal claim detection (flagging text with "causes" or "results in")
    - Test disclaimer insertion in evaluation summary
    - _Requirements: 9.1, 9.7_

- [ ] 15. Implement ConvergenceChecker with multi-objective tracking
  - [ ] 15.1 Create src/utils/convergence.py with ConvergenceChecker class
    - Define ConvergenceState: revenue_history (deque, maxlen=50), theta_history (deque, maxlen=50), utilization_history (deque, maxlen=50), queue_history (deque, maxlen=50), consecutive_convergence_steps (int)
    - Implement update(metrics, theta) to append to history deques
    - Implement check() to evaluate all 4 convergence conditions (see design doc)
    - Check 1: var(revenue_history) < revenue_variance_threshold
    - Check 2: max(|theta_deltas|) < parameter_delta_threshold (compare consecutive theta vectors)
    - Check 3: std(utilization_history) < utilization_std_threshold AND max(utilization_history) < max_utilization_threshold
    - Check 4: mean(queue_history) < (1 - queue_reduction_target) * baseline_mean_queue
    - Increment consecutive_convergence_steps if all 4 met, else reset to 0
    - Return ConvergenceResult with met=True if consecutive_convergence_steps >= convergence_window
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [ ] 15.2 Implement get_status_summary() for logging
    - Return dict with current values of all 4 metrics and their thresholds
    - Include consecutive_convergence_steps and steps_remaining (convergence_window - consecutive)
    - _Requirements: 1.7, 14.4_
  
  - [ ]* 15.3 Write property tests for convergence conditions
    - **Property 1: Convergence Termination Correctness**
    - **Property 2: Revenue Stability Computation**
    - **Property 3: Parameter Stability Tracking**
    - **Property 4: Utilization Health Check**
    - **Property 5: Queue Reduction Verification**
    - **Validates: Requirements 1.1-1.5**
  
  - [ ] 15.4 Write unit tests for ConvergenceChecker
    - Test convergence detection with synthetic stable sequence (50 steps, all conditions met)
    - Test non-convergence with unstable revenue (var > threshold)
    - Test consecutive_steps reset when any condition fails
    - Test max_iterations termination (1000 steps without convergence)
    - _Requirements: 1.1, 1.6_

- [ ] 16. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.


- [ ] 17. Implement System Orchestrator with parameter update arbitration
  - [ ] 17.1 Create src/orchestrator.py with SystemOrchestrator class
    - Initialize agents: DemandAgent, PricingAgent, MonitoringAgent
    - Initialize ConvergenceChecker, MetricsEngine, LLMProviderWrapper
    - Load SystemConfig and validate at startup
    - Load test dataset (chronological last 20% from train_test_split)
    - Compute baseline_mean_queue from test dataset for convergence criterion 4
    - _Requirements: 1.1, 12.1, 12.7_
  
  - [ ] 17.2 Implement run_optimization_loop()
    - Iterate over test dataset rows (max_iterations limit)
    - For each step: call demand_agent.predict() → pricing_agent.compute_tariff() → metrics_engine.compute_step_metrics() → monitoring_agent.evaluate_and_propose()
    - Enforce max_llm_calls_per_step rate limit (count Pricing + Monitoring LLM calls per step)
    - If rate limit exceeded, skip LLM agents for that step and use deterministic fallbacks
    - Track cumulative LLM tokens against llm_token_budget, raise QuotaExceededError if exceeded
    - After each step, call convergence_checker.update() and convergence_checker.check()
    - If convergence met, log termination reason and break loop
    - If step == max_iterations, log MaxIterationsExceeded warning and break loop
    - Export agentic_outcomes.csv after loop completion
    - _Requirements: 1.1, 1.6, 9.9, 15.1_
  
  - [ ] 17.3 Implement apply_parameter_update() with precedence arbitration
    - Accept monitoring_proposal (LearningUpdate), validation_passed (bool), fallback_delta (np.ndarray)
    - If validation_passed: delta = [proposal.delta_epsilon, proposal.delta_alpha, proposal.delta_beta], source = "agent"
    - Elif fallback_delta is not None: delta = fallback_delta, source = "fallback"
    - Else: delta = zeros, source = "no_change"
    - Compute learning_rate with decay: eta = eta_0 / (1 + decay * step)
    - Apply update: theta_new = theta_old + eta * delta
    - Log parameter update: source, old_theta, proposed_delta, final_applied_delta, learning_rate
    - Call pricing_agent.apply_update(delta_scaled)
    - _Requirements: 4.1, 4.2, 4.3, 4.7, 4.8, 14.3_
  
  - [ ] 17.4 Implement theta checkpointing
    - At steps where step % 100 == 0, save theta to outputs/checkpoints/theta_step_{step}.json
    - Include metadata: step, timestamp, convergence_status
    - _Requirements: 15.4_
  
  - [ ]* 17.5 Write property tests for orchestrator behavior
    - **Property 14: Valid Agent Proposal Application**
    - **Property 15: Fallback on Agent Validation Failure**
    - **Property 16: Parameter Update Logging Structure**
    - **Property 41: Graceful Agent Failure Handling**
    - **Property 42: Theta Checkpointing Frequency**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.7, 15.1, 15.4**
  
  - [ ] 17.6 Write unit tests for orchestrator
    - Test single optimization step executes all agents in sequence
    - Test convergence detection triggers loop termination (mock 50 stable steps)
    - Test max_iterations termination (mock 1000 steps without convergence)
    - Test agent failure uses fallback and continues to next step (no crash)
    - Test parameter update arbitration (agent success → agent delta, agent failure → fallback delta)
    - Test theta checkpointing at step 100, 200, etc.
    - Test LLM rate limiting (max_llm_calls_per_step enforcement)
    - Test LLM token budget enforcement (QuotaExceededError)
    - _Requirements: 1.1, 1.6, 4.1, 15.1, 15.4_

- [ ] 18. Implement comprehensive EDA generator with weekday/weekend and volatility analysis
  - [ ] 18.1 Create src/eda.py with EDAGenerator class
    - Accept unified_analytical_base.csv and trained DemandAgent as inputs
    - Create outputs/eda/ directory structure
    - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6_
  
  - [ ] 18.2 Implement generate_temporal_analysis()
    - Hourly utilization heatmap (day_of_week vs hour_of_day)
    - Daily utilization time series with trend line (rolling 7-day average)
    - Peak hour identification histogram (utilization distribution by is_peak_hour)
    - Export as: temporal_hourly_heatmap.png, temporal_daily_series.png, temporal_peak_histogram.png (300 DPI)
    - _Requirements: 17.1_
  
  - [ ] 18.3 Implement generate_weekday_weekend_comparison()
    - Compare charging patterns: weekday (is_weekend=0) vs. weekend (is_weekend=1)
    - Plot: mean utilization by hour for weekday vs. weekend (line plot overlay)
    - Plot: session count distribution for weekday vs. weekend (box plot)
    - Plot: queue length comparison by hour for weekday vs. weekend
    - Compute summary statistics: mean utilization (weekday vs. weekend), mean sessions, mean queue
    - Export as: weekday_weekend_utilization.png, weekday_weekend_sessions.png, weekday_weekend_queue.png (300 DPI)
    - _Requirements: EDA gap from issues.txt_
  
  - [ ] 18.4 Implement generate_volatility_analysis()
    - Define shoulder periods: hours [6, 10-16, 20-22] (between peak and off-peak)
    - Compute volatility metrics for peak / shoulder / off-peak periods:
      - Rolling standard deviation (7-day window) of utilization
      - Coefficient of variation (std / mean) for each period
      - Utilization range (max - min) for each period
    - Plot: rolling std time series with period annotations (peak/shoulder/off-peak zones color-coded)
    - Plot: box plot of utilization by period (peak / shoulder / off-peak)
    - Export as: volatility_rolling_std.png, volatility_by_period.png (300 DPI)
    - Export summary to outputs/eda/volatility_summary.json
    - _Requirements: EDA gap from issues.txt_
  
  - [ ] 18.5 Implement generate_utilization_distribution()
    - Utilization histogram with quartile markers (25th, 50th, 75th percentiles)
    - Utilization box plot by day_of_week
    - Regime frequency bar chart (surge / neutral / discount hour counts based on u > 0.80, u < 0.30 thresholds)
    - Export as: utilization_histogram.png, utilization_by_day.png, regime_frequency.png (300 DPI)
    - _Requirements: 17.2_
  
  - [ ] 18.6 Implement generate_feature_importance()
    - Extract feature importance from trained DemandAgent
    - Bar chart: top 10 features ranked by importance score
    - If SHAP available: SHAP summary plot for top 5 features
    - Export as: feature_importance.png, feature_shap_summary.png (300 DPI)
    - _Requirements: 17.3_
  
  - [ ] 18.7 Implement generate_correlation_analysis()
    - Correlation heatmap: revenue vs utilization vs queue vs sessions vs Occupancy_Density
    - Scatter plot: utilization vs queue_length with regime coloring
    - Scatter plot: acn_sessions_count vs acn_base_revenue with trend line
    - Export as: correlation_heatmap.png, utilization_queue_scatter.png, sessions_revenue_scatter.png (300 DPI)
    - _Requirements: 17.4_
  
  - [ ] 18.8 Implement generate_queue_analysis()
    - Queue length distribution histogram
    - Queue vs time-of-day scatter plot (hour_of_day on x-axis)
    - Congestion event identification: flag hours where queue > 75th percentile
    - Congestion frequency by hour_of_day bar chart
    - Export as: queue_distribution.png, queue_by_hour.png, congestion_frequency.png (300 DPI)
    - _Requirements: 17.5_
  
  - [ ] 18.9 Implement generate_summary_statistics()
    - Compute descriptive statistics: mean, std, min, max, quartiles for all key metrics
    - Export to outputs/eda/eda_summary.json
    - _Requirements: 17.7_
  
  - [ ] 18.10 Implement run_full_analysis()
    - Call all generate_* methods in sequence
    - Log completion of each analysis component
    - _Requirements: 17.6_
  
  - [ ] 18.11 Write unit tests for EDA generator
    - Test all plots are generated and saved to outputs/eda/ (check file existence)
    - Test PNG files have 300 DPI resolution (read metadata)
    - Test eda_summary.json has expected keys (mean_utilization, mean_queue, etc.)
    - Test weekday/weekend comparison produces 3 plots
    - Test volatility analysis defines shoulder period correctly (hours [6, 10-16, 20-22])
    - _Requirements: 17.6, 17.7_


- [ ] 19. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 20. Implement evaluation metrics and benchmarking with pricing efficiency trend
  - [ ] 20.1 Create src/evaluation.py with EvaluationEngine class
    - Accept agentic_outcomes.csv as input
    - Compute aggregate metrics from optimization results
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_
  
  - [ ] 20.2 Implement compute_revenue_uplift()
    - Compute mean(revenue_gain_pct) across all optimization steps
    - Return revenue uplift percentage relative to configured baseline
    - _Requirements: 10.3_
  
  - [ ]* 20.3 Write property test for revenue uplift aggregation
    - **Property 31: Revenue Uplift Aggregation**
    - **Validates: Requirements 10.3**
  
  - [ ] 20.4 Implement compute_off_peak_uplift()
    - Filter steps where hour_of_day NOT in [7,8,9,17,18,19] AND regime == "discount"
    - Compute percentage increase in mean utilization during discount regime hours vs. baseline off-peak utilization
    - _Requirements: 10.4_
  
  - [ ]* 20.5 Write property test for off-peak utilization uplift
    - **Property 32: Off-Peak Utilization Uplift**
    - **Validates: Requirements 10.4**
  
  - [ ] 20.6 Implement compute_congestion_reduction()
    - Compute mean(queue_lengths) from optimization results
    - Compute baseline_mean_queue from test dataset
    - Return (baseline_mean_queue - mean_optimized_queue) / baseline_mean_queue * 100
    - _Requirements: 10.5_
  
  - [ ]* 20.7 Write property test for congestion reduction metric
    - **Property 33: Congestion Reduction Metric**
    - **Validates: Requirements 10.5**
  
  - [ ] 20.8 Implement compute_agent_decision_quality()
    - Compute format_validation_pass_rate = count(agent_success) / total_steps * 100
    - Compute business_logic_pass_rate = count(agent_success AND not fallback_used) / total_steps * 100
    - Compare against 80% target threshold from requirements
    - Return metrics with pass/fail flag
    - _Requirements: 10.6, 3.5, 3.8_
  
  - [ ] 20.9 Implement generate_pricing_efficiency_trend()
    - Extract pricing_efficiency time series from agentic_outcomes.csv (step, pricing_efficiency)
    - Plot time series with rolling 10-step average overlay
    - Annotate with trend direction (increasing/decreasing/stable)
    - Export as outputs/figures/pricing_efficiency_trend.png (300 DPI)
    - Compute efficiency improvement: (mean(last_50_steps) - mean(first_50_steps)) / mean(first_50_steps) * 100
    - Log efficiency improvement percentage
    - _Requirements: Evaluation gap from issues.txt_
  
  - [ ] 20.10 Implement run_benchmarking()
    - Compare optimized system against 3 baselines: fixed baseline, time-of-day, deterministic formula
    - For each baseline, simulate pricing decisions on test dataset
    - Compute metrics: mean revenue_gain_pct, mean utilization, mean queue, mean reward
    - Perform paired t-test between agentic system and each baseline
    - Export results to outputs/benchmark_comparison.csv
    - _Requirements: 16.1, 16.2, 16.3, 16.4, 16.5_
  
  - [ ] 20.11 Implement generate_evaluation_summary()
    - Aggregate all metrics: revenue_uplift, off_peak_uplift, congestion_reduction, agent_decision_quality, final_theta, convergence_status
    - Include causal inference disclaimer: "All reported associations are correlational. Causal claims are not supported by this observational study design."
    - Export to outputs/evaluation_summary.json
    - _Requirements: 10.7_
  
  - [ ] 20.12 Write unit tests for evaluation engine
    - Test revenue uplift computation with synthetic outcomes
    - Test off-peak uplift filters discount regime correctly
    - Test congestion reduction formula
    - Test agent decision quality metrics (format pass rate, business logic pass rate)
    - Test pricing efficiency trend plot generation (file exists, 300 DPI)
    - Test benchmark comparison against fixed baseline
    - Test causal disclaimer inclusion in evaluation summary
    - _Requirements: 10.3, 10.4, 10.5, 10.6, 16.4_

- [ ] 21. Implement presentation asset generator
  - [ ] 21.1 Create src/presentation.py with PresentationAssetGenerator class
    - Generate slide-ready figures and summary tables for 5-7 slide deck
    - Target slides: (1) Data Landscape, (2) EDA Findings, (3) Demand Modeling Results, (4) Tariff Optimization Logic, (5) Monitoring Performance, (6) Business Implications
    - _Requirements: Presentation gap from issues.txt_
  
  - [ ] 21.2 Generate Slide 1 assets: Data Landscape
    - Table: dataset provenance (ACN-Data sessions, UrbanEV utilization/queue, merge strategy)
    - Table: feature summary (5 engineered features with descriptions)
    - Figure: temporal coverage plot (dataset start/end dates, train/test split boundary)
    - Export as: slide1_data_provenance_table.csv, slide1_feature_summary_table.csv, slide1_temporal_coverage.png (300 DPI)
    - _Requirements: Presentation gap_
  
  - [ ] 21.3 Generate Slide 2 assets: EDA Findings
    - Figure: weekday vs. weekend utilization comparison (most striking pattern)
    - Figure: volatility by period (peak/shoulder/off-peak box plot)
    - Table: key statistics (mean utilization, peak queue, surge/neutral/discount hour counts)
    - Export as: slide2_weekday_weekend.png, slide2_volatility.png, slide2_key_stats_table.csv (300 DPI)
    - _Requirements: Presentation gap_
  
  - [ ] 21.4 Generate Slide 3 assets: Demand Modeling Results
    - Table: model comparison (XGBoost vs. LightGBM vs. RandomForest with RMSE, R²)
    - Figure: feature importance bar chart (top 5 features)
    - Table: prediction accuracy (RMSE, MAE, R² for utilization, queue, congestion probability)
    - Export as: slide3_model_comparison_table.csv, slide3_feature_importance.png, slide3_accuracy_table.csv (300 DPI)
    - _Requirements: Presentation gap_
  
  - [ ] 21.5 Generate Slide 4 assets: Tariff Optimization Logic
    - Diagram: agent pipeline flowchart (Demand → Pricing → Monitoring with feedback loop) - use simple matplotlib/seaborn visualization
    - Table: regime rules (surge/neutral/discount thresholds and pricing bounds)
    - Table: parameter evolution (initial theta vs. final theta after convergence)
    - Export as: slide4_agent_pipeline.png, slide4_regime_rules_table.csv, slide4_parameter_evolution_table.csv (300 DPI)
    - _Requirements: Presentation gap_
  
  - [ ] 21.6 Generate Slide 5 assets: Monitoring Performance
    - Figure: convergence metrics over time (4 subplots: revenue variance, parameter delta, utilization std, queue mean with threshold lines)
    - Figure: pricing efficiency trend (from evaluation engine)
    - Table: agent decision quality (format pass rate, business logic pass rate, fallback frequency)
    - Export as: slide5_convergence_metrics.png, slide5_pricing_efficiency_trend.png, slide5_agent_quality_table.csv (300 DPI)
    - _Requirements: Presentation gap_
  
  - [ ] 21.7 Generate Slide 6 assets: Business Implications
    - Table: benchmark comparison (agentic vs. fixed baseline vs. time-of-day vs. deterministic formula)
    - Table: multi-objective performance summary (revenue uplift, off-peak uplift, congestion reduction with %-change)
    - Figure: reward distribution (histogram showing optimization reward scores)
    - Export as: slide6_benchmark_table.csv, slide6_multi_objective_table.csv, slide6_reward_distribution.png (300 DPI)
    - _Requirements: Presentation gap_
  
  - [ ] 21.8 Generate presentation README
    - Create outputs/presentation/README.md with instructions for assembling the deck
    - List all asset files with suggested slide placement
    - Include causal inference disclaimer and reproducibility notes
    - _Requirements: Presentation gap_
  
  - [ ] 21.9 Write unit tests for presentation asset generator
    - Test all slide assets are generated (check file existence for 6 slides worth of assets)
    - Test all figures have 300 DPI resolution
    - Test all CSV tables have expected columns
    - Test README.md includes causal disclaimer
    - _Requirements: Presentation gap_


- [ ] 22. Implement CLI and logging infrastructure
  - [ ] 22.1 Create src/cli.py with main() entry point
    - Accept command-line arguments: --config <path>, --seed <int>, --log-level <DEBUG|INFO|WARNING|ERROR>, --mode <train|optimize|eda|evaluate|presentation>
    - Load SystemConfig from --config path
    - Validate API credentials at startup (check environment variables for selected LLM provider)
    - Set random seed for numpy and XGBoost
    - Initialize logging with specified log level and structured format
    - Route to appropriate mode: train (train Demand_Agent only), optimize (full optimization loop), eda (EDA only), evaluate (evaluation only), presentation (generate assets)
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 14.5_
  
  - [ ] 22.2 Implement structured logging infrastructure
    - Create src/utils/logging_config.py with setup_logging()
    - Configure logging format: timestamp, level, module, message (JSON format for structured logs)
    - Configure log rotation: daily rotation, keep 30 days of logs
    - Log to both file (outputs/logs/system.log) and console
    - _Requirements: 14.1, 14.2, 14.3, 14.4_
  
  - [ ]* 22.3 Write property tests for logging output
    - **Property 38: Optimization Step Logging**
    - **Property 39: LLM API Call Logging**
    - **Property 40: Convergence Check Logging**
    - **Property 43: Graceful Degradation Event Logging**
    - **Validates: Requirements 1.7, 14.1, 14.2, 14.4, 15.6**
  
  - [ ] 22.4 Write unit tests for CLI
    - Test config loading with valid config file
    - Test config loading with invalid config (raises ConfigurationError)
    - Test API credential validation (missing OPENAI_API_KEY raises EnvironmentError)
    - Test seed setting (verify XGBoost random_state matches)
    - Test mode routing (--mode train calls train_demand_agent only)
    - Test log level configuration (--log-level DEBUG enables debug logs)
    - _Requirements: 2.7, 11.1, 11.4, 14.5_

- [ ] 23. Implement output file versioning and reproducibility
  - [ ] 23.1 Create src/utils/versioning.py with generate_output_filename()
    - Accept base_name (e.g., "agentic_outcomes"), extension (e.g., ".csv")
    - Compute config_hash from SystemConfig (hash of theta_init, pricing_bounds, baseline_tariff, convergence thresholds)
    - Generate timestamp in ISO 8601 format
    - Return filename: f"{base_name}_{timestamp}_{config_hash[:8]}.{extension}"
    - _Requirements: 11.5_
  
  - [ ]* 23.2 Write property test for output file versioning
    - **Property 34: Output File Versioning**
    - **Validates: Requirements 11.5**
  
  - [ ] 23.3 Implement save_with_versioning() wrapper
    - Wrap pandas.to_csv() and json.dump() to use versioned filenames
    - Log saved file path with full version info
    - _Requirements: 11.5_
  
  - [ ] 23.4 Write unit tests for versioning
    - Test filename generation includes timestamp and config hash
    - Test different configs produce different hashes (theta_init change → different hash)
    - Test same config produces same hash (reproducibility check)
    - _Requirements: 11.5_

- [ ] 24. Final integration and end-to-end testing
  - [ ] 24.1 Create integration test with synthetic dataset
    - Generate synthetic unified_analytical_base.csv (100 rows, known patterns: high utilization at peak hours, low at off-peak)
    - Run full pipeline: load data → train Demand_Agent → run optimization loop (10 steps) → generate EDA → evaluate
    - Verify outputs exist: agentic_outcomes.csv, predictions.csv, evaluation_summary.json, EDA plots (at least 10 PNG files)
    - Verify convergence_checker is invoked and logged
    - Verify LLM cost tracking logs cumulative tokens
    - Verify causal disclaimer appears in evaluation_summary.json
    - _Requirements: All_
  
  - [ ] 24.2 Test with actual ACN-Data and UrbanEV datasets
    - Place raw JSON (ACN-Data) and CSV (UrbanEV) in data/raw/
    - Run preprocessing: python -m src.preprocessing.acn_parser, python -m src.preprocessing.urbanev_parser, python -m src.preprocessing.dataset_fusion
    - Verify unified_analytical_base.csv is created with correct schema
    - Run EDA: python -m src.cli --mode eda --config configs/indian_market.json
    - Verify EDA outputs include weekday/weekend comparison and volatility analysis
    - _Requirements: 7.1, 7.8, 17.1_
  
  - [ ] 24.3 Test convergence-based termination
    - Configure tight convergence thresholds (revenue_variance=0.5%, parameter_delta=0.005) on synthetic stable dataset
    - Run optimization and verify it terminates before max_iterations due to convergence
    - Verify convergence_met=True in evaluation_summary.json
    - _Requirements: 1.1, 1.6_
  
  - [ ] 24.4 Test max_iterations safety
    - Configure impossible convergence thresholds on volatile synthetic dataset
    - Run optimization and verify it terminates at exactly max_iterations=1000
    - Verify MaxIterationsExceeded warning in logs
    - Verify convergence_met=False in evaluation_summary.json
    - _Requirements: 1.6, 15.2_
  
  - [ ] 24.5 Test LLM cost budget enforcement
    - Configure small llm_token_budget (e.g., 1000 tokens) and small llm_cost_budget_usd (e.g., $0.10)
    - Run optimization and verify it raises QuotaExceededError before completing test dataset
    - Verify cumulative_tokens_used is logged
    - _Requirements: Cost budgeting gap from issues.txt_
  
  - [ ] 24.6 Test agent failure graceful degradation
    - Mock LLM provider to return invalid JSON for 5 consecutive steps
    - Run optimization and verify: (1) fallback is invoked, (2) optimization continues, (3) all 5 steps complete with fallback_used=True
    - Verify no crash or termination due to agent failures
    - _Requirements: 15.1, 15.6_
  
  - [ ] 24.7 Test presentation asset generation
    - Run: python -m src.cli --mode presentation --config configs/indian_market.json
    - Verify outputs/presentation/ contains assets for all 6 slides
    - Verify README.md exists and includes causal disclaimer
    - _Requirements: Presentation gap_
  
  - [ ] 24.8 Test reproducibility with fixed seed
    - Run optimization twice with same config and same --seed 42
    - Verify Demand_Agent predictions are identical (RMSE difference = 0)
    - Note: LLM outputs will differ (non-deterministic even at low temperature) - this is expected and logged
    - _Requirements: 11.1, 11.2, 11.3_

- [ ] 25. Documentation and final polish
  - [ ] 25.1 Create comprehensive README.md
    - Installation instructions (requirements.txt, Python 3.9+)
    - Quick start guide (preprocessing → training → optimization → evaluation → presentation)
    - Configuration examples (Indian market, US market)
    - CLI usage examples for all modes
    - Troubleshooting section (API key setup, convergence tuning, LLM cost management)
    - Reproducibility notes (seed behavior, LLM non-determinism disclaimer)
    - Causal inference disclaimer
    - _Requirements: All_
  
  - [ ] 25.2 Create configs/indian_market.json example
    - Baseline: ₹15/kWh, bounds: [10, 22]
    - OpenAI GPT-4o as LLM provider
    - Default convergence thresholds, reward weights, learning rate schedule
    - llm_cost_budget_usd: 10.0, max_llm_calls_per_step: 5, llm_token_budget: 100000
    - _Requirements: 12.3, 12.4_
  
  - [ ] 25.3 Create configs/us_market.json example
    - Baseline: $0.30/kWh, bounds: [0.20, 0.45]
    - Anthropic Claude as LLM provider
    - Same convergence thresholds and learning rate as Indian market
    - _Requirements: 12.3, 12.4_
  
  - [ ] 25.4 Add inline code documentation
    - Docstrings for all classes and public methods (Google style)
    - Inline comments for complex logic (especially parameter update arbitration, convergence checks, fallback formulas)
    - Type hints for all function signatures
    - _Requirements: Best practices_
  
  - [ ] 25.5 Run final test suite and coverage
    - Execute: pytest tests/ -v --cov=src --cov-report=html --hypothesis-profile=ci
    - Verify line coverage > 85%
    - Verify all 44 property tests pass
    - Verify all unit tests pass
    - Generate coverage report in outputs/coverage/
    - _Requirements: All_

- [ ] 26. Final checkpoint - Full system validation
  - Ensure all tests pass, ask the user if questions arise.
  - Verify all critical gaps from issues.txt are addressed in implementation.

## Notes

- Tasks marked with `*` are optional property-based tests and can be skipped for faster MVP
- Each task references specific requirements for traceability (e.g., _Requirements: 7.1, 7.8_)
- Checkpoints at tasks 5, 8, 11, 16, 19, 26 ensure incremental validation
- All property tests validate universal correctness properties from the design document
- Unit tests validate specific examples, edge cases, and integration points
- The system includes all 16 critical gaps identified in issues.txt:
  1. ✅ JSON → CSV conversion for ACN-Data (Task 2)
  2. ✅ Dataset alignment/fusion (Task 4)
  3. ✅ Occupancy Density and Energy Cost per kWh features (Tasks 2, 7)
  4. ✅ Spatial features with station clustering (Tasks 3, 4, 7)
  5. ✅ Congestion probability third output (Task 9)
  6. ✅ Model selection/comparison (Task 9.2)
  7. ✅ LangGraph prompt templates (Task 12.2, 12.3)
  8. ✅ Monitoring_Agent ground-truth feedback loop (Task 13.2)
  9. ✅ Weekday vs. weekend analysis (Task 18.3)
  10. ✅ Volatility analysis with shoulder period (Task 18.4)
  11. ✅ Pricing efficiency trend (Task 20.9)
  12. ✅ Presentation asset generation (Task 21)
  13. ✅ LLM cost budgeting (Tasks 6.1, 10.1, 17.2)
  14. ✅ Causal inference safeguards (Task 14.2)
  15. ✅ Hypothesis profiles for property tests (Task 1)
  16. ✅ Max iteration safety already enforced (Task 17.2)

