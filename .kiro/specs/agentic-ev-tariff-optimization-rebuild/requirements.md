# Requirements Document

## Introduction

This document specifies requirements for rebuilding the EV Charging Dynamic Tariff Optimization System with proper agentic architecture. The current system (v2.0.0) suffers from critical architectural failures including hardcoded iteration limits, >90% agent failure rates requiring fallbacks, and no genuine self-improving behavior. The rebuild will deliver a robust three-agent system that autonomously optimizes EV charging tariffs through continuous learning from real-world outcomes.

## Glossary

- **System**: The complete EV Charging Dynamic Tariff Optimization System
- **Demand_Agent**: The machine learning component that predicts charging station utilization and queue length
- **Pricing_Agent**: The agentic component that determines optimal per-kWh tariffs based on demand forecasts
- **Monitoring_Agent**: The agentic component that evaluates pricing outcomes and updates system parameters
- **Convergence**: A stable state where system metrics (revenue, utilization, parameters) stop changing significantly
- **LLM_Provider**: The large language model API service (OpenAI, Anthropic, or local model)
- **Agent_Success_Rate**: Percentage of pricing/monitoring decisions where LLM agents produce valid, business-logic-compliant outputs
- **Valid_Decision**: An agent output that passes format validation AND business logic validation (price within bounds, regime matches demand conditions, parameter adjustments directionally correct)
- **Unified_Base**: The merged dataset from ACN-Data and UrbanEV containing hourly charging session features
- **ACN_Data**: Caltech/JPL Adaptive Charging Network dataset with session-level metrics (US-based charging stations)
- **UrbanEV_Data**: Shenzhen urban EV charging dataset with utilization and queue metrics (China-based charging stations)
- **Indian_Market_Context**: Configuration targeting Indian EV charging market with baseline ₹15/kWh and bounds [₹10, ₹22] per kWh (configurable for other geographies)
- **Theta**: The parameter vector [epsilon, alpha, beta] governing pricing sensitivity
- **Episode**: One complete pass through the test dataset during optimization
- **demand_shift**: The percentage change in demand due to price elasticity, calculated as -epsilon * (price_new - price_baseline) / price_baseline, where epsilon is the elasticity parameter from theta
- **reward**: The weighted evaluation metric combining revenue gain, utilization improvement, and queue reduction, calculated as w1 * revenue_gain_pct + w2 * utilization_improvement - w3 * queue_penalty, with configurable weights

## Requirements

### Requirement 1: Multi-Objective Convergence Criteria

**User Story:** As a system operator, I want the optimization loop to run until convergence is achieved across revenue, utilization distribution, and queue management, so that the system finds stable optimal parameters that balance all business objectives.

#### Acceptance Criteria

1. THE System SHALL terminate optimization WHEN all three convergence conditions are met simultaneously for 50 consecutive steps
2. THE System SHALL track revenue stability: variance of revenue_gain_pct over 50 steps less than 1.0%
3. THE System SHALL track parameter stability: all theta parameter changes over 50 steps less than 0.01
4. THE System SHALL track utilization health: standard deviation of hourly utilization over 50 steps less than 0.15 AND maximum utilization less than 0.80
5. THE System SHALL track queue reduction: mean queue length over 50 steps at least 20% below baseline mean queue
6. THE System SHALL enforce a maximum iteration limit of 1000 steps to prevent infinite loops
7. THE System SHALL log all convergence metrics per step and the criterion that triggered termination

### Requirement 2: Reliable LLM Infrastructure

**User Story:** As a developer, I want agents to use a reliable LLM provider with sufficient context windows, so that agent failures are minimized and system behavior is predictable.

#### Acceptance Criteria

1. THE System SHALL support OpenAI GPT-4 as the primary LLM provider
2. THE System SHALL support Anthropic Claude as an alternative LLM provider
3. THE System SHALL support local models (Ollama) as a third option for LLM provider
4. WHERE OpenAI is selected, THE System SHALL use gpt-4o with 128K context window
5. WHERE Anthropic is selected, THE System SHALL use claude-3-5-sonnet-20241022 with 200K context window
6. THE System SHALL allow LLM provider selection through configuration without code changes
7. THE System SHALL validate API credentials at startup before beginning optimization

### Requirement 3: High Agent Decision Quality

**User Story:** As a system architect, I want agentic decisions to be both valid and business-logic-compliant >80% of the time, so that the system exhibits genuine autonomous reasoning.

#### Acceptance Criteria

1. THE System SHALL validate agent decisions with business logic checks before accepting them
2. THE Pricing_Agent output SHALL be a structured JSON object with fields: recommended_price (float), regime (enum: "surge" | "neutral" | "discount"), reasoning (string)
3. THE Pricing_Agent decision SHALL pass validation WHEN price is within configured pricing_bounds AND regime matches demand conditions (surge when utilization >0.80, neutral when utilization in [0.30, 0.80], discount when utilization <0.30)
4. THE Monitoring_Agent decision SHALL pass validation WHEN parameter adjustments are directionally correct (increase alpha when high utilization persists, increase beta when low utilization persists, adjust epsilon based on revenue direction)
5. THE System SHALL achieve >80% validation pass rate across a full optimization episode for both agents
6. WHEN an agent invocation fails format validation, THE System SHALL retry up to 3 times with exponential backoff
7. WHEN all retries fail OR business logic validation fails, THE System SHALL log detailed error information and use deterministic fallback
8. THE System SHALL export separate metrics for format validation success and business logic validation success in the final evaluation report

### Requirement 4: Parameter Update Arbitration

**User Story:** As a pricing strategist, I want clear precedence rules when both agent and fallback propose parameter updates, so that the system behavior is predictable and debuggable.

#### Acceptance Criteria

1. THE System SHALL apply parameter updates according to this precedence: (1) Monitoring_Agent proposal if validation passes, (2) deterministic fallback computation, (3) no change
2. WHEN Monitoring_Agent proposes epsilon adjustment AND passes business logic validation, THE System SHALL apply the agent proposal with learning rate decay
3. WHEN Monitoring_Agent proposal fails validation, THE System SHALL compute fallback adjustment using deterministic rules and log the conflict
4. THE deterministic fallback SHALL decrease epsilon by 0.02 WHEN revenue gain is negative for 3 consecutive steps
5. THE deterministic fallback SHALL increase alpha by 0.10 WHEN utilization remains above 0.80 during surge pricing for 3 consecutive steps
6. THE deterministic fallback SHALL increase beta by 0.10 WHEN utilization remains below 0.30 during discount pricing for 3 consecutive steps
7. THE System SHALL log all parameter updates with source (agent or fallback), old values, proposed delta, and final applied delta
8. THE System SHALL apply learning rate decay to all updates: eta = eta_0 / (1 + decay * step)

### Requirement 5: Simplified Agent Prompts

**User Story:** As an AI engineer, I want agent prompts to focus on strategic reasoning rather than formula execution, so that agents leverage their language understanding capabilities effectively.

#### Acceptance Criteria

1. THE Pricing_Agent prompt SHALL request economic reasoning about demand conditions without specifying formulas
2. THE Pricing_Agent prompt SHALL provide context about current utilization, queue length, time-of-day, and elasticity parameters
3. THE Monitoring_Agent prompt SHALL request evaluation of pricing effectiveness without specifying calculation methods
4. THE Monitoring_Agent prompt SHALL provide context about revenue outcomes, demand shifts, and utilization changes
5. THE System SHALL compute all quantitative metrics (revenue, utilization, elasticity) in deterministic Python code, not through LLM generation

### Requirement 6: Comprehensive Error Handling

**User Story:** As a system reliability engineer, I want comprehensive error handling without silent fallbacks, so that I can diagnose and fix agent failures.

#### Acceptance Criteria

1. WHEN an LLM API call fails, THE System SHALL log the HTTP status code, error message, and request context
2. WHEN JSON parsing fails, THE System SHALL log the raw LLM response for debugging
3. WHEN schema validation fails, THE System SHALL log the validation errors and the attempted data structure
4. THE System SHALL distinguish between transient failures (rate limits, timeouts) and permanent failures (invalid API key)
5. IF transient failure occurs, THEN THE System SHALL retry with exponential backoff up to 3 attempts

### Requirement 7: Dataset Integration and Feature Engineering

**User Story:** As a data engineer, I want to use existing merged dataset features directly and engineer only necessary derived features, so that feature computation is grounded in actual data schema.

#### Acceptance Criteria

1. THE System SHALL load unified_analytical_base.csv with columns: hourly_timestamp, acn_sessions_count, acn_total_kwh, acn_base_revenue, acn_avg_kwh_per_session, acn_revenue_per_session, acn_energy_cost_per_kwh, time_step, urban_mean_utilization, urban_peak_queue, urban_total_volume, hour_of_day, day_of_week, is_weekend
2. THE System SHALL use urban_mean_utilization directly as the utilization feature without transformation (already in [0,1] range)
3. THE System SHALL use urban_peak_queue directly as the queue feature without transformation
4. THE System SHALL engineer is_peak_hour as boolean (hour_of_day in [7,8,9,17,18,19])
5. THE System SHALL engineer revenue_per_kwh as acn_base_revenue / acn_total_kwh (when acn_total_kwh > 0)
6. THE System SHALL handle missing values: forward-fill for utilization and queue, drop rows with missing revenue data
7. THE System SHALL enforce chronological 80/20 train/test split based on time_step ordering
8. THE System SHALL document dataset merge provenance: ACN-Data (Caltech/JPL) provides session-level metrics, UrbanEV (Shenzhen) provides utilization and queue metrics, merged on normalized hourly timestamps
9. THE System SHALL treat merged data as a single unified network for training purposes (timezone normalization already applied in preprocessing)

### Requirement 8: XGBoost Demand Prediction with Multi-Output Training

**User Story:** As a forecasting analyst, I want reliable demand predictions using proven gradient boosting methods, so that pricing decisions are based on accurate forecasts.

#### Acceptance Criteria

1. THE Demand_Agent SHALL use XGBoost MultiOutputRegressor to jointly predict urban_mean_utilization and urban_peak_queue
2. THE Demand_Agent SHALL train on chronological first 80% of unified_analytical_base (by time_step ordering) with no shuffling
3. THE Demand_Agent SHALL evaluate on held-out chronological last 20% test set
4. THE Demand_Agent SHALL use input features: acn_sessions_count, acn_total_kwh, acn_avg_kwh_per_session, hour_of_day, day_of_week, is_weekend, is_peak_hour
5. THE Demand_Agent SHALL report RMSE, MAE, and R² for both prediction targets
6. THE Demand_Agent SHALL clip predicted urban_mean_utilization to [0, 1] and predicted urban_peak_queue to [0, infinity]
7. THE Demand_Agent SHALL use feature importance analysis to identify top 5 predictive features and log them

### Requirement 9: Real-Time Optimization Metrics with Reward Function

**User Story:** As a business analyst, I want to track revenue, utilization, and customer response in real-time with a unified reward signal, so that I can evaluate system performance continuously.

#### Acceptance Criteria

1. THE System SHALL compute revenue_gain_pct as (revenue_new - revenue_baseline) / revenue_baseline * 100
2. THE System SHALL compute demand_shift as -epsilon * (price_new - price_baseline) / price_baseline (where epsilon is from theta vector)
3. THE System SHALL compute charger_utilization after applying demand_shift elasticity adjustment
4. THE System SHALL compute pricing_efficiency as revenue_new / kwh_delivered
5. THE System SHALL compute customer_response_rate as demand_shift * 100 (percentage demand change)
6. THE System SHALL compute avg_wait_reduction as -demand_shift * queue_actual
7. THE System SHALL compute reward as w1 * revenue_gain_pct + w2 * utilization_improvement - w3 * queue_penalty, where utilization_improvement = (utilization_new - utilization_baseline) * 100 and queue_penalty = max(0, queue_new - queue_baseline) * 100
8. THE System SHALL accept reward_weights [w1, w2, w3] as configuration parameters (default: [1.0, 0.5, 0.3])
9. THE System SHALL export all metrics per-step to agentic_outcomes.csv

### Requirement 10: Evaluation Against Configurable Baseline

**User Story:** As a product manager, I want to compare optimized performance against a configurable baseline tariff, so that I can quantify business value across different geographies.

#### Acceptance Criteria

1. THE System SHALL accept baseline_tariff_per_kwh as a configuration parameter (default: 15.0 for Indian context, derived from acn_energy_cost_per_kwh in dataset)
2. THE System SHALL accept pricing_bounds as configuration parameters [lower_bound, upper_bound] (default: [10.0, 22.0] representing ₹10-22/kWh for Indian EV market context)
3. THE System SHALL compute revenue uplift as mean(revenue_gain_pct) across all optimization steps relative to the configured baseline
4. THE System SHALL compute off-peak uplift as percentage increase in utilization during discount regime hours (hour_of_day not in [7,8,9,17,18,19])
5. THE System SHALL compute congestion reduction as mean(queue_length) compared to baseline mean queue
6. THE System SHALL report final agent decision quality metrics (format validation pass rate, business logic validation pass rate) compared to 80% target threshold
7. THE System SHALL generate a summary report with all evaluation metrics including multi-objective performance (revenue, utilization distribution, queue reduction)

### Requirement 11: Reproducible Results

**User Story:** As a researcher, I want reproducible optimization runs with configurable random seeds, so that I can validate experimental results.

#### Acceptance Criteria

1. THE System SHALL accept random_seed parameter for XGBoost training
2. THE System SHALL accept random_seed parameter for train/test split operations
3. THE System SHALL set random_seed for numpy random number generation
4. THE System SHALL log all hyperparameters (learning rate, decay, initial theta) at startup
5. THE System SHALL version all output files with timestamp and configuration hash

### Requirement 12: Geography-Aware Configuration Management

**User Story:** As a system administrator, I want centralized configuration for all system parameters including geography-specific pricing bounds, so that I can deploy the same system across different markets without code changes.

#### Acceptance Criteria

1. THE System SHALL load LLM provider selection from environment variable or config file
2. THE System SHALL load initial theta values [epsilon, alpha, beta] from configuration (default: [1.5, 2.5, 2.5])
3. THE System SHALL load baseline_tariff_per_kwh from configuration (default: 15.0 for Indian market)
4. THE System SHALL load pricing_bounds [lower, upper] from configuration (default: [10.0, 22.0] for Indian market, derivable from baseline ± elasticity margin)
5. THE System SHALL load convergence thresholds (revenue variance, parameter delta, utilization std, queue reduction) from configuration
6. THE System SHALL load learning rate schedule (eta_0, decay) from configuration
7. THE System SHALL validate all configuration values at startup: epsilon in [0.1, 5.0], alpha in [1.0, 10.0], beta in [1.0, 10.0], pricing_bounds[0] > 0, pricing_bounds[1] > pricing_bounds[0]
8. THE System SHALL reject invalid configurations with clear error messages identifying the constraint violation

### Requirement 13: Configuration Parser with Round-Trip Guarantee

**User Story:** As a developer, I want to parse and serialize configuration files reliably, so that configuration management is robust and maintainable.

#### Acceptance Criteria

1. WHEN a valid JSON configuration file is provided, THE Configuration_Parser SHALL parse it into a SystemConfig object
2. WHEN an invalid JSON configuration file is provided, THE Configuration_Parser SHALL return a descriptive error identifying the validation failure
3. THE Configuration_Pretty_Printer SHALL format SystemConfig objects back into valid JSON configuration files with 2-space indentation
4. FOR ALL valid SystemConfig objects, parsing then printing then parsing SHALL produce an equivalent object (round-trip property)
5. THE Configuration_Parser SHALL validate that epsilon is in range [0.1, 5.0], alpha in [1.0, 10.0], and beta in [1.0, 10.0]
6. THE Configuration_Parser SHALL validate that pricing_bounds is a 2-element array with pricing_bounds[0] > 0 AND pricing_bounds[1] > pricing_bounds[0]
7. THE Configuration_Parser SHALL validate that baseline_tariff_per_kwh is within pricing_bounds range

### Requirement 14: Comprehensive Logging

**User Story:** As a DevOps engineer, I want detailed structured logging of all system events, so that I can monitor, debug, and audit system behavior.

#### Acceptance Criteria

1. THE System SHALL log each optimization step with: step number, timestamp, regime, price, revenue_gain_pct, reward, demand_shift
2. THE System SHALL log all LLM API calls with: provider, model, tokens used, latency, success/failure status
3. THE System SHALL log parameter updates with: old theta, delta, new theta, learning rate used
4. THE System SHALL log convergence checks with: metric name, current value, threshold, pass/fail status
5. THE System SHALL support log level configuration (DEBUG, INFO, WARNING, ERROR) via command-line argument

### Requirement 15: Graceful Degradation with Parameter Checkpointing

**User Story:** As a reliability engineer, I want the system to degrade gracefully under failure conditions, so that partial results are preserved and system state remains valid.

#### Acceptance Criteria

1. WHEN an agent failure occurs, THE System SHALL complete the current step using deterministic fallback and continue optimization
2. WHEN convergence cannot be achieved within maximum iterations, THE System SHALL export partial results with a warning flag
3. WHEN LLM API quota is exceeded, THE System SHALL pause optimization and wait for quota reset before continuing
4. THE System SHALL checkpoint theta parameters every 100 steps to allow recovery from crashes
5. THE System SHALL validate all pricing decisions against configured pricing_bounds before application
6. THE System SHALL log all graceful degradation events with context (failure type, fallback used, state preserved)

### Requirement 16: Performance Benchmarking

**User Story:** As a performance analyst, I want to benchmark the optimized system against multiple baselines, so that I can quantify improvement rigorously.

#### Acceptance Criteria

1. THE System SHALL compare performance against configured baseline_tariff_per_kwh (no dynamic pricing)
2. THE System SHALL compare performance against time-of-day pricing baseline (peak/off-peak only)
3. THE System SHALL compare performance against deterministic formula pricing (no agent reasoning)
4. THE System SHALL compute statistical significance of revenue improvements using paired t-test
5. THE System SHALL export comparison results to benchmark_comparison.csv with mean, std, min, max for each baseline

### Requirement 17: Exploratory Data Analysis Deliverables

**User Story:** As a data scientist, I want comprehensive EDA outputs that characterize demand patterns and system behavior, so that I can validate model assumptions and identify optimization opportunities.

#### Acceptance Criteria

1. THE System SHALL generate temporal demand pattern analysis showing hourly and daily utilization trends
2. THE System SHALL generate utilization distribution histogram with quartile statistics
3. THE System SHALL generate peak-hour analysis identifying top 5 high-demand time windows
4. THE System SHALL generate correlation analysis between revenue, utilization, and queue length
5. THE System SHALL generate feature importance ranking from Demand_Agent training
6. THE System SHALL export all EDA visualizations to outputs/eda/ directory as PNG files with 300 DPI resolution
7. THE System SHALL export EDA summary statistics to outputs/eda/eda_summary.json with descriptive statistics for all key metrics

