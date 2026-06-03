# Requirements Document

## Introduction

This feature covers the improvement, restructuring, and extension of the existing OP'26 Agentic AI-Based Dynamic Tariff Optimisation system for EV Charging Networks. The project uses real-world session data from ACN-Data (Caltech/JPL, 30,000+ sessions) and the UrbanEV ST-EVCDP dataset (24,798 charging piles) to build a self-improving three-agent pricing engine. The goal is to produce clean, modular, well-structured code that fully meets the OP'26 evaluation criteria, with a properly organised repository, additional analytical features, and all known issues resolved.

## Glossary

- **System**: The complete OP'26 EV charging analytics and tariff optimisation codebase
- **Pipeline**: The data preprocessing module that ingests raw ACN and UrbanEV data and produces the unified analytical base
- **EDA_Module**: The exploratory data analysis module that generates insight-driven visualisations
- **DemandPredictionAgent**: The ML agent that forecasts charger utilisation rate and queue length proxy using XGBoost
- **TariffPricingAgent**: The Gemini-powered agent that translates demand forecasts into optimal per-kWh tariffs
- **MonitoringLearningAgent**: The Gemini-powered agent that evaluates pricing decisions against outcomes and updates the parameter vector θ
- **Orchestrator**: The module that drives the closed three-agent feedback loop over the test set
- **Unified_Base**: The `unified_analytical_base.csv` file produced by the Pipeline, containing aligned hourly ACN and UrbanEV features
- **ACN_Data**: The Adaptive Charging Network dataset from Caltech/JPL (acndata_sessions.json.xlsx)
- **UrbanEV_Data**: The ST-EVCDP spatial-temporal dataset (volume.csv, occupancy.csv, duration.csv, price.csv, stations.csv)
- **Theta**: The mutable parameter vector [ε, α, β] representing price elasticity, surge sensitivity, and discount sensitivity
- **Surge_Regime**: Pricing state triggered when predicted utilisation exceeds 0.80
- **Discount_Regime**: Pricing state triggered when predicted utilisation falls below 0.30
- **Neutral_Regime**: Pricing state when utilisation is between 0.30 and 0.80
- **Off_Peak_Uplift**: The increase in charging sessions during low-demand periods after discount pricing is applied
- **Pricing_Efficiency_Score**: Revenue per kWh delivered, tracked over time to measure feedback loop improvement
- **Repository**: The full project directory structure including source code, data, outputs, and documentation

---

## Requirements

### Requirement 1: Repository Structure and Modularity

**User Story:** As a developer and evaluator, I want the project to have a clean, modular repository structure, so that code is easy to navigate, reproduce, and extend.

#### Acceptance Criteria

1. THE System SHALL organise source code into a `src/` directory with clearly separated modules: `pipeline/`, `eda/`, `agents/`, and `utils/`.
2. THE System SHALL separate raw data files, generated outputs, and source code into distinct top-level directories (`data/`, `outputs/`, `src/`).
3. THE System SHALL provide a `requirements.txt` file listing all Python dependencies with pinned versions.
4. THE System SHALL provide a top-level `README.md` with setup instructions, directory layout, and step-by-step run commands.
5. WHEN a module is imported, THE System SHALL not execute side effects such as training models or reading files at import time.
6. THE System SHALL expose each pipeline stage as a callable function or class, not only as a `__main__` block.
7. THE System SHALL remove duplicate files (the `preprocessed_data/p1_preprocess_pipeline.py` copy) so each module exists in exactly one canonical location.

---

### Requirement 2: Data Preprocessing Pipeline

**User Story:** As a data engineer, I want a robust, well-documented preprocessing pipeline, so that the unified analytical base is reproducible and free of data quality issues.

#### Acceptance Criteria

1. WHEN the Pipeline is executed, THE Pipeline SHALL load ACN_Data from `data/raw/acndata_sessions.json.xlsx` and UrbanEV_Data from `data/raw/*.csv` using configurable path arguments.
2. WHEN ACN_Data contains rows where `kWhDelivered` is zero or null, THE Pipeline SHALL log the count of affected rows and exclude them from aggregation rather than silently propagating zero values.
3. WHEN UrbanEV_Data wide matrices are melted into long format, THE Pipeline SHALL validate that the resulting merge on `[time_step, station_node]` produces no unexpected nulls before aggregation.
4. THE Pipeline SHALL engineer and document the following features: Charger Utilisation Rate (`occupancy_density × 1.2`, clipped to [0,1]), Queue Length Proxy, Baseline Revenue per Session (kWh × ₹15), hour_of_day, day_of_week, and is_weekend.
5. WHEN the Pipeline aligns ACN hourly aggregates with UrbanEV hourly aggregates, THE Pipeline SHALL log the number of rows from each source and the final row count of the Unified_Base.
6. THE Pipeline SHALL export the Unified_Base to `data/processed/unified_analytical_base.csv` with a documented schema comment at the top of the output file.
7. IF any required input file is missing, THEN THE Pipeline SHALL raise a `FileNotFoundError` with the missing file path and a human-readable message indicating which step requires it.

---

### Requirement 3: Exploratory Data Analysis

**User Story:** As an analyst, I want comprehensive, insight-driven EDA visualisations, so that demand behaviour and pricing implications are clearly communicated.

#### Acceptance Criteria

1. WHEN the EDA_Module is executed, THE EDA_Module SHALL generate and save all visualisation plots to `outputs/eda/`.
2. THE EDA_Module SHALL produce the following plots: long-run demand trend with 7-day rolling mean, intraday demand cycle, weekday vs weekend profiles with DoW×Hour heatmap, ACN session distributions (kWh, duration, station usage, hourly counts), peak/shoulder/off-peak volatility box plots, feature correlation heatmap, revenue analysis (tariff distribution, comparison, regime pie, hourly gain), session efficiency scatter, tariff narrative with annotated pricing zones, and XGBoost feature importance.
3. WHEN the EDA_Module generates the revenue analysis plot, THE EDA_Module SHALL compute and display the Revenue Gain % using the formula `((Dynamic Revenue − Baseline Revenue) / Baseline Revenue) × 100` against the ₹15/kWh fixed baseline.
4. THE EDA_Module SHALL accept `--base` and `--acn` CLI arguments for input file paths, defaulting to the canonical processed data paths.
5. WHEN the EDA_Module cannot load the XGBoost library, THE EDA_Module SHALL skip the feature importance plot and log a warning rather than raising an exception.
6. THE EDA_Module SHALL produce an additional plot showing predicted vs actual utilisation after the Orchestrator has been run, reading from `outputs/predictions.csv`.
7. THE EDA_Module SHALL produce an additional plot showing reward convergence and θ (ε, α, β) evolution over episodes, reading from `outputs/agentic_outcomes.csv`.

---

### Requirement 4: Demand Prediction Agent

**User Story:** As an ML engineer, I want a well-evaluated demand prediction model, so that utilisation forecasts are accurate and the evaluation metrics meet OP'26 specification.

#### Acceptance Criteria

1. WHEN the DemandPredictionAgent is initialised, THE DemandPredictionAgent SHALL train a MultiOutputRegressor with XGBoost on a strict chronological 80/20 train-test split with no data shuffling.
2. THE DemandPredictionAgent SHALL apply causal, leakage-free feature engineering: all lag and rolling features SHALL use `.shift(1)` as the anchor, and `dropna()` SHALL be called only after all features are computed.
3. THE DemandPredictionAgent SHALL expose an `evaluation_metrics()` method returning RMSE, MAE, and R² for both `urban_mean_utilization` and `urban_peak_queue` targets.
4. WHEN `predict_state(row_idx)` is called, THE DemandPredictionAgent SHALL return a validated `ForecastState` Pydantic object with `u_pred` clipped to [0, 1] and `kwh_delivered` floored at 0.01.
5. THE DemandPredictionAgent SHALL support an optional LightGBM backend selectable via a constructor argument, and SHALL report RMSE, MAE, and R² for both backends side-by-side when both are trained.
6. WHEN the DemandPredictionAgent is trained, THE DemandPredictionAgent SHALL log the train row count, test row count, and feature list used.

---

### Requirement 5: Tariff Pricing Agent

**User Story:** As a pricing strategist, I want a Gemini-powered tariff agent that produces economically sound pricing decisions, so that revenue is maximised while congestion is reduced.

#### Acceptance Criteria

1. WHEN `compute_tariff(state)` is called with a `ForecastState` where `u_pred > 0.80`, THE TariffPricingAgent SHALL return a `PricingDecision` in the `surge` regime with `p_new` in the range (₹15, ₹22].
2. WHEN `compute_tariff(state)` is called with a `ForecastState` where `u_pred < 0.30`, THE TariffPricingAgent SHALL return a `PricingDecision` in the `discount` regime with `p_new` in the range [₹10, ₹15).
3. WHEN `compute_tariff(state)` is called with a `ForecastState` where `0.30 ≤ u_pred ≤ 0.80`, THE TariffPricingAgent SHALL return a `PricingDecision` in the `neutral` regime with `p_new` near ₹15.
4. IF the Gemini API call fails after 3 retry attempts, THEN THE TariffPricingAgent SHALL return a deterministic sigmoid-based fallback `PricingDecision` and log an error.
5. WHEN `apply_update(delta)` is called, THE TariffPricingAgent SHALL update Theta by adding the delta vector and clip each parameter to its defined bounds: ε ∈ [0.1, 5.0], α ∈ [1.0, 10.0], β ∈ [1.0, 10.0].
6. THE TariffPricingAgent SHALL never return a `p_new` outside the range [₹10, ₹22] regardless of Gemini output.

---

### Requirement 6: Monitoring and Learning Agent

**User Story:** As a system operator, I want the monitoring agent to evaluate pricing decisions and improve the system over time, so that the feedback loop demonstrably converges.

#### Acceptance Criteria

1. WHEN `step(state, decision)` is called, THE MonitoringLearningAgent SHALL compute all OP'26 metrics: `demand_shift`, `revenue_new`, `revenue_gain_pct`, `charger_utilisation`, `avg_wait_reduction`, `pricing_efficiency`, and `reward`.
2. THE MonitoringLearningAgent SHALL use the demand-adjusted revenue formula: `revenue_new = p_new × kwh × max(0.05, 1 + demand_shift)`, ensuring consistency between the episode log and the `LearningUpdate` schema.
3. WHEN the MonitoringLearningAgent applies a parameter update, THE MonitoringLearningAgent SHALL scale the Gemini-proposed Δθ by the current learning rate `η = η₀ / (1 + decay × t)` before calling `apply_update()`.
4. IF the Gemini API call fails after 3 retry attempts, THEN THE MonitoringLearningAgent SHALL use the deterministic economic fallback and log an error.
5. THE MonitoringLearningAgent SHALL expose a `summary()` method returning a DataFrame with one row per episode step containing all logged metrics.
6. WHEN the full episode run completes, THE MonitoringLearningAgent SHALL compute and report the Off_Peak_Uplift metric: the percentage change in mean session count during discount-regime hours compared to pre-optimisation baseline.

---

### Requirement 7: Orchestrator and Agentic Loop

**User Story:** As a researcher, I want the orchestrator to run the full three-agent feedback loop and produce all required output files, so that the OP'26 evaluation criteria can be verified.

#### Acceptance Criteria

1. WHEN the Orchestrator is run, THE Orchestrator SHALL execute the sequential three-agent loop: DemandPredictionAgent → TariffPricingAgent → MonitoringLearningAgent for each episode step.
2. THE Orchestrator SHALL accept CLI arguments for: `--csv`, `--steps`, `--verbose`, `--lr`, `--decay`, `--delay`, `--epsilon`, `--alpha`, `--beta`, `--out`, `--predictions`, and `--log-level`.
3. WHEN the episode loop completes, THE Orchestrator SHALL print a final evaluation report covering all OP'26 metrics: RMSE/MAE/R² for demand prediction, Revenue Gain %, Charger Utilisation Rate, Off_Peak_Uplift, Avg Wait Reduction, and Pricing_Efficiency_Score.
4. THE Orchestrator SHALL export `agentic_outcomes.csv` to `outputs/` with one row per episode step and all metric columns defined in the `LearningUpdate` schema plus θ state columns.
5. THE Orchestrator SHALL export `predictions.csv` to `outputs/` with actual vs predicted values for both demand targets.
6. WHEN the Orchestrator is initialised, THE Orchestrator SHALL validate that the input CSV exists and that the `GEMINI_API_KEY` environment variable is set, raising descriptive errors if either is missing.
7. THE Orchestrator SHALL include a rate-limit guard between consecutive Gemini API calls, configurable via `--delay`.

---

### Requirement 8: Post-Run Analysis and Additional Metrics

**User Story:** As an evaluator, I want additional analysis plots and metrics generated after the agentic run, so that the system's learning behaviour and pricing outcomes are fully documented.

#### Acceptance Criteria

1. WHEN `outputs/predictions.csv` exists, THE System SHALL generate a predicted vs actual utilisation plot with the train/test split boundary clearly marked.
2. WHEN `outputs/agentic_outcomes.csv` exists, THE System SHALL generate a reward convergence plot showing per-step reward and a rolling mean over 50 steps.
3. WHEN `outputs/agentic_outcomes.csv` exists, THE System SHALL generate a θ evolution plot showing ε, α, and β values over all episode steps on a single figure.
4. THE System SHALL compute and report the Off_Peak_Uplift as: `(mean sessions in discount-regime hours after optimisation − mean sessions in discount-regime hours at baseline) / mean sessions at baseline × 100`.
5. WHERE LightGBM is installed, THE System SHALL train a LightGBM model alongside XGBoost and produce a side-by-side RMSE/MAE/R² comparison table exported to `outputs/model_comparison.csv`.
6. THE System SHALL generate a sensitivity analysis report by running the deterministic fallback with ε ∈ {0.5, 1.0, 1.5, 2.0} and reporting the resulting revenue gain distribution, exported to `outputs/sensitivity_analysis.csv`.

---

### Requirement 9: Configuration and Environment Management

**User Story:** As a developer, I want all constants, paths, and model hyperparameters centralised in a single configuration module, so that the system is easy to tune and reproduce.

#### Acceptance Criteria

1. THE System SHALL centralise all pricing constants (P_BASE, P_SURGE_CAP, P_DISCOUNT_FLOOR, SURGE_THRESHOLD, DISCOUNT_THRESHOLD), model hyperparameters, file paths, and Gemini model name in `src/config.py`.
2. THE System SHALL read the `GEMINI_API_KEY` exclusively from the environment variable `GEMINI_API_KEY` and SHALL NOT hardcode any API key in source files.
3. WHEN `build_gemini_model()` is called without `GEMINI_API_KEY` set, THE System SHALL raise an `EnvironmentError` with a clear message showing the exact export command needed.
4. THE System SHALL define all Pydantic schemas (`ForecastState`, `PricingDecision`, `LearningUpdate`) in `src/config.py` with field-level validation ensuring pricing bounds and non-negative values.
5. THE System SHALL define the `engineer_features()` function in `src/config.py` as the single source of truth for feature engineering, used by both the EDA_Module and the DemandPredictionAgent.

---

### Requirement 10: Output Reproducibility and Documentation

**User Story:** As a submission evaluator, I want clean, reproducible outputs with documented assumptions, so that results can be independently verified.

#### Acceptance Criteria

1. THE System SHALL produce all CSV outputs (unified_analytical_base.csv, agentic_outcomes.csv, predictions.csv) with consistent column names matching the documented schema.
2. WHEN the Pipeline is run with the same input files, THE Pipeline SHALL produce a deterministic Unified_Base with the same row count and column values on repeated runs.
3. THE System SHALL document all assumptions made during preprocessing (zero-kWh exclusion, occupancy scaling factor of 1.2, queue proxy formula) as inline comments in the Pipeline source.
4. THE System SHALL include a `RANDOM_STATE = 42` constant used consistently across all model training calls to ensure reproducibility.
5. THE System SHALL log the software versions of key dependencies (pandas, numpy, xgboost, sklearn) at the start of each pipeline and agent run.
