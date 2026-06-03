# Implementation Plan: EV Charging Analytics & Tariff Optimisation (OP'26)

## Overview

Restructure the existing monolithic codebase into a clean package hierarchy, fix known data-quality issues, implement all three agents with Gemini + deterministic fallbacks, wire the closed feedback loop in the orchestrator, and cover all 15 correctness properties with Hypothesis property tests and unit tests.

## Tasks

- [x] 1. Repository restructuring and project skeleton
  - Create the target directory layout: `src/pipeline/`, `src/eda/`, `src/agents/`, `src/utils/`, `data/raw/`, `data/processed/`, `outputs/eda/`, `tests/`
  - Add `__init__.py` files to every `src/` sub-package
  - Remove the duplicate `preprocessed_data/p1_preprocess_pipeline.py` file so each module exists in exactly one canonical location
  - Create empty stub files for every module listed in the architecture (`src/config.py`, `src/pipeline/preprocess.py`, `src/eda/plots.py`, `src/agents/demand.py`, `src/agents/pricing.py`, `src/agents/monitoring.py`, `src/utils/logging_utils.py`, `orchestrator.py`)
  - _Requirements: 1.1, 1.2, 1.7_

- [x] 2. `src/config.py` — centralised constants, schemas, Gemini factory, feature engineering
  - [x] 2.1 Define all pricing constants, model hyperparameters, file paths, and `RANDOM_STATE = 42`
    - `P_BASE`, `P_SURGE_CAP`, `P_DISCOUNT_FLOOR`, `SURGE_THRESHOLD`, `DISCOUNT_THRESHOLD`, `TRAIN_RATIO`
    - `XGB_PARAMS`, `LGB_PARAMS` dicts
    - `RAW_ACN_PATH`, `RAW_URBAN_DIR`, `PROCESSED_BASE_PATH`, `OUTPUTS_DIR`, `EDA_OUTPUTS_DIR`
    - `GEMINI_MODEL = "gemini-2.0-flash"`
    - _Requirements: 9.1, 10.4_

  - [x] 2.2 Implement `build_gemini_model(system_instruction, temperature)` factory
    - Read `GEMINI_API_KEY` exclusively from environment; raise `EnvironmentError` with exact `export` command if missing
    - _Requirements: 9.2, 9.3_

  - [x] 2.3 Define Pydantic schemas `ForecastState`, `PricingDecision`, `LearningUpdate` with field validators
    - `ForecastState`: `u_pred` clipped to `[0,1]` via `@field_validator`, `kwh_delivered > 0`
    - `PricingDecision`: `p_new` clipped to `[P_DISCOUNT_FLOOR, P_SURGE_CAP]` via validator, `regime` literal
    - `LearningUpdate`: all metric fields, delta bounds as documented
    - _Requirements: 9.4_

  - [x] 2.4 Implement `engineer_features(df)` as the single source of truth for causal feature engineering
    - All lag/rolling features anchored with `.shift(1)`; `dropna()` called only after all features are computed
    - Define `FEATURE_COLS` list (22 features)
    - _Requirements: 4.2, 9.5_

  - [ ]* 2.5 Write property test for Pydantic schema bounds (Property 15)
    - **Property 15: Pydantic schemas enforce field bounds**
    - **Validates: Requirements 9.4**
    - Use `st.floats` outside valid ranges; verify `ValidationError` is raised or value is clipped

  - [ ]* 2.6 Write unit tests for `build_gemini_model()` without API key
    - Verify `EnvironmentError` is raised when `GEMINI_API_KEY` is unset

- [x] 3. `src/utils/logging_utils.py` — version logging and log configuration
  - Implement `configure_logging(level)` setting up root logger with timestamp format
  - Implement `log_dependency_versions()` logging pandas, numpy, xgboost, sklearn versions at INFO level
  - _Requirements: 10.5_

- [x] 4. `src/pipeline/preprocess.py` — full preprocessing pipeline
  - [x] 4.1 Implement `_load_acn(path)` — load ACN Excel, parse timestamps, log and exclude zero/null kWh rows
    - Raise `FileNotFoundError` with descriptive message if file is missing
    - Log count of excluded zero/null kWh rows at WARNING level
    - _Requirements: 2.1, 2.2, 2.7_

  - [x] 4.2 Implement `_load_urban(urban_dir)` — melt wide matrices, validate merge nulls
    - Melt volume, occupancy, duration CSVs from wide to long format
    - Merge on `[time_step, station_node]`; raise `ValueError` if unexpected nulls appear in key columns
    - Raise `FileNotFoundError` with descriptive message if any required CSV is missing
    - _Requirements: 2.1, 2.3, 2.7_

  - [ ]* 4.3 Write property test for zero-kWh exclusion (Property 1)
    - **Property 1: Zero-kWh rows are excluded from pipeline output**
    - **Validates: Requirements 2.2**
    - Generate ACN DataFrames with random zero/null kWh rows; verify `acn_total_kwh` aggregates exclude them

  - [ ]* 4.4 Write property test for merge null-free guarantee (Property 2)
    - **Property 2: Melted UrbanEV merge produces no nulls in key columns**
    - **Validates: Requirements 2.3**
    - Generate random wide matrices; verify melted merge has zero nulls in `traffic_volume`, `occupancy_density`, `avg_duration`

  - [ ]* 4.5 Write property test for missing file raises FileNotFoundError (Property 4)
    - **Property 4: Missing input file raises FileNotFoundError**
    - **Validates: Requirements 2.7**
    - For any non-existent path string, verify `run_pipeline()` raises `FileNotFoundError` containing the path

  - [x] 4.6 Implement `_aggregate_acn_hourly(df)` and `_aggregate_urban_hourly(df)`
    - ACN: group by `hourly_timestamp` → `acn_sessions_count`, `acn_total_kwh`, `acn_base_revenue`
    - UrbanEV: group by `time_step` → `urban_mean_utilization`, `urban_peak_queue`, `urban_total_volume`
    - Raise `ValueError` if either aggregation produces zero rows
    - _Requirements: 2.4, 2.5_

  - [x] 4.7 Implement `_align_and_merge(acn_hourly, urban_hourly)` and `run_pipeline()`
    - Align by positional index; add `hour_of_day`, `day_of_week`, `is_weekend`
    - Log source row counts and final merged row count
    - Write output CSV with documented schema comment at top; create `data/processed/` if needed
    - Expose as callable function (not only `__main__`)
    - Document all preprocessing assumptions as inline comments (zero-kWh exclusion, 1.2 scaling, queue proxy formula)
    - _Requirements: 2.5, 2.6, 1.5, 1.6, 10.3_

  - [ ]* 4.8 Write property test for pipeline determinism (Property 14)
    - **Property 14: Pipeline is deterministic (idempotent on same inputs)**
    - **Validates: Requirements 10.2**
    - Run `run_pipeline()` twice on same sample data; verify `df1.equals(df2)`

  - [ ]* 4.9 Write unit test for `run_pipeline()` end-to-end with sample data files
    - Verify output CSV is created with correct schema columns

- [x] 5. Checkpoint — pipeline complete
  - Ensure all pipeline tests pass, ask the user if questions arise.

- [ ] 6. `src/eda/plots.py` — all EDA visualisation functions
  - [x] 6.1 Implement `run_eda(base_path, acn_path, output_dir)` entry point and core demand plots
    - `plot_demand_trend`: long-run trend with 7-day rolling mean
    - `plot_intraday_cycle`: hourly demand cycle
    - `plot_weekday_weekend`: weekday vs weekend profiles with DoW×Hour heatmap
    - Accept `--base` and `--acn` CLI arguments; default to canonical processed paths
    - _Requirements: 3.1, 3.2, 3.4_

  - [x] 6.2 Implement ACN distribution and volatility plots
    - `plot_acn_distributions`: kWh, duration, station usage, hourly counts
    - `plot_peak_volatility`: peak/shoulder/off-peak volatility box plots
    - `plot_correlation_heatmap`: feature correlation heatmap
    - _Requirements: 3.2_

  - [x] 6.3 Implement revenue analysis and session efficiency plots
    - `plot_revenue_analysis`: tariff distribution, comparison, regime pie, hourly gain; compute Revenue Gain % as `((Dynamic − Baseline) / Baseline) × 100` vs ₹15/kWh baseline
    - `plot_acn_session_efficiency`: session efficiency scatter
    - `plot_tariff_narrative`: annotated pricing zones
    - _Requirements: 3.2, 3.3_

  - [x] 6.4 Implement `plot_feature_importance(df, output_dir)` with graceful XGBoost fallback
    - Wrap xgboost import in `try/except ImportError`; log WARNING and return without raising if unavailable
    - _Requirements: 3.5_

  - [x] 6.5 Implement post-run plots reading from orchestrator outputs
    - `plot_predicted_vs_actual(predictions_path, output_dir)`: reads `outputs/predictions.csv`; marks train/test split boundary
    - `plot_reward_convergence(outcomes_path, output_dir)`: reads `outputs/agentic_outcomes.csv`; per-step reward + 50-step rolling mean
    - `plot_theta_evolution(outcomes_path, output_dir)`: reads `outputs/agentic_outcomes.csv`; ε, α, β on single figure
    - Check file existence before loading; log WARNING and skip if absent
    - _Requirements: 3.6, 3.7, 8.1, 8.2, 8.3_

  - [ ]* 6.6 Write unit tests for EDA output file existence
    - Verify expected PNG files are created in `outputs/eda/` after `run_eda()` with sample data

- [ ] 7. `src/agents/demand.py` — DemandPredictionAgent
  - [x] 7.1 Implement `DemandPredictionAgent.__init__()` with chronological 80/20 split and XGBoost training
    - Load CSV, call `engineer_features()` from `src/config.py`, apply strict chronological split (no shuffling)
    - Log train row count, test row count, and feature list
    - Train `MultiOutputRegressor(XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE))`
    - _Requirements: 4.1, 4.2, 4.6_

  - [ ]* 7.2 Write property test for chronological split (Property 5)
    - **Property 5: Train/test split is strictly chronological**
    - **Validates: Requirements 4.1**
    - For any sorted timestamp series, verify `max(train_timestamps) < min(test_timestamps)`

  - [ ]* 7.3 Write property test for causal lag features (Property 6)
    - **Property 6: Lag features are causally correct**
    - **Validates: Requirements 4.2**
    - For any DataFrame, verify `util_lag1[i] == urban_mean_utilization[i-1]` before `dropna`

  - [x] 7.4 Implement `predict_state(row_idx)` returning validated `ForecastState`
    - Clip `u_pred` to `[0,1]`, floor `kwh_delivered` at `0.01`
    - _Requirements: 4.4_

  - [ ]* 7.5 Write property test for ForecastState bounds (Property 7)
    - **Property 7: ForecastState output satisfies schema bounds**
    - **Validates: Requirements 4.4**
    - For any row with extreme raw values, verify `u_pred ∈ [0,1]` and `kwh_delivered ≥ 0.01`

  - [x] 7.6 Implement `evaluation_metrics()` returning RMSE, MAE, R² for both targets
    - _Requirements: 4.3_

  - [x] 7.7 Implement optional LightGBM backend and `compare_backends()` method
    - Selectable via `use_lightgbm=True` constructor argument
    - `compare_backends()` returns DataFrame with RMSE/MAE/R² for both backends side-by-side
    - _Requirements: 4.5, 8.5_

  - [ ]* 7.8 Write unit tests for `evaluation_metrics()` return shape and LightGBM comparison
    - Verify dict has both target keys with RMSE/MAE/R² sub-keys
    - When `use_lightgbm=True`, verify `compare_backends()` returns DataFrame with expected columns

- [ ] 8. `src/agents/pricing.py` — TariffPricingAgent
  - [x] 8.1 Implement `TariffPricingAgent.__init__()` with theta initialisation and bounds
    - `_THETA_BOUNDS = {"epsilon": (0.1, 5.0), "alpha": (1.0, 10.0), "beta": (1.0, 10.0)}`
    - _Requirements: 5.5_

  - [x] 8.2 Implement `compute_tariff(state)` with Gemini call and deterministic sigmoid fallback
    - Retry up to `max_retries` times with exponential backoff (`1.5 × attempt` seconds)
    - After exhausting retries, use deterministic sigmoid fallback and log ERROR
    - Enforce regime/price consistency: surge if `u_pred > 0.80`, discount if `u_pred < 0.30`, neutral otherwise
    - `p_new` always clipped to `[P_DISCOUNT_FLOOR, P_SURGE_CAP]` regardless of Gemini output
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_

  - [ ]* 8.3 Write property test for pricing regime and price bounds (Property 8)
    - **Property 8: Pricing regime and price bounds are consistent**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.6**
    - For any `u_pred ∈ [0,1]`, verify regime and price bounds from deterministic fallback

  - [x] 8.4 Implement `apply_update(delta)` with theta clipping
    - Add delta to theta; clip each parameter to its defined bounds
    - _Requirements: 5.5_

  - [ ]* 8.5 Write property test for theta bounds after any update (Property 9)
    - **Property 9: Theta parameters remain within bounds after any update**
    - **Validates: Requirements 5.5**
    - For any delta vector, verify theta stays within `ε ∈ [0.1,5.0]`, `α ∈ [1.0,10.0]`, `β ∈ [1.0,10.0]`

  - [x] 8.6 Implement `run_sensitivity_analysis(test_df, epsilon_values)` for ε sweep
    - Run deterministic fallback with each ε value; return revenue gain distribution DataFrame
    - Export schema: `epsilon`, `mean_revenue_gain_pct`, `std_revenue_gain_pct`, `min_revenue_gain_pct`, `max_revenue_gain_pct`
    - _Requirements: 8.6_

  - [ ]* 8.7 Write unit test for Gemini fallback in TariffPricingAgent
    - Mock Gemini to always raise; verify valid `PricingDecision` is returned with correct regime/bounds

- [ ] 9. `src/agents/monitoring.py` — MonitoringLearningAgent
  - [x] 9.1 Implement `MonitoringLearningAgent.__init__()` and `step(state, decision)`
    - Compute all OP'26 metrics: `demand_shift`, `revenue_new`, `revenue_gain_pct`, `charger_utilisation`, `avg_wait_reduction`, `pricing_efficiency`, `reward`
    - Use revenue formula: `revenue_new = p_new × kwh × max(0.05, 1 + demand_shift)` where `demand_shift = −ε × ((p_new − 15) / 15)`
    - Scale Gemini-proposed Δθ by `η = η₀ / (1 + decay × t)` before calling `apply_update()`
    - Append all metrics plus θ state and `lr_used` to episode log
    - _Requirements: 6.1, 6.2, 6.3_

  - [ ]* 9.2 Write property test for revenue formula consistency (Property 10)
    - **Property 10: Revenue formula is applied consistently**
    - **Validates: Requirements 6.2**
    - For any `p_new`, `kwh`, `epsilon`, verify `revenue_new` equals the documented formula

  - [ ]* 9.3 Write property test for learning rate monotonic decay (Property 11)
    - **Property 11: Learning rate schedule decays monotonically**
    - **Validates: Requirements 6.3**
    - For any `η₀ > 0` and `decay > 0`, verify `η_t` is strictly decreasing as `t` increases

  - [x] 9.4 Implement Gemini retry logic with deterministic economic fallback
    - Retry up to `max_retries` times with exponential backoff; use fallback and log ERROR after exhaustion
    - _Requirements: 6.4_

  - [x] 9.5 Implement `summary()` returning DataFrame with one row per episode step
    - Include all metric columns from `LearningUpdate` schema plus `step`, `timestamp`, θ state columns, `lr_used`
    - _Requirements: 6.5_

  - [x] 9.6 Implement `off_peak_uplift(baseline_df)` computing Off_Peak_Uplift metric
    - Formula: `(mean_post − mean_baseline) / mean_baseline × 100` over discount-regime hours only
    - _Requirements: 6.6, 8.4_

  - [ ]* 9.7 Write property test for Off_Peak_Uplift formula (Property 12)
    - **Property 12: Off_Peak_Uplift formula is correct**
    - **Validates: Requirements 6.6, 8.4**
    - For any session count arrays, verify the formula is applied correctly over discount-regime hours

  - [ ]* 9.8 Write unit tests for `summary()` DataFrame columns and Gemini fallback
    - Verify all required columns are present after N steps
    - Mock Gemini to always raise; verify valid `LearningUpdate` is returned

- [x] 10. Checkpoint — all agents complete
  - Ensure all agent tests pass, ask the user if questions arise.

- [ ] 11. `orchestrator.py` — AgenticOrchestrator, CLI, exports, and final report
  - [x] 11.1 Implement `AgenticOrchestrator.__init__()` with startup validation
    - Validate `csv_path` exists via `Path(csv_path).exists()`; raise `FileNotFoundError` with descriptive message
    - Validate `GEMINI_API_KEY` is set (via `build_gemini_model()` call); raise `EnvironmentError` if missing
    - Construct `DemandPredictionAgent`, `TariffPricingAgent`, `MonitoringLearningAgent`
    - Create `outputs/` and `outputs/eda/` with `Path.mkdir(parents=True, exist_ok=True)`
    - _Requirements: 7.6_

  - [x] 11.2 Implement `run(max_steps, verbose_every)` — the three-agent feedback loop
    - Sequential loop: `predict_state(i)` → `compute_tariff(state)` → `monitor.step(state, decision)` → `time.sleep(api_delay)`
    - _Requirements: 7.1, 7.7_

  - [ ]* 11.3 Write property test for episode step count invariant (Property 13)
    - **Property 13: Episode step count invariant**
    - **Validates: Requirements 7.1, 7.4**
    - For any N steps, verify `summary()` has exactly N rows and exported CSV has exactly N data rows

  - [x] 11.4 Implement `_print_final_report(df)` covering all OP'26 metrics
    - Print RMSE/MAE/R² for demand prediction, Revenue Gain %, Charger Utilisation Rate, Off_Peak_Uplift, Avg Wait Reduction, Pricing_Efficiency_Score
    - _Requirements: 7.3_

  - [x] 11.5 Implement all export methods
    - `export(path)` → `outputs/agentic_outcomes.csv` with all schema columns
    - `export_predictions(path)` → `outputs/predictions.csv` with actual vs predicted columns
    - `export_model_comparison(path)` → `outputs/model_comparison.csv` (when LightGBM enabled)
    - `export_sensitivity_analysis(path)` → `outputs/sensitivity_analysis.csv`
    - _Requirements: 7.4, 7.5, 8.5, 8.6_

  - [x] 11.6 Implement `parse_args()` and `main()` CLI entry point
    - Accept all documented args: `--csv`, `--steps`, `--verbose`, `--lr`, `--decay`, `--delay`, `--epsilon`, `--alpha`, `--beta`, `--out`, `--predictions`, `--log-level`, `--lightgbm`
    - Call `configure_logging()` and `log_dependency_versions()` at startup
    - _Requirements: 7.2, 10.5_

  - [ ]* 11.7 Write unit tests for CLI argument parsing and orchestrator initialisation errors
    - Verify all `--` arguments are accepted by `parse_args()`
    - Verify `FileNotFoundError` is raised when CSV path is missing
    - Verify output CSV schemas match documented column names

- [x] 12. `tests/conftest.py` — shared fixtures
  - Create shared pytest fixtures: sample ACN DataFrame, sample UrbanEV DataFrames, mock Gemini client that always raises, sample `ForecastState` and `PricingDecision` objects
  - Configure Hypothesis profile: `max_examples=100`, `suppress_health_check=[HealthCheck.too_slow]`
  - _Requirements: (test infrastructure)_

- [ ] 13. `requirements.txt` and `README.md`
  - [x] 13.1 Create `requirements.txt` with all Python dependencies at pinned versions
    - Include: pandas, numpy, xgboost, lightgbm, scikit-learn, google-generativeai, pydantic, hypothesis, pytest, openpyxl, matplotlib, seaborn
    - _Requirements: 1.3_

  - [x] 13.2 Create `README.md` with setup instructions, directory layout, and step-by-step run commands
    - Document: environment setup, `GEMINI_API_KEY` export, pipeline run, EDA run, orchestrator run, post-run EDA
    - _Requirements: 1.4_

- [x] 14. Final checkpoint — full system integration
  - Ensure all tests pass across `test_pipeline.py`, `test_features.py`, `test_agents.py`, `test_monitoring.py`, `test_schemas.py`, `test_eda.py`
  - Verify no module executes side effects at import time
  - Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- LightGBM tasks (7.7, 11.5 model comparison export) are optional — the system must work without LightGBM installed
- All 15 correctness properties from the design document are covered by property tests in tasks 2.5, 4.3, 4.4, 4.5, 4.8, 7.2, 7.3, 7.5, 8.3, 8.5, 9.2, 9.3, 9.7, 11.3, 2.5
- Property tests use Hypothesis with `@settings(max_examples=100)` and must include the comment `# Feature: ev-charging-analytics-optimization, Property N: <property_text>`
- `RANDOM_STATE = 42` must be passed to all model training calls for reproducibility

export GEMINI_API_KEY='your-key'
python -m src.pipeline.preprocess          # Step 1: preprocess data
python -m src.eda.plots                    # Step 2: EDA plots
python orchestrator.py --steps 200         # Step 3: agentic loop
python -m src.eda.plots                    # Step 4: post-run plots
pytest tests/ -q                           # Run all tests