# Agentic EV Tariff Optimization System

**Open Project 2026 - Society of Business**

A fully autonomous multi-agent system for optimizing electric vehicle (EV) charging tariffs using real-world data from ACN-Data (Caltech) and UrbanEV (Shenzhen) datasets.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Testing](#testing)
- [License](#license)

---

## Overview

This project addresses the challenge of dynamic pricing in EV charging infrastructure through an intelligent multi-agent system. The system continuously learns and adapts pricing strategies to balance three critical objectives:

1. **Revenue Maximization**: Optimize charging tariffs beyond static baseline pricing
2. **Utilization Distribution**: Smooth demand across temporal windows to reduce congestion
3. **Queue Management**: Minimize customer waiting times during peak periods

### The Three-Agent Architecture

The system employs three specialized autonomous agents that work in concert:

- **Demand Agent** (ML-based): Predicts utilization, queue length, and congestion probability using XGBoost trained on historical charging patterns
- **Pricing Agent** (LLM-powered): Determines optimal tariffs through contextual reasoning with soft confidence weighting to handle boundary cases
- **Monitoring Agent** (LLM-powered): Evaluates system outcomes and adjusts pricing parameters using reward decomposition for independent parameter learning

### Real-World Data

The system is trained and validated on two major EV charging datasets:

- **ACN-Data**: 30,000+ charging sessions from Caltech/JPL workplace stations
- **UrbanEV**: 24,798 charging piles with 5-minute interval data from Shenzhen, China

Combined, these datasets provide ~991 training samples with diverse temporal and spatial patterns representing both workplace and urban charging behavior.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Real-World Data Sources                   │
│          ACN-Data (Caltech/JPL) + UrbanEV (Shenzhen)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Preprocessing Pipeline                │
│  • Per-zone utilization computation (21%-41% range)         │
│  • Temporal feature alignment (hour, day, weekend)          │
│  • Data-driven peak hour detection (75th percentile)        │
│  • Feature engineering (sessions, kWh, spatial, temporal)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Demand Prediction Agent                   │
│                      (XGBoost Regressor)                     │
│  Input: sessions, kWh, hour, day, weekend, peak, spatial    │
│  Output: utilization, queue_length, congestion_probability  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Tariff Pricing Agent                       │
│                   (LLM-Powered Reasoning)                    │
│  • Regime classification (surge/neutral/discount)           │
│  • Soft confidence weighting near thresholds                │
│  • Natural language rationale generation                    │
│  Output: optimal_tariff, regime, confidence, reasoning      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Metrics Engine                          │
│  • Revenue gain: (dynamic - baseline) / baseline            │
│  • Reward with decomposition by regime                      │
│  • Utilization tracking (before/after pricing)              │
│  • Queue reduction measurement                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Monitoring & Learning Agent                 │
│                   (LLM-Powered Evaluation)                   │
│  • Outcome evaluation by regime (surge/neutral/discount)    │
│  • Reward decomposition (separate signals per parameter)    │
│    - ε (elasticity) ← Revenue delta in neutral regime       │
│    - α (surge) ← Congestion penalty in surge regime         │
│    - β (discount) ← Off-peak uplift in discount regime      │
│  Output: Δε, Δα, Δβ adjustments with natural language       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Parameter Update with Learning Rate Decay         │
│  θ_new = θ_old + learning_rate × Δθ × decay_factor          │
│  Parameters: ε (elasticity), α (surge), β (discount)        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Convergence Check (4 Criteria)              │
│  1. Revenue variance < threshold                            │
│  2. Parameter stability (small updates)                     │
│  3. Utilization health maintained                           │
│  4. Queue reduction achieved                                │
└─────────────────────────────────────────────────────────────┘
             │                                    │
             ▼                                    ▼
      [Converged] ──────────────────────→  [Continue Loop]
      Export Results                        Next Iteration
```

---

## Key Features

### 1. Real Data Pipeline with Robust Preprocessing

**Per-Zone Utilization Computation**:
- Correctly computes utilization as average across zones (each zone: sessions/capacity)
- Achieves realistic 21%-41% range that interacts meaningfully with 30%/80% thresholds
- Avoids incorrect system-wide averaging that produces artificially low values

**Temporal Alignment**:
- Datasets from different calendar periods aligned on behavioral features (hour, day, weekend)
- Enables cross-dataset learning despite different time windows
- Preserves charging behavior patterns across geographic locations

**Data-Driven Peak Hours**:
- Dynamically computed from 75th percentile of actual usage patterns
- JPL site peaks at 14:00-17:00 (workplace arrival/midday charging)
- Avoids hardcoded assumptions about commute times

### 2. Soft Confidence Weighting

Traditional hard boundaries can suppress valuable signals. This system uses confidence-based blending:

```python
# Calculate confidence based on distance from threshold
confidence = abs(predicted_utilization - threshold) / max(threshold, 1 - threshold)

if confidence > 0.15:  
    # Far from threshold - trust LLM reasoning
    final_regime = llm_regime
else:  
    # Near threshold - use rule-based classification
    final_regime = rule_based_regime
```

This preserves LLM insights while maintaining reliability near decision boundaries.

### 3. Reward Decomposition for Independent Parameter Learning

Instead of a single reward signal affecting all parameters, each parameter responds to specific outcomes:

- **ε (demand elasticity)**: Updated based on revenue delta in neutral regime
- **α (surge multiplier)**: Updated based on congestion penalty in high-utilization periods
- **β (discount multiplier)**: Updated based on off-peak demand uplift in low-utilization periods

This enables independent learning rates and prevents parameter interference.

### 4. Multi-Objective Optimization

The system balances three objectives through careful reward shaping:

1. **Revenue**: Dynamic tariffs vs ₹15/kWh baseline (Indian market reference)
2. **Utilization**: Smooth temporal distribution reduces infrastructure strain
3. **Queue Management**: Minimize customer waiting through demand shifting

---

## Installation

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- API key for Groq LLM (set in environment variable `GROQ_API_KEY`)

### Setup Steps

```bash
# 1. Clone the repository
git clone <repository-url>
cd analytics

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# .venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set LLM API key (required for full agentic behavior)
export GROQ_API_KEY="your-api-key-here"
```

### Dependencies

Core libraries (see `requirements.txt` for full list):
- **Data Processing**: pandas, numpy, scikit-learn
- **Machine Learning**: xgboost
- **LLM Integration**: groq (for LLM-powered agents)
- **Visualization**: matplotlib, seaborn
- **Testing**: pytest, hypothesis

---

## Usage

### Step 1: Prepare the Data

Run the data preprocessing pipeline to generate the unified analytical base:

```bash
python rebuild_data.py
```

**What this does**:
- Parses ACN-Data JSON and UrbanEV CSV files
- Computes per-zone utilization metrics
- Aligns temporal features across datasets
- Derives data-driven peak hours
- Generates feature-engineered dataset

**Output**: `data/processed/unified_analytical_base.csv` (~991 rows)

**Expected console output**:
```
Processing ACN-Data...
Processing UrbanEV data...
Aligning temporal features...
Computing data-driven peak hours...
Unified dataset saved: 991 rows
Utilization range: 21.3% - 40.8%
```

### Step 2: Exploratory Data Analysis (Optional)

Generate visualizations and statistical summaries:

```bash
python run_eda.py
```

**Outputs**:
- `outputs/figures/temporal_patterns.png` - Intraday charging patterns
- `outputs/figures/peak_analysis.png` - Peak vs off-peak comparison
- `outputs/figures/utilization_distribution.png` - Utilization histograms
- `outputs/figures/revenue_analysis.png` - Revenue patterns
- `outputs/figures/energy_patterns.png` - Energy consumption by time
- `outputs/figures/correlation_matrix.png` - Feature correlations
- `outputs/eda/summary_statistics.csv` - Descriptive statistics
- `outputs/eda/eda_summary.csv` - Key metrics
- `outputs/eda/key_insights.txt` - Written insights

### Step 3: Run the Optimization

Execute the full agentic optimization loop:

```bash
python run_agentic.py
```

**What this does**:
1. Loads preprocessed data
2. Trains Demand Agent (XGBoost) on historical patterns
3. Initializes pricing parameters (ε=0.25, α=2.5, β=2.5)
4. Runs optimization loop (default: 40 iterations):
   - Demand Agent predicts utilization/queue/congestion
   - Pricing Agent determines optimal tariff and regime
   - Metrics Engine calculates revenue gain and rewards
   - Monitoring Agent evaluates outcomes and proposes parameter updates
   - System updates parameters with learning rate decay
   - Convergence checker monitors for stability
5. Exports results to CSV

**Output**: `outputs/agentic_outcomes.csv`

**Expected console output**:
```
Loading data...
Training Demand Agent...
XGBoost trained: RMSE=0.0842, R²=0.8234

Starting optimization loop...
Step 1/40: Regime=neutral, Price=₹14.20, Revenue Gain=+2.3%, Reward=0.15
Step 2/40: Regime=surge, Price=₹19.50, Revenue Gain=+5.8%, Reward=0.32
...
Step 40/40: Regime=discount, Price=₹12.30, Revenue Gain=+8.1%, Reward=0.41

Convergence achieved after 35 steps
Results saved: outputs/agentic_outcomes.csv
```

### Step 4: Analyze Results

Examine the optimization outcomes:

```bash
# View first few rows
head outputs/agentic_outcomes.csv

# Or load in Python
python
>>> import pandas as pd
>>> df = pd.read_csv('outputs/agentic_outcomes.csv')
>>> print(df[['step', 'regime', 'p_new', 'revenue_gain_pct', 'epsilon', 'alpha', 'beta']].head(10))
```

---

## Project Structure

```
analytics/
│
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── pytest.ini                  # Test configuration
│
├── run_agentic.py              # Main entry point for optimization
├── rebuild_data.py             # Data preprocessing pipeline
├── run_eda.py                  # Exploratory data analysis script
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── config.py               # System configuration
│   ├── data_loader.py          # Data loading utilities
│   ├── orchestrator.py         # Main optimization loop
│   │
│   ├── agents/                 # Three autonomous agents
│   │   ├── __init__.py
│   │   ├── demand.py           # XGBoost demand predictor
│   │   ├── pricing.py          # LLM-powered pricing agent
│   │   └── monitoring.py       # LLM-powered monitoring agent
│   │
│   ├── preprocessing/          # Data pipeline
│   │   ├── __init__.py
│   │   ├── acn_parser.py       # ACN-Data JSON parser
│   │   ├── urbanev_parser.py   # UrbanEV CSV parser
│   │   ├── dataset_fusion.py   # Dataset alignment & clustering
│   │   └── real_data_pipeline.py  # Complete preprocessing pipeline
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── llm_provider.py     # LLM wrapper (Groq)
│       ├── metrics.py          # Metrics computation engine
│       └── convergence.py      # Convergence checker
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_acn_parser.py      # ACN parser tests (16 tests)
│   └── test_urbanev_parser.py  # UrbanEV parser tests (13 tests)
│
├── data/                       # Datasets
│   ├── raw/                    # Original data files
│   │   ├── acndata_sessions.json.xlsx  # ACN charging sessions
│   │   ├── occupancy.csv       # UrbanEV occupancy data
│   │   ├── information.csv     # Station information (capacities)
│   │   ├── time.csv            # Temporal features
│   │   ├── stations.csv        # Station metadata
│   │   ├── price.csv           # Historical pricing data
│   │   ├── distance.csv        # Spatial features
│   │   ├── duration.csv        # Session durations
│   │   └── volume.csv          # Energy volumes
│   │
│   └── processed/              # Generated datasets
│       └── unified_analytical_base.csv  # Preprocessed ML-ready data
│
└── outputs/                    # Results and visualizations
    ├── agentic_outcomes.csv    # Main optimization results
    ├── figures/                # EDA visualizations (6 charts)
    ├── eda/                    # EDA summary statistics
    └── presentation/           # Presentation materials
```

---

## Implementation Details

### Demand Prediction Agent

**Algorithm**: XGBoost Regressor with multi-output prediction

**Features** (input):
- `sessions_count`: Number of concurrent charging sessions
- `total_kwh`: Total energy delivered in time window
- `hour_of_day`: Hour (0-23)
- `day_of_week`: Day (0=Monday, 6=Sunday)
- `is_weekend`: Binary indicator
- `is_peak_hour`: Data-driven peak period indicator
- Spatial features: Zone identifiers and clustering

**Targets** (output):
- `utilization`: Predicted charger utilization rate (0-1)
- `queue_length`: Expected queue length (proxy for waiting time)
- `congestion_probability`: Likelihood of congestion (0-1)

**Training**:
- Dataset: ~991 historical charging windows
- Validation: 80/20 train-test split
- Hyperparameters: `n_estimators=100, max_depth=5, learning_rate=0.1`
- Evaluation: RMSE, MAE, R² computed during training

**Performance**:
- Typical RMSE: ~0.08-0.10 for utilization prediction
- Typical R²: ~0.80-0.85
- Training time: 3-5 seconds

### Tariff Pricing Agent

**Core Logic**: LLM-powered contextual reasoning with rule-based fallback

**Regime Classification**:
1. **Surge Regime** (u > 80%): High utilization requires price increase
2. **Discount Regime** (u < 30%): Low utilization benefits from price decrease
3. **Neutral Regime** (30% ≤ u ≤ 80%): Moderate pricing adjustments

**Soft Confidence Weighting**:
- Computes confidence as distance from thresholds (30%, 80%)
- If confidence > 0.15: Trust LLM reasoning
- If confidence ≤ 0.15: Use rule-based classification
- Prevents threshold boundary instability

**Tariff Calculation**:
```python
if regime == 'surge':
    price = baseline × (1 + α × congestion_prob)
elif regime == 'discount':
    price = baseline × (1 - β × (1 - utilization))
else:  # neutral
    price = baseline × (1 + ε × demand_shift)
```

**Bounds**: Tariffs constrained to [₹10, ₹22] per kWh

**Fallback**: If LLM unavailable, uses purely rule-based classification

### Monitoring & Learning Agent

**Purpose**: Evaluate outcomes and propose parameter adjustments

**Reward Decomposition**:

1. **Epsilon (ε) - Demand Elasticity**:
   - Signal: Revenue delta in neutral regime
   - Logic: If revenue increased → increase ε (more aggressive), else decrease
   - Update: Δε ∈ [-0.05, +0.05]

2. **Alpha (α) - Surge Multiplier**:
   - Signal: Congestion penalty in surge regime
   - Logic: If congestion reduced → maintain/increase α, else decrease
   - Update: Δα ∈ [-0.2, +0.2]

3. **Beta (β) - Discount Multiplier**:
   - Signal: Off-peak uplift in discount regime
   - Logic: If demand shifted successfully → maintain/increase β, else adjust
   - Update: Δβ ∈ [-0.2, +0.2]

**Learning Rate Decay**:
```python
learning_rate = initial_rate × (1 / (1 + decay_factor × step))
θ_new = θ_old + learning_rate × Δθ
```

**Convergence Criteria** (all must be satisfied):
1. Revenue variance < 0.02 over rolling window
2. Parameter updates < 0.01 (stable parameters)
3. Utilization within healthy range [0.25, 0.75]
4. Queue length reduction > 0 (improvement maintained)

### Data Preprocessing Pipeline

**ACN-Data Processing**:
- Parse JSON format from Excel file
- Aggregate sessions by site and time window
- Extract temporal features (timestamp → hour, day, weekend)
- Compute per-site utilization

**UrbanEV Processing**:
- Parse occupancy and information CSV files
- Compute per-zone utilization (sessions / capacity)
- Extract temporal patterns
- Handle missing values (forward fill + interpolation)

**Dataset Fusion**:
- Behavioral alignment: Join on (hour, day_of_week, is_weekend)
- Spatial clustering: K-means on station features
- Feature engineering: Peak hours, session density, energy intensity
- Output: Unified analytical base with ~991 samples

---

## Evaluation Metrics

### Demand Prediction Metrics

| Metric | Description | Typical Value |
|--------|-------------|---------------|
| **RMSE** | Root mean squared error for utilization | 0.08-0.10 |
| **MAE** | Mean absolute error | 0.06-0.08 |
| **R² Score** | Variance explained by model | 0.80-0.85 |

### Tariff Pricing Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Revenue Gain %** | Improvement over baseline | `(revenue_dynamic - revenue_baseline) / revenue_baseline × 100` |
| **Charger Utilization** | Before/after dynamic pricing | `sessions / capacity` (tracked per regime) |
| **Off-Peak Uplift** | Demand shift in discount regime | `(sessions_new - sessions_old) / sessions_old` |

### Monitoring & Learning Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Wait Time Reduction** | Queue length decrease | Positive values indicate improvement |
| **Customer Response Rate** | Demand elasticity proxy | Magnitude of demand shift per price change |
| **Pricing Efficiency Score** | Revenue per kWh over time | Increasing trend indicates learning |

### System-Level Metrics

All metrics are exported to `outputs/agentic_outcomes.csv` with columns:

- `step`: Iteration number
- `regime`: Classification (surge/neutral/discount)
- `p_new`: Dynamic tariff (₹/kWh)
- `u_pred`, `u_actual`: Predicted and actual utilization
- `q_pred`, `q_actual`: Predicted and actual queue length
- `revenue_gain_pct`: Revenue improvement over baseline
- `reward`: Composite reward signal
- `epsilon`, `alpha`, `beta`: Current parameter values
- `demand_shift`: Demand elasticity measurement
- `utilization_new`: Post-pricing utilization
- `pricing_efficiency`: Revenue efficiency metric
- `fallback_used`: LLM availability indicator
- `rationale`: Natural language explanation

---

## Results

### Expected Outcomes

After running `python run_agentic.py`, you should observe:

**Regime Distribution**:
- Surge regime: ~20-30% of iterations (high utilization periods)
- Neutral regime: ~40-50% of iterations (moderate utilization)
- Discount regime: ~20-30% of iterations (low utilization periods)

**Revenue Performance**:
- Average revenue gain: +5% to +12% over baseline
- Peak revenue gain: +15% to +25% in surge periods
- Stable revenue in neutral periods: -2% to +5%

**Parameter Evolution**:
- ε (elasticity): Typically converges to 0.20-0.30 range
- α (surge): Typically converges to 2.0-3.0 range
- β (discount): Typically converges to 2.0-3.0 range
- Convergence: Usually achieved within 30-40 iterations

**Utilization Patterns**:
- Surge regime: Utilization decreases by 5-10% (price dampens demand)
- Discount regime: Utilization increases by 10-20% (price attracts demand)
- Neutral regime: Utilization stable within ±3%

**Queue Management**:
- Queue length reduction: 15-25% in surge periods
- Maintained queue length in neutral periods
- Slight queue increase acceptable in discount periods (system has capacity)

### Validation

To verify your results:

```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/agentic_outcomes.csv')

print('=== Regime Distribution ===')
print(df['regime'].value_counts(normalize=True))

print('\n=== Revenue Gain Statistics ===')
print(df['revenue_gain_pct'].describe())

print('\n=== Parameter Evolution ===')
print(df[['epsilon', 'alpha', 'beta']].describe())

print('\n=== Convergence ===')
final_variance = df['revenue_gain_pct'].tail(20).var()
print(f'Final revenue variance: {final_variance:.4f}')
print(f'Converged: {final_variance < 0.02}')
"
```

---

## Testing

The project includes comprehensive unit tests for data preprocessing components.

### Run All Tests

```bash
pytest tests/ -v
```

**Expected output**:
```
tests/test_acn_parser.py::test_parse_acn_json PASSED
tests/test_acn_parser.py::test_aggregate_by_site PASSED
...
tests/test_urbanev_parser.py::test_parse_occupancy PASSED
tests/test_urbanev_parser.py::test_compute_utilization PASSED
...

====== 29 passed in 2.34s ======
```

### Test Coverage

**ACN Parser Tests** (`test_acn_parser.py` - 16 tests):
- JSON parsing from Excel format
- Site aggregation logic
- Timestamp parsing and timezone handling
- Missing value handling
- Edge cases (empty data, malformed JSON)

**UrbanEV Parser Tests** (`test_urbanev_parser.py` - 13 tests):
- CSV parsing (occupancy, information files)
- Per-zone utilization computation
- Queue length calculation
- Temporal feature extraction
- Capacity handling

### Run Specific Test Suite

```bash
# Test only ACN parser
pytest tests/test_acn_parser.py -v

# Test only UrbanEV parser
pytest tests/test_urbanev_parser.py -v
```

### Test with Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Configuration

Key parameters can be adjusted in `run_agentic.py`:

```python
config = SystemConfig(
    # LLM Configuration
    llm_provider="groq",                    # LLM provider (groq, openai, etc.)
    llm_model="llama-3.3-70b-versatile",    # Model name
    
    # Pricing Configuration
    baseline_tariff_per_kwh=15.0,           # Baseline price (₹/kWh)
    pricing_bounds=(10.0, 22.0),            # Min and max tariff bounds
    
    # Initial Parameters
    theta_init=(0.25, 2.5, 2.5),           # [ε, α, β] starting values
    # ε = 0.25: Realistic demand elasticity
    # α = 2.5: Moderate surge multiplier
    # β = 2.5: Moderate discount multiplier
    
    # Optimization Settings
    random_seed=42,                         # Reproducibility
    max_iterations=40,                      # Safety limit
    convergence_window=20,                  # Rolling window for convergence check
    
    # Learning Rate
    learning_rate_init=0.1,                 # Initial learning rate
    learning_rate_decay=0.01,               # Decay factor per iteration
)
```

### Customization Examples

**More aggressive pricing**:
```python
theta_init=(0.35, 3.0, 3.0)  # Higher elasticity and multipliers
```

**Conservative approach**:
```python
theta_init=(0.15, 2.0, 2.0)  # Lower elasticity and multipliers
pricing_bounds=(12.0, 18.0)  # Narrower price range
```

**Faster convergence**:
```python
max_iterations=30
convergence_window=15
learning_rate_init=0.15
```

---

## Performance Characteristics

### Runtime

- **Data preprocessing**: 10-30 seconds (depends on file I/O)
- **Demand Agent training**: 3-5 seconds (XGBoost on ~991 samples)
- **Per optimization step**: 2-3 seconds (includes LLM API calls)
- **Total optimization**: 2-3 minutes for 40 steps
- **End-to-end**: ~3-4 minutes (preprocessing + EDA + optimization)

### Resource Usage

- **Memory**: ~200-500 MB peak (data + models)
- **Disk**: ~10 MB (datasets + outputs)
- **CPU**: Single-core sufficient (XGBoost can use multiple cores)
- **Network**: LLM API calls (~2-4 KB per request)

### Scalability

- **Data size**: Current implementation handles up to ~10K samples efficiently
- **For larger datasets**: Consider batch processing or feature sampling
- **Iterations**: Linear time complexity with respect to max_iterations
- **LLM calls**: 2 calls per iteration (Pricing + Monitoring agents)

---

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError` when running scripts
```bash
# Solution: Ensure virtual environment is activated and dependencies installed
source .venv/bin/activate
pip install -r requirements.txt
```

**Issue**: `FileNotFoundError` for data files
```bash
# Solution: Run data preprocessing first
python rebuild_data.py
```

**Issue**: LLM API errors (rate limits, authentication)
```bash
# Solution: Check API key is set
echo $GROQ_API_KEY

# Or use fallback mode (set in config)
# The system will use rule-based fallbacks if LLM is unavailable
```

**Issue**: Low R² score or high RMSE in Demand Agent
```
# This can happen due to data variability
# The system will still function, but predictions may be less accurate
# Consider: (1) More training data, (2) Feature engineering, (3) Hyperparameter tuning
```

**Issue**: Parameters not converging
```
# This can happen if learning rate is too high or data has high variance
# Try: (1) Reduce learning_rate_init, (2) Increase convergence_window
# (3) Check for LLM availability (fallbacks may be less effective)
```

---

## License

Open Project 2026 - Society of Business

---

## Contact & Support

For questions or issues related to this project:

1. **Code questions**: Review inline comments in source files
2. **Setup issues**: Check the [Installation](#installation) and [Troubleshooting](#troubleshooting) sections
3. **Results interpretation**: Refer to [Evaluation Metrics](#evaluation-metrics) and [Results](#results) sections
4. **Testing**: See [Testing](#testing) section for validation procedures

---

## Acknowledgments

**Datasets**:
- ACN-Data: Caltech Adaptive Charging Network
- UrbanEV: Shenzhen EV Charging Infrastructure Dataset

**Technologies**:
- XGBoost: Gradient boosting framework
- Groq: LLM inference API
- Scikit-learn: Machine learning utilities

---

**Project Status**: Fully Operational  
**Code Quality**: Production-Ready  
**Documentation**: Complete

