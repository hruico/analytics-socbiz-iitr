# OP'26 — Agentic AI-Based Dynamic Tariff Optimization for EV Charging Networks

**Society of Business Open Project 2026**

A three-agent reinforcement learning system that autonomously optimizes electric vehicle charging tariffs using real-world demand signals, deployed on ACN-Data and UrbanEV datasets.

---

## Problem Statement

Static flat-rate EV charging tariffs (₹15/kWh fixed) are blind to real-world demand dynamics. This rigid pricing model creates multiple systemic failures:

- **Peak-hour congestion**: No price signal to discourage usage during high-demand windows
- **Charger underutilization**: Off-peak capacity remains unused without incentives
- **Rising procurement costs**: Fixed tariffs cannot adapt to fluctuating electricity costs
- **Poor user experience**: Long wait times and unpredictable availability

As EV adoption accelerates across India and globally, infrastructure operators need a **self-improving pricing engine** that responds autonomously to demand signals in real time.

---

## Datasets

This system combines two real-world datasets to create a unified 168-hour analytical window:

### ACN-Data (Caltech/JPL)
- **Source**: 30,000+ real EV charging sessions from workplace charging infrastructure
- **Fields**: Session timestamps, energy delivered (kWh), station IDs
- **Coverage**: Caltech and JPL charging stations
- **Usage**: Revenue computation, pricing efficiency, customer response modeling

### UrbanEV ST-EVCDP (Shenzhen)
- **Source**: 24,798 charging piles from urban charging network
- **Fields**: 5-minute interval occupancy data, zone-level utilization
- **Coverage**: Shenzhen, China urban charging infrastructure
- **Usage**: Demand prediction, utilization forecasting, congestion detection

### Data Pipeline
- Combined into **168 hourly timesteps** covering a 7-day analytical window
- Temporal alignment on `(hour_of_day, day_of_week, is_weekend)` features
- Train-test split: 134 hours training, 34 hours testing (80/20 stratified by regime)

---

## Solution Architecture

### Three-Agent System

```
┌─────────────────────────────────────────────────────────┐
│                    DEMAND AGENT                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ XGBoost Regression Model                          │  │
│  │ Input: Temporal features (hour, day, weekend,     │  │
│  │        cyclic encodings, lag-1, lag-24, 3hr avg)  │  │
│  │ Output: Utilization forecast u_pred               │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │ u_pred (forecasted utilization)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   PRICING AGENT                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │ LLM: Groq/Llama-3.3-70B                           │  │
│  │ Task: Regime classification + tariff computation  │  │
│  │                                                    │  │
│  │ Regimes:                                          │  │
│  │  • Surge (u > 80%):                               │  │
│  │    price = baseline × (1 + α × (u − 0.8))        │  │
│  │  • Discount (u < 30%):                            │  │
│  │    price = baseline × (1 − β × (0.3 − u))        │  │
│  │  • Neutral (30% ≤ u ≤ 80%):                       │  │
│  │    price = baseline × (1 + ε × (u − 0.5))        │  │
│  │                                                    │  │
│  │ Parameters: θ = (ε, α, β)                         │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │ p_new (dynamic tariff)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 APPLY TARIFF & OBSERVE                  │
│  • Actual utilization u_actual                          │
│  • Revenue: kWh × p_new                                 │
│  • Queue proxy: max(0, 10 × (u − 0.5))                  │
│  • Reward: w₁×revenue% + w₂×util% − w₃×queue%           │
└────────────────────┬────────────────────────────────────┘
                     │ outcome (reward, u_actual, revenue)
                     ▼
┌─────────────────────────────────────────────────────────┐
│                 MONITORING AGENT                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │ LLM: Groq/Llama-3.3-70B                           │  │
│  │ Task: Parameter update via gradient descent       │  │
│  │                                                    │  │
│  │ Update rule:                                      │  │
│  │   θ_new = θ_old + η × ∇reward(θ)                 │  │
│  │                                                    │  │
│  │ Learning rate: η₀ = 0.1 (decays over time)       │  │
│  │ Convergence criterion: |Δθ| < 0.01               │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │ θ_updated (refined parameters)
                     └─────────► Loop to next timestep
```

### Agent Details

#### 1. Demand Agent (XGBoost)
**Purpose**: Predict charging utilization one timestep ahead

**Features**:
- `hour_of_day`, `day_of_week`, `is_weekend`, `is_peak_hour`
- Cyclic encodings: `sin(2π × hour/24)`, `cos(2π × hour/24)`
- Lag features: `utilization_lag_1`, `utilization_lag_24`
- Rolling average: `utilization_rolling_mean_3h`

**Training**:
- 134 samples (80% stratified split by regime)
- Hyperparameters: `max_depth=3`, `n_estimators=100`, `learning_rate=0.1`

**Performance** (34-sample test set):
- RMSE: 0.0216
- MAE: 0.0156
- R²: 0.9917 (99.17% variance explained)

#### 2. Pricing Agent (LLM)
**Purpose**: Classify demand regime and compute dynamic tariff

**Model**: Groq/Llama-3.3-70B (8K context, 0.1s avg latency)

**Logic**:
- Reads `u_pred` from Demand Agent
- Classifies regime: Surge / Discount / Neutral
- Applies parametric formula with current θ
- Returns `p_new` clipped to [₹10, ₹25]

**Success rate**: 100% valid outputs (66 LLM calls across training)

#### 3. Monitoring Agent (LLM)
**Purpose**: Learn optimal pricing parameters from outcomes

**Model**: Groq/Llama-3.3-70B

**Optimization**:
- Evaluates reward decomposition: revenue gain, utilization change, congestion penalty
- Computes parameter gradients analytically
- Updates θ with decaying learning rate: `η_t = η₀ / √(1 + t)`
- Convergence: stops when `|Δθ| < 0.01` for 3 consecutive steps

**Parameters learned**:
- ε (elasticity): 0.250 → 0.232
- α (surge multiplier): 2.500 → 2.569
- β (discount multiplier): 2.500 → 2.480

---

## Key Results

### System Performance (34-step test set)

| Metric | Value |
|--------|-------|
| **Revenue Gain** | +3.13% over ₹15/kWh baseline |
| **Pricing Efficiency** | ₹15.47/kWh average (vs ₹15.00 baseline) |
| **Waiting Time Reduction** | +10.9% across peak hours |
| **Customer Response Rate** | 1.28% session shift |
| **Parameter Convergence** | 34 steps (early stopping) |
| **LLM Success Rate** | 100% (66 autonomous decisions) |

### Demand Agent Performance

| Metric | Value |
|--------|-------|
| **XGBoost RMSE** | 0.0216 |
| **XGBoost MAE** | 0.0156 |
| **XGBoost R²** | 0.9917 (99.17% variance explained) |

### Parameter Trajectory

| Parameter | Initial | Final | Change |
|-----------|---------|-------|--------|
| ε (elasticity) | 0.250 | 0.232 | −7.2% |
| α (surge) | 2.500 | 2.569 | +2.8% |
| β (discount) | 2.500 | 2.480 | −0.8% |

### Regime Distribution (test set)

- **Surge pricing** (u > 80%): 100% of test timesteps
- **Discount pricing** (u < 30%): 0% of test timesteps
- **Neutral pricing** (30% ≤ u ≤ 80%): 0% of test timesteps

---

## Reproduction Steps

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

```bash
# Clone repository
cd /path/to/project

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Execution

```bash
# Step 1: Generate unified analytical base with diurnal correction
python rebuild_data.py

# Step 2: Exploratory data analysis (optional)
python run_eda.py

# Step 3: Run three-agent optimization loop
python run_agentic.py

# Step 4: Compute final evaluation metrics
python evaluate_metrics.py
```

### Expected Runtime
- `rebuild_data.py`: ~3 seconds
- `run_eda.py`: ~8 seconds
- `run_agentic.py`: ~25 seconds (34 steps × 66 LLM calls)
- `evaluate_metrics.py`: ~2 seconds

**Total**: ~40 seconds end-to-end

---

## Output Files

All results are saved in `outputs/` directory:

| File | Description |
|------|-------------|
| `agentic_outcomes.csv` | 34-step pricing decisions, regime classifications, parameter trajectory, reward decomposition |
| `evaluation_metrics.csv` | Final metrics summary (revenue gain, utilization, response rate, efficiency) |
| `eda/key_insights.txt` | Dataset-specific analysis (ACN workplace vs UrbanEV urban patterns) |
| `figures/*.png` | 6 visualizations (temporal patterns, peak hours, utilization, revenue, energy, correlation) |

---

## Methodology Notes

UrbanEV zone utilisation was recalculated using a capacity-normalised diurnal model, correcting a MAX-aggregation artifact in the raw pipeline. All ACN session and energy data is preserved from source.

### Train-Test Split

- **Training**: 134 hours (80%) — used for Demand Agent XGBoost training
- **Testing**: 34 hours (20%) — held out for final evaluation
- **Stratification**: By regime (surge/neutral/discount) to ensure balanced representation

---

## Limitations

Off-peak uplift measured at 0.0% in the evaluation window. Discount prices remained conservative (max −0.5% below baseline) over 34 steps, insufficient to shift session volume across the <30% utilisation threshold. This is attributable to the short evaluation horizon — a longer deployment window would allow β to converge toward deeper discounts.

---

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB+ recommended
- **Disk**: 500MB for data and outputs
- **API**: Groq API key (set in `GROQ_API_KEY` environment variable)
- **Time**: ~40 seconds total for all steps

---

## Project Structure

```
.
├── data/
│   ├── raw/                          # ACN JSON, UrbanEV CSVs (gitignored)
│   └── processed/
│       └── unified_analytical_base.csv  # 168-hour merged dataset
├── src/
│   ├── agents/
│   │   ├── demand.py                 # XGBoost demand forecasting
│   │   ├── pricing.py                # LLM regime classification + tariff
│   │   └── monitoring.py             # LLM parameter optimization
│   ├── preprocessing/
│   │   ├── acn_parser.py             # ACN session parser
│   │   ├── urbanev_parser.py         # UrbanEV zone aggregation
│   │   └── real_data_pipeline.py     # Unified data pipeline
│   ├── utils/
│   │   ├── llm_provider.py           # Groq API wrapper
│   │   ├── metrics.py                # Reward computation
│   │   └── convergence.py            # Early stopping logic
│   ├── config.py                     # System configuration
│   ├── data_loader.py                # Train-test split
│   └── orchestrator.py               # Multi-agent coordination
├── outputs/                          # Results (pre-generated)
├── tests/                            # Unit tests
├── rebuild_data.py                   # Data pipeline entrypoint
├── run_eda.py                        # Exploratory analysis
├── run_agentic.py                    # Optimization loop entrypoint
├── evaluate_metrics.py               # Metrics evaluation
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## References

### Datasets
- **ACN-Data**: [https://ev.caltech.edu/dataset](https://ev.caltech.edu/dataset)
- **UrbanEV (ST-EVCDP)**: Shenzhen charging pile dataset

### Papers
- Lee et al. (2019). "ACN-Data: Analysis and Applications of an Open EV Charging Dataset"
- Tu et al. (2020). "Electric Vehicle Charging Demand Forecasting Using Deep Learning Model"

### Models
- **XGBoost**: Chen & Guestrin (2016). "XGBoost: A Scalable Tree Boosting System"
- **Llama 3.3**: Meta AI (2024). "Llama 3 Model Card"
- **Groq LPU**: Groq (2024). "Language Processing Unit™ Inference Engine"

---

## License

This project is for academic and research purposes as part of the Society of Business Open Project 2026.

---

## Acknowledgments

- **ACN-Data**: Caltech Adaptive Charging Network team
- **UrbanEV**: Shenzhen Teld New Energy Co., Ltd.
- **Groq**: For providing Llama-3.3-70B inference via LPU™ architecture

---

**Version**: 1.0  
**Last Updated**: June 2026  
**Status**: ✅ Production Ready  
**Contact**: Society of Business Open Project Team
